"""One-step controls with frozen real parameters and independent outer evaluation.

Callbacks take a tuple of functional adapter tensors. No real optimizer is
created or advanced. Meta-eval is used only to report candidates, never select.
"""
from __future__ import annotations
import random
import torch
from ._mp_opd_oracle import hard_max_partition, span_utility_table


def matched_partition(partition, seed):
    lengths = []
    cursor = 0
    for start, end in partition:
        if start != cursor or end <= start:
            raise ValueError("invalid reference partition")
        lengths.append(end - start)
        cursor = end
    random.Random(seed).shuffle(lengths)
    result, cursor = [], 0
    for length in lengths:
        result.append((cursor, cursor + length))
        cursor += length
    return tuple(result)


def partition_rates(base, weight, partition):
    rates = torch.zeros_like(base)
    cursor = 0
    for start, end in partition:
        if start != cursor or not start < end <= len(base):
            raise ValueError("invalid partition")
        rates[start:end] = base[start:end].sum() / weight[start:end].sum()
        cursor = end
    if cursor != len(base):
        raise ValueError("incomplete partition")
    return rates.detach()


class AtomWeighting(torch.nn.Module):
    """Sign-preserving multiplier control, identity at initialization.

    Multipliers average to one under token weights. This fixes mean multiplier,
    NOT signed/absolute credit mass or gradient norm; report these separately.
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(3, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, base, weight):
        rate = (base / weight).detach()
        features = torch.stack((rate, base.detach(), weight.detach().log()), -1)
        multiplier = torch.exp(2 * torch.tanh(self.net(features).squeeze(-1)))
        multiplier = multiplier / ((multiplier * weight).sum() / weight.sum())
        return rate * multiplier


def gradients(loss, params, create_graph=False):
    result = torch.autograd.grad(loss, params, retain_graph=True, create_graph=create_graph, allow_unused=True)
    return tuple(torch.zeros_like(p) if g is None else g for p, g in zip(params, result))


def norm(gs):
    return torch.stack([g.float().square().sum() for g in gs]).sum().sqrt()


def diagnostic(params, atom_nll, base, weight, select_loss, eval_loss,
               *, lr, max_span=4, seed=43, weighting=None, energy=None, energy_features=None,
               energy_temperature=1., training_payload=None, norm_controls=False):
    if lr <= 0 or max_span <= 0 or len(base) == 0:
        raise ValueError("positive lr, span and atom count required")
    if not (base.shape == weight.shape == atom_nll(params).shape):
        raise ValueError("atom dimensions differ")
    if not torch.isfinite(base).all() or not torch.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("invalid credits")
    snapshots = tuple(p.detach().clone() for p in params)
    nll = atom_nll(params)
    v = tuple(g.detach() for g in gradients(select_loss(params), params))
    # z_i = <grad log p_i, grad F_select>; no NLL/token double weighting.
    z = torch.stack([sum((g * x).sum() for g, x in zip(gradients(-h, params), v)) for h in nll]).detach()
    utilities, valid = span_utility_table(base, weight, z, max_span)
    if training_payload is not None:
        training_payload.update(utilities=utilities.detach() / weight.sum(), valid=valid.detach())
    oracle = hard_max_partition(utilities, valid).partition
    n = len(base)
    fixed = lambda length: tuple((i, min(i + length, n)) for i in range(0, n, length))
    partitions = {"atomic": fixed(1), "fixed2": fixed(2), "fixed4": fixed(4), "oracle": oracle}
    partitions["random_oracle_lengths"] = matched_partition(oracle, seed)
    denominator = weight.sum()
    gs = {name: gradients((partition_rates(base, weight, part) * nll).sum() / denominator, params)
          for name, part in partitions.items()}
    gs["skip"] = tuple(torch.zeros_like(p) for p in params)
    gs["atomic_half_lr"] = tuple(g * 0.5 for g in gs["atomic"])
    gs["atomic_quarter_lr"] = tuple(g * 0.25 for g in gs["atomic"])
    na, no = norm(gs["atomic"]), norm(gs["oracle"])
    gs["atomic_oracle_norm"] = tuple(g * no / na.clamp_min(1e-30) for g in gs["atomic"])
    gs["meta_sft"] = v
    if weighting is not None:
        rates = weighting(base, weight).detach()
        gs["learned_weighting"] = gradients((rates * nll).sum() / denominator, params)
    energy_metrics = None
    if energy is not None:
        from ._mp_opd_semimarkov import semi_markov_partition
        from ._mp_opd_credit import span_tables, expected_atom_rates, credit_conservation_residual
        with torch.no_grad():
            distribution = semi_markov_partition(energy(energy_features, max_span), temperature=energy_temperature, valid_mask=valid)
            _, _, span_rates, _ = span_tables(base, weight, max_span)
            rates = expected_atom_rates(distribution.marginals, span_rates).detach()
            residual = credit_conservation_residual(base, weight, rates)
        gs['learned_partition'] = gradients((rates * nll).sum() / denominator, params)
        energy_metrics = dict(entropy=float(distribution.entropy), expected_span_length=float(distribution.expected_span_length),
                             coverage_max_error=float(distribution.coverage_max_error), credit_residual=float(residual))
        if not torch.isfinite(rates).all() or float(distribution.coverage_max_error)>1e-3:
            raise RuntimeError('Invalid learned partition marginals')
    norm_valid = {}
    if norm_controls:
        for name in ('fixed2','fixed4','random_oracle_lengths','oracle','learned_weighting','learned_partition','meta_sft'):
            if name not in gs: continue
            size = norm(gs[name])
            ok = bool(torch.isfinite(size) and size > 0 and na > 0)
            key = name + '_atomic_norm'
            norm_valid[key] = ok
            gs[key] = tuple(g * na / size.clamp_min(1e-30) for g in gs[name])
    for alpha in (.25,.5,.75):
        gs[f"atomic_oracle_mix_{alpha}"] = tuple((1-alpha)*a+alpha*o for a,o in zip(gs["atomic"],gs["oracle"]))
    before_select, before_eval = float(select_loss(params).detach()), float(eval_loss(params).detach())
    report = {}
    for name, gradient in gs.items():
        updated = tuple((p - lr * g).detach() for p, g in zip(params, gradient))
        selected = float(select_loss(updated).detach())
        evaluated = float(eval_loss(updated).detach())
        report[name] = {
            "norm_match_valid": norm_valid.get(name),
            "select_nll": selected, "eval_nll": evaluated,
            "eval_improvement": before_eval - evaluated,
            "predicted_select_improvement": float(lr * sum((g * x).sum() for g, x in zip(gradient, v))),
            "gradient_norm": float(norm(gradient)),
            "virtual_update_norm": float(lr * norm(gradient)),
            "select_nll_change": before_select-selected,
            "eval_nll_change": before_eval-evaluated,
            "gradient_delta_from_atomic_norm": float(norm(tuple(g-a for g,a in zip(gradient,gs["atomic"])))),
            "rate_delta_from_atomic_norm": float((partition_rates(base,weight,partitions[name])-partition_rates(base,weight,partitions["atomic"])).norm()) if name in partitions else None,
            "partition": partitions.get(name),
        }
    candidates=["atomic","atomic_oracle_mix_0.25","atomic_oracle_mix_0.5","atomic_oracle_mix_0.75","oracle"]
    chosen=min(candidates,key=lambda name:report[name]["select_nll"])
    report["select_chosen_atomic_oracle_mix"]={**report[chosen],"chosen_control":chosen,"selection":"actual select NLL only; extra candidate-forward compute"}
    if not all(torch.equal(p.detach(), snap) for p, snap in zip(params, snapshots)):
        raise RuntimeError("diagnostic mutated real adapter")
    return {"all_controls_eval_unchanged": all(x["eval_nll"] == before_eval for x in report.values()),
            "meta_sft_select_unchanged": report["meta_sft"]["select_nll"] == before_select,
            "before_select": before_select, "before_eval": before_eval, "controls": report,
            "learned_partition": energy_metrics,
            "random_degenerate": partitions["random_oracle_lengths"] == oracle,
            "atomic_zero_norm": bool(na == 0), "parameters_unchanged": True,
            "normalization": "sum valid student tokens", "subspace": "declared functional adapter",
            "oracle_selection": "first-order select utility; eval never selects"}


def train_weighting_step(network, optimizer, params, atom_nll, base, weight, select_loss, lr):
    """Train only on B/M_select; caller must keep M_eval out of this callback."""
    optimizer.zero_grad(set_to_none=True)
    loss = (network(base, weight) * atom_nll(params)).sum() / weight.sum()
    grad = gradients(loss, params, create_graph=True)
    virtual = tuple(p - lr * g for p, g in zip(params, grad))
    outer = select_loss(virtual)
    wg = torch.autograd.grad(outer, tuple(network.parameters()))
    if not all(torch.isfinite(g).all() for g in wg):
        raise RuntimeError("nonfinite weighting hypergradient")
    for p, g in zip(network.parameters(), wg):
        p.grad = g
    optimizer.step()
    return float(outer.detach())
