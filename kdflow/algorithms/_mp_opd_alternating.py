"""Single-device, adapter-subspace bilevel SGD; independent of Ray/FSDP."""
import math
import torch
from ._mp_opd_credit import span_tables, expected_atom_rates
from ._mp_opd_semimarkov import semi_markov_partition


def meta_objective(energy, parameters, atom_nll, meta_nll, features, base, weight, *, lr, max_span, temperature):
    """Exact one-virtual-SGD-step objective; no real parameter mutation."""
    _, _, rates, valid = span_tables(base.detach(), weight.detach(), max_span)
    q = semi_markov_partition(energy(features.detach(), max_span), temperature=temperature, valid_mask=valid)
    pooled = expected_atom_rates(q.marginals, rates)
    inner = (pooled * atom_nll(parameters)).sum()
    gradients = torch.autograd.grad(inner, parameters, create_graph=True)
    virtual = tuple(p - lr * g for p, g in zip(parameters, gradients))
    return meta_nll(virtual), q


def alternating_step(energy, energy_optimizer, parameters, atom_nll, meta_nll, features, base, weight,
                     *, rollout_ids, meta_ids, lr, max_span, temperature=1., update_energy=True):
    """Update phi first, recompute detached rates, then exactly one real SGD step.

    Callbacks must use the supplied functional parameter tuple, response-only NLL,
    and deterministic forwards. Caller supplies disjoint prompt-identity hashes.
    This function deliberately does not approximate Adam or distributed updates.
    """
    parameters = tuple(parameters)
    if not rollout_ids or not meta_ids or set(rollout_ids) & set(meta_ids):
        raise ValueError('rollout/meta IDs must be nonempty and disjoint')
    if not parameters or not math.isfinite(lr) or lr <= 0 or max_span < 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('invalid alternating configuration')
    if base.numel() == 0 or not all(torch.isfinite(x).all() for x in (features,base,weight)):
        raise ValueError('empty or nonfinite atom data')
    owned = {id(p) for g in energy_optimizer.param_groups for p in g['params']}
    energy_params = tuple(energy.parameters())
    if owned != {id(p) for p in energy_params} or owned & {id(p) for p in parameters}:
        raise ValueError('energy optimizer ownership mismatch')
    if any(p.grad is not None for p in parameters):
        raise ValueError('pending student gradients; accumulation is unsupported')
    energy.eval()  # deterministic GRU, with autograd enabled
    before = [p.detach().clone() for p in energy_params]
    energy_norm = 0.
    objective_value = None
    if update_energy:
        objective, _ = meta_objective(energy,parameters,atom_nll,meta_nll,features,base,weight,
                                    lr=lr,max_span=max_span,temperature=temperature)
        grads = torch.autograd.grad(objective, energy_params)
        if not torch.isfinite(objective) or not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError('nonfinite energy objective/gradient')
        energy_optimizer.zero_grad(set_to_none=True)
        for p,g in zip(energy_params,grads):p.grad=g.detach()
        energy_norm=float(torch.stack([g.detach().square().sum() for g in grads]).sum().sqrt())
        energy_optimizer.step()
        energy_optimizer.zero_grad(set_to_none=True)
        objective_value=float(objective.detach())
    with torch.no_grad():
        _,_,rates,valid=span_tables(base.detach(),weight.detach(),max_span)
        q=semi_markov_partition(energy(features.detach(),max_span),temperature=temperature,valid_mask=valid)
        pooled=expected_atom_rates(q.marginals,rates).detach()
    loss=(pooled*atom_nll(parameters)).sum()
    grads=torch.autograd.grad(loss,parameters)
    if not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in grads):
        raise FloatingPointError('nonfinite student objective/gradient')
    with torch.no_grad():
        for p,g in zip(parameters,grads):p.add_(g,alpha=-lr)
    return {'student_loss':float(loss.detach()),'meta_virtual_nll':objective_value,
            'energy_gradient_norm':energy_norm,'energy_updates':int(update_energy),'student_updates':1,
            'energy_parameter_delta':float(sum((p.detach()-b).square().sum() for p,b in zip(energy_params,before)).sqrt()),
            'student_gradient_norm':float(sum(g.square().sum() for g in grads).sqrt()),
            'marginal_coverage_error':float(q.coverage_max_error),
            'credit_residual':float((weight*pooled).sum()-base.sum())}