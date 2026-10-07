#!/usr/bin/env python3
"""Span poolability diagnostics for atom credits.

The definitions below are taken verbatim from the method note (v1.2, the section
right after §9), and are quoted here so the code cannot drift from the spec:

    P_c          = ones(k,1) @ w[None,:] / W_c,   W_c = sum_{i in c} w_i
    C_c          = I - P_c                        (weighted centering / residual)
    S_c(alpha)   = I - alpha * C_c                (GRASS shrinkage operator)

    V_between(c) = (1 / W_c) * sum_{i in c} w_i (r_i - rbar_c)^2
    rbar_c       = w_c^T r_c / W_c

    V_within(c)  = (1 / W_c) * tr(C_c^T W_mat C_c Sigma_c),
                   W_mat = diag(w), Sigma_c = sigma2 * diag(1/w)
                = (k - 1) * sigma2 / W_c            (exact, see below)

    S_p(c)       = clip(V_within / (V_between + eps), 0, 1)
    R_p(c)       = (Rhat_c(0) - Rhat_c(alpha*)) / (tr(H_c Sigma_c) + eps)

``C_c`` is symmetric in the weighted inner product (``C^T W = W C``) and, because
``P^2 = 1 (w^T 1) w^T / W_c^2 = P``, it is idempotent for **any** positive ``w``.
Hence ``tr(C^T W C Sigma) = sigma2 * tr(W C W^-1) = sigma2 * tr(C) = (k-1) sigma2``
with no equal-weight assumption. ``C`` and ``S_c(alpha)`` are different operators
and are never interchanged here.

Noise scale: one global estimate for the whole method, computed from adjacent
normalised differences and robust-ified with MAD, then EMA'd in sample order:

    z_i         = (r_{i+1} - r_i) / sqrt(1/w_i + 1/w_{i+1})
    sigma_batch = 1.4826 * MAD(z)
    sigma2_t    = rho * sigma2_{t-1} + (1 - rho) * sigma2_batch

**Dimensional note, flagged rather than silently resolved:** ``Var(z_i) = sigma2``
under the model ``Var(r_i) = sigma2 / w_i``, so ``1.4826 * MAD(z)`` estimates
*sigma*, not *sigma^2*. Because ``Sigma_c = sigma2 * diag(1/w)`` needs a variance,
the operational value used here is ``sigma2 = (1.4826 * MAD(z))**2``. Both numbers
are emitted in the artifact (``sigma_robust`` and ``sigma2_robust``) so the
alternative reading can be checked without a rerun.

A per-span ``span_residual_scale`` is also logged for diagnosis only. It mixes
genuine between-atom signal with noise, so it never feeds ``V_within``, ``S_p``
or ``R_p``; the name deliberately avoids the word "noise".

The safety gate (``S_p >= 0.50``, ``R_p >= 0.02``, ``alpha* >= 0.10``) is
**logged, never enforced**: core GRASS keeps ``use_safety_gate=false``. ``R_p``
and ``alpha*`` are owned by GRASS, so unless the caller supplies them per span,
the gate row is recorded as ``None`` with a reason instead of being guessed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

MAD_SCALE = 1.4826
LOCAL_SHRINK_PSEUDOCOUNT = 4.0
GATE_SP_MIN = 0.50
GATE_RP_MIN = 0.02
GATE_ALPHA_MIN = 0.10


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = math.fsum(weights)
    if total <= 0.0:
        return math.fsum(values) / max(1, len(values))
    return math.fsum(v * w for v, w in zip(values, weights)) / total


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return 0.0
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def weighted_center_operator(weights: Sequence[float]) -> tuple[list[list[float]], float]:
    """``P = ones(k,1) @ w[None,:] / W_c`` and ``C = I - P`` as plain lists."""
    count = len(weights)
    total = math.fsum(weights)
    if count == 0:
        return [], 0.0
    projection = [[w / total for w in weights] for _ in range(count)]
    center = [[(1.0 if i == j else 0.0) - projection[i][j] for j in range(count)]
              for i in range(count)]
    return center, total


def weighted_center_trace(weights: Sequence[float]) -> float:
    """``tr(C) = k - tr(P) = k - 1``; kept separate to document the identity."""
    return float(max(0, len(weights) - 1))


def v_between(rates: Sequence[float], weights: Sequence[float]) -> float:
    """Weighted dispersion of observed atomic credit inside the span."""
    total = math.fsum(weights)
    if total <= 0.0 or len(rates) < 2:
        return 0.0
    mean = weighted_mean(rates, weights)
    return math.fsum(w * (r - mean) ** 2 for r, w in zip(rates, weights)) / total


def v_within(weights: Sequence[float], sigma2: Sequence[float]) -> float:
    """``(1/W_c) tr(C^T W C Sigma_c)`` with ``Sigma_c = diag(sigma2(i) / w_i)``.

    ``sigma2`` may be a per-atom sequence. When every entry is equal the result
    collapses to the old homoskedastic closed form ``(k-1) sigma2 / W_c``,
    which :func:`v_within_from_trace` is still used to cross-check. Under the
    position-dependent model the closed form is **not** used: it would ignore
    that ``tr(C^T W C Sigma) = sum_i (C^T W C)_ii sigma2(i) / w_i``.
    """
    center, total = weighted_center_operator(weights)
    count = len(weights)
    if count == 0 or total <= 0.0 or len(sigma2) != count:
        return 0.0
    trace = 0.0
    for i in range(count):
        diagonal = sum(center[j][i] * weights[j] * center[j][i] for j in range(count))
        trace += diagonal * sigma2[i] / weights[i]
    return trace / total


def v_within_from_trace(weights: Sequence[float], sigma2: float) -> float:
    """The general trace form, computed explicitly and divided by ``W_c``.

    Kept as a cross-check of :func:`v_within` rather than as the operational
    path: it is O(k^3) and only exists so a regression test can prove the
    closed form against the matrix expression it claims to equal.
    """
    center, total = weighted_center_operator(weights)
    count = len(weights)
    if count == 0 or total <= 0.0:
        return 0.0
    # W is diagonal, so (C^T W C)_ii = sum_j C_ji w_j C_ji, and Sigma is diagonal
    # too, so tr(C^T W C Sigma) = sum_i (C^T W C)_ii * sigma2 / w_i. Both use the
    # same index j twice; C is *not* symmetric (C^T = I - w 1^T / W), which is why
    # the squared factor must keep the j/i order rather than become w_i.
    trace = 0.0
    for i in range(count):
        diagonal = sum(center[j][i] * weights[j] * center[j][i] for j in range(count))
        trace += diagonal * sigma2 / weights[i]
    return trace / total


def poolability(v_btw: float, v_with: float, eps: float) -> float:
    if v_btw + eps <= 0.0:
        return 0.0
    return min(1.0, max(0.0, v_with / (v_btw + eps)))


def span_residual_scale(rates: Sequence[float], weights: Sequence[float]) -> float:
    """Diagnostic only — contains genuine signal, so it is not a noise scale."""
    degrees = len(rates) - 1
    if degrees <= 0 or math.fsum(weights) <= 0.0:
        return 0.0
    mean = weighted_mean(rates, weights)
    return math.fsum(w * (r - mean) ** 2 for r, w in zip(rates, weights)) / degrees


def robust_noise_estimate(rates: Sequence[float], weights: Sequence[float]) -> dict[str, float]:
    """``z_i`` from adjacent normalised differences, then a MAD scale."""
    if len(rates) < 2:
        return {'sigma_robust': 0.0, 'sigma2_robust': 0.0, 'pairs': 0.0}
    z_values = []
    for index in range(len(rates) - 1):
        w_left, w_right = weights[index], weights[index + 1]
        if w_left <= 0.0 or w_right <= 0.0:
            continue
        scale = 1.0 / w_left + 1.0 / w_right
        if scale <= 0.0:
            continue
        z_values.append((rates[index + 1] - rates[index]) / math.sqrt(scale))
    if not z_values:
        return {'sigma_robust': 0.0, 'sigma2_robust': 0.0, 'pairs': 0.0}
    sigma = MAD_SCALE * median([abs(value - median(z_values)) for value in z_values])
    return {'sigma_robust': sigma, 'sigma2_robust': sigma * sigma, 'pairs': float(len(z_values))}


def ema(previous: float, batch: float, rho: float) -> float:
    return rho * previous + (1.0 - rho) * batch


def local_sigma2(rates: Sequence[float], weights: Sequence[float],
                 window: int, global_sigma2: float) -> list[float]:
    """Per-atom ``sigma2_hat`` from a sliding window of adjacent differences.

    The window is centred on atom ``i`` and only ever sees pairs that are *not*
    inside a single candidate span's own estimate, so this stays independent of
    the span under test. Each local MAD is shrunk towards the global EMA with a
    pseudo-count of four pairs, because a short window's MAD is otherwise very
    noisy:

        sigma2_local(i) = (n * s2_local + 4 * sigma2_global) / (n + 4)

    ``global_sigma2`` is the position-independent EMA for this sample, so the
    method degrades to the old behaviour when the data really is homoskedastic.
    """
    count = len(rates)
    result = []
    half = max(1, window // 2)
    for i in range(count):
        # pairs j -> j+1 with |j - i| < half
        z_values = []
        for j in range(max(0, i - half), min(count - 1, i + half)):
            w_left, w_right = weights[j], weights[j + 1]
            if w_left <= 0.0 or w_right <= 0.0:
                continue
            scale = 1.0 / w_left + 1.0 / w_right
            if scale <= 0.0:
                continue
            z_values.append((rates[j + 1] - rates[j]) / math.sqrt(scale))
        if not z_values:
            result.append(global_sigma2)
            continue
        mad = median(z_values)
        local = (MAD_SCALE * median([abs(value - mad) for value in z_values])) ** 2
        pairs = float(len(z_values))
        if local <= 0.0:
            result.append(global_sigma2)
            continue
        # Shrink in log space, not linearly: sigma2 spans orders of magnitude
        # across a response (measured 19x), so a linear pull towards a global
        # value that is itself far off biases the local estimate towards the
        # wrong scale. The geometric form is scale-invariant.
        weight = pairs / (pairs + LOCAL_SHRINK_PSEUDOCOUNT)
        result.append(math.exp(weight * math.log(local)
                               + (1.0 - weight) * math.log(global_sigma2)))
    return result


def quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[position]


def describe(values: Sequence[float]) -> dict[str, float]:
    """Tail summary that can falsify the Gaussian-SURE model, not just report it."""
    usable = [v for v in values if math.isfinite(v)]
    if not usable:
        return {'count': 0}
    mean = math.fsum(usable) / len(usable)
    variance = math.fsum((v - mean) ** 2 for v in usable) / len(usable)
    centre = median(usable)
    mad = median([abs(v - centre) for v in usable])
    # Kurtosis against a robust scale; a tied/degenerate MAD falls back to the
    # standard deviation so a heavy tail is never silently reported as zero.
    scale = MAD_SCALE * mad
    if scale <= 0.0:
        scale = math.sqrt(variance)
    kurtosis = (math.fsum((v - centre) ** 4 for v in usable) / len(usable) / scale ** 4
                if scale > 0.0 else 0.0)
    return {
        'count': float(len(usable)),
        'mean': mean,
        'std': math.sqrt(variance),
        'median': centre,
        'mad': mad,
        'p90': quantile(usable, 0.90),
        'p95': quantile(usable, 0.95),
        'p99': quantile(usable, 0.99),
        'p999': quantile(usable, 0.999),
        'max': max(usable),
        'kurtosis_median_scaled': kurtosis,
    }


def standardized_residuals(rates: Sequence[float], weights: Sequence[float],
                           sigma2: Sequence[float]) -> list[float]:
    """``u_i = residual_i / sqrt(sigma2_hat(i) / w_i)``; residual is centred per span."""
    residual = [r - weighted_mean(rates, weights) for r in rates]
    result = []
    for value, w, s2 in zip(residual, weights, sigma2):
        denominator = math.sqrt(s2 / w) if w > 0.0 and s2 > 0.0 else float('nan')
        result.append(value / denominator if denominator > 0.0 else float('nan'))
    return result


def risk_reduction(risk_at_zero: float, risk_at_alpha: float,
                   trace_h_sigma: float, eps: float) -> float:
    """``R_p``; inputs come from GRASS, which owns ``H_c`` and ``Rhat_c``."""
    return (risk_at_zero - risk_at_alpha) / (trace_h_sigma + eps)


@dataclass(frozen=True)
class SpanRow:
    sample_index: int
    start: int
    length: int
    weight_total: float
    rate_mean: float
    v_between: float
    v_within: float
    s_p: float
    span_residual_scale: float
    r_sigma: float
    sigma2_min: float
    sigma2_max: float
    r_p: float | None
    alpha_star: float | None
    gate_pass: bool | None
    gate_missing: str

    def as_dict(self) -> dict[str, Any]:
        return {
            'sample_index': self.sample_index, 'start': self.start, 'length': self.length,
            'weight_total': self.weight_total, 'rate_mean': self.rate_mean,
            'v_between': self.v_between, 'v_within': self.v_within, 's_p': self.s_p,
            'span_residual_scale': self.span_residual_scale, 'r_sigma': self.r_sigma,
            'sigma2_min': self.sigma2_min, 'sigma2_max': self.sigma2_max,
            'r_p': self.r_p, 'alpha_star': self.alpha_star,
            'gate_pass': self.gate_pass, 'gate_missing': self.gate_missing,
        }


def sliding_spans(atom_count: int, max_span: int) -> Iterable[tuple[int, int]]:
    for length in range(2, min(max_span, atom_count) + 1):
        for start in range(0, atom_count - length + 1):
            yield start, length


def evaluate_sample(index: int, rates: Sequence[float], weights: Sequence[float],
                    sigma2: Sequence[float], max_span: int, eps: float,
                    risk_by_span: dict[tuple[int, int], dict[str, float]] | None,
                    gates: dict[str, float]) -> list[SpanRow]:
    rows: list[SpanRow] = []
    for start, length in sliding_spans(len(rates), max_span):
        span_rates = rates[start:start + length]
        span_weights = weights[start:start + length]
        span_sigma2 = sigma2[start:start + length]
        v_btw = v_between(span_rates, span_weights)
        v_with = v_within(span_weights, span_sigma2)
        residual_scale = span_residual_scale(span_rates, span_weights)
        mean_sigma2 = math.fsum(span_sigma2) / len(span_sigma2) if span_sigma2 else 0.0
        extras = (risk_by_span or {}).get((start, length), {})
        r_p = extras.get('r_p')
        alpha_star = extras.get('alpha_star')
        missing = ''
        if r_p is None or alpha_star is None:
            missing = 'r_p/alpha_star supplied by GRASS; gate not evaluated'
        gate = None
        if r_p is not None and alpha_star is not None:
            gate = (poolability(v_btw, v_with, eps) >= gates['s_p']
                    and r_p >= gates['r_p'] and alpha_star >= gates['alpha_star'])
        rows.append(SpanRow(
            sample_index=index, start=start, length=length,
            weight_total=math.fsum(span_weights),
            rate_mean=weighted_mean(span_rates, span_weights),
            v_between=v_btw, v_within=v_with,
            s_p=poolability(v_btw, v_with, eps),
            span_residual_scale=residual_scale,
            r_sigma=(residual_scale / mean_sigma2
                     if mean_sigma2 > 0.0 else float('inf')),
            sigma2_min=min(span_sigma2) if span_sigma2 else 0.0,
            sigma2_max=max(span_sigma2) if span_sigma2 else 0.0,
            r_p=r_p, alpha_star=alpha_star,
            gate_pass=gate, gate_missing=missing,
        ))
    return rows


def _finite(values: Sequence[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def summarise(rows: Sequence[SpanRow]) -> dict[str, Any]:
    s_values = _finite([row.s_p for row in rows])
    r_values = _finite([row.r_sigma for row in rows])
    by_length: dict[str, dict[str, Any]] = {}
    for length in sorted({row.length for row in rows}):
        subset = [row.s_p for row in rows if row.length == length and math.isfinite(row.s_p)]
        if subset:
            by_length[str(length)] = {'count': len(subset), 's_p_mean': math.fsum(subset) / len(subset),
                                      's_p_min': min(subset), 's_p_max': max(subset)}
    evaluated = [row for row in rows if row.gate_pass is not None]
    return {
        'spans': len(rows),
        'samples_with_2plus_atoms': len({row.sample_index for row in rows}),
        's_p_mean': (math.fsum(s_values) / len(s_values)) if s_values else None,
        's_p_min': min(s_values) if s_values else None,
        's_p_max': max(s_values) if s_values else None,
        'r_sigma_mean': (math.fsum(r_values) / len(r_values)) if r_values else None,
        'by_span_length': by_length,
        'gate_evaluated_spans': len(evaluated),
        'gate_pass_spans': sum(1 for row in evaluated if row.gate_pass),
        'gate_enforced': False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--atoms', required=True,
                        help='JSONL from the probe --dump-atoms flag')
    parser.add_argument('--max-span', type=int, default=4)
    parser.add_argument('--rho', type=float, default=0.9, help='EMA coefficient for sigma2')
    parser.add_argument('--sigma-window', type=int, default=11,
                        help='sliding window (in atoms) for the local robust sigma2')
    parser.add_argument('--noise-model', choices=('local', 'global'), default='local',
                        help='local = position-dependent sigma2 shrunk to the global EMA; '
                             'global = the earlier homoskedastic model, kept as a control')
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument('--gate-s-p', dest='gate_s_p', type=float, default=GATE_SP_MIN)
    parser.add_argument('--gate-r-p', dest='gate_r_p', type=float, default=GATE_RP_MIN)
    parser.add_argument('--gate-alpha', dest='gate_alpha', type=float, default=GATE_ALPHA_MIN)
    parser.add_argument('--risk', default=None,
                        help='optional JSON map "sample,start,length" -> {r_p, alpha_star}')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    gates = {'s_p': args.gate_s_p, 'r_p': args.gate_r_p, 'alpha_star': args.gate_alpha}
    risk_by_sample: dict[int, dict[tuple[int, int], dict[str, float]]] = {}
    if args.risk:
        loaded = json.loads(Path(args.risk).read_text(encoding='utf-8'))
        for sample_index, spans in loaded.items():
            bucket: dict[tuple[int, int], dict[str, float]] = {}
            for key, values in spans.items():
                start, length = (int(part) for part in key.split(','))
                bucket[(start, length)] = {k: float(v) for k, v in values.items()}
            risk_by_sample[int(sample_index)] = bucket

    rows: list[SpanRow] = []
    sigma2 = 0.0
    samples = 0
    degenerate = 0
    rate_pool: list[float] = []
    standardized_pool: list[float] = []
    sigma_ratio_pool: list[float] = []
    source = Path(args.atoms)
    with source.open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rates = [float(v) for v in record['rate']]
            weights = [float(v) for v in record['weight']]
            if len(rates) != len(weights):
                raise SystemExit('ANALYZE_FAIL: rate/weight length mismatch')
            estimate = robust_noise_estimate(rates, weights)
            sigma2 = ema(sigma2, estimate['sigma2_robust'], args.rho)
            samples += 1
            if sigma2 <= 0.0:
                degenerate += 1
            index = int(record.get('index', samples - 1))
            if args.noise_model == 'global':
                local = [sigma2] * len(rates)
            else:
                local = local_sigma2(rates, weights, args.sigma_window, sigma2)
            rate_pool.extend(rates)
            standardized_pool.extend(standardized_residuals(rates, weights, local))
            sigma_ratio_pool.extend(local[atom] / sigma2 if sigma2 > 0.0 else float('nan')
                                    for atom in range(len(rates)))
            rows.extend(evaluate_sample(
                index, rates, weights, local, args.max_span, args.eps,
                risk_by_sample.get(index), gates))

    payload = {
        'source': str(source), 'samples': samples, 'max_span': args.max_span,
        'rho': args.rho, 'eps': args.eps, 'sigma2_final': sigma2,
        'noise_model': args.noise_model, 'sigma_window': args.sigma_window,
        'sigma2_degenerate_samples': degenerate, 'gates': gates,
        'gate_enforced': False,
        'note_sigma_units': ('1.4826*MAD(z) estimates sigma; sigma2 = that squared'),
        'rate_distribution': describe(rate_pool),
        'standardized_residual_distribution': describe(standardized_pool),
        'sigma2_local_over_global': describe(sigma_ratio_pool),
        'summary': summarise(rows), 'spans': [row.as_dict() for row in rows],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print('ANALYZE_SAMPLES: %d  degenerate_sigma2: %d  sigma2_final: %.6g'
          % (samples, degenerate, sigma2), flush=True)
    print('ANALYZE_NOISE_MODEL: %s window=%d' % (args.noise_model, args.sigma_window),
          flush=True)
    summary = payload['summary']
    print('ANALYZE_SPANS: %d  S_p mean=%s  min=%s  max=%s'
          % (summary['spans'], summary['s_p_mean'], summary['s_p_min'], summary['s_p_max']),
          flush=True)
    for length, stats in summary['by_span_length'].items():
        print('ANALYZE_SPAN_LEN %s: count=%d S_p mean=%.4f min=%.4f max=%.4f'
              % (length, stats['count'], stats['s_p_mean'], stats['s_p_min'], stats['s_p_max']),
              flush=True)
    print('ANALYZE_R_SIGMA mean=%s' % summary['r_sigma_mean'], flush=True)
    ratio = payload['sigma2_local_over_global']
    if ratio.get('count'):
        print('ANALYZE_SIGMA_SPREAD: local/global median=%.3f p95=%.3f max=%.3f'
              % (ratio['median'], ratio['p95'], ratio['max']), flush=True)
    standardized = payload['standardized_residual_distribution']
    if standardized.get('count'):
        print('ANALYZE_U: median=%.3f MAD=%.3f p99=%.3f p99.9=%.3f max=%.3f kurtosis=%.2f'
              % (standardized['median'], standardized['mad'], standardized['p99'],
                 standardized['p999'], standardized['max'],
                 standardized['kurtosis_median_scaled']), flush=True)
    print('ANALYZE_GATE: enforced=false evaluated=%d pass=%d (log only)'
          % (summary['gate_evaluated_spans'], summary['gate_pass_spans']), flush=True)
    print('ANALYZE_ARTIFACT=%s' % out_path, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
