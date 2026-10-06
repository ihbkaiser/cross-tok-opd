"""GRASS: gradient-risk adaptive span shrinkage for MP-OPD.

GRASS replaces GBV-Span's hard broadcast inside a dynamically chosen span with a
*SURE-optimal partial shrinkage* of the atom credits, and it scores candidate
spans in the exact output-head gradient geometry instead of GBV's diagonal,
hand-weighted distortion/DoF surrogate. Everything else - the SimCT atoms, the
contiguous full-cover dynamic program, the OPD loss and the optimizer - is the
code that already ships.

For one response the module consumes

    r_i  = b_i / w_i            atomic credit rate (``credits.rate``)
    w_i                        atom token mass (``credits.weight``)
    H_ij                       exact banded LM-head atom Gram

and returns the shrunk rates ``r~`` that the training loss should multiply the
atom NLL by. Three claims from the method note are load bearing here and each is
pinned by a unit test in ``tests/mp_opd/test_grass_span.py``:

* ``H_ij`` is *exactly* ``<grad_W l_i, grad_W l_j>_F`` for the output head,
  including the elementwise derivative of Gemma-2 final logit softcapping, so
  ``delta_t`` cannot be replaced by a cheap proxy;
* the SURE-optimal shrinkage strength is the closed form
  ``alpha* = clip(V_c / D_c, 0, 1)`` with ``D_c`` the observed within-span credit
  heterogeneity measured in that geometry and ``V_c`` the part of it the noise
  model predicts;
* the shrinkage conserves weighted signed credit exactly, so the loss stays on
  the same credit budget as Atomic / Fixed / GBV and only the *resolution*
  changes.

Scope, stated plainly: this module implements sections 2-14 and 18 of the method
note plus the Tier-A/Tier-B scalar telemetry of section 17 that fits the existing
metrics channel. The Tier-C artifacts of section 17 (``span_samples.jsonl``,
``matrix_samples_stepXXXX.npz``, full-Transformer Gram calibration) are **not**
implemented; every value they would need is already returned here, so a later
patch can add them without touching the algorithm.

Numerical notes that are easy to get wrong later:

* the vocabulary inner product ``p_t^T p_s`` is computed over the *full*
  vocabulary. The method note lists no top-k approximation, and a top-k Gram
  would silently change ``D_c`` and therefore ``alpha*``;
* ``delta_t`` for a softcapped head is ``(p_t - e_y) * (1 - (z_t/sc)^2)``. That
  factor is the derivative of ``sc * tanh(u/sc)`` with respect to its
  pre-softcap logit ``u``, and ``tanh(u/sc) = z/sc`` is readable straight off the
  returned logits - no second ``[tokens, vocab] x [vocab, hidden]`` matmul through
  the head weight is needed;
* the Gram is accumulated in fp32 with TF32 explicitly disabled. Under TF32 the
  ~1e-3 relative error is larger than the fp32 round-off that ``D_c`` and ``V_c``
  are meant to resolve, and an exactly PSD head Gram starts producing materially
  negative ``D_c``;
* ``tr(H_c Sigma_c)`` does not depend on the partition, so the dynamic program
  uses the *excess* SURE cost ``C(c) = alpha^2 D_c - 2 alpha V_c`` and leaves
  singletons at exactly zero. ``D_c`` inside that expression is the zeroed one of
  section 14 - a ``D_c`` below the floor is replaced by zero rather than used as a
  small negative number - while the reported ``distortion`` table keeps the raw
  value, because a silently clamped diagnostic is how a broken Gram stays
  invisible;
* the dynamic program runs on host floats from one device->host copy of the cost
  table. The device version costs about ``2n`` synchronisations per response,
  which on a long response dominates everything else this module does.
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass
from typing import Sequence

import torch

from ..fused_logprob import _logsumexp_chunked

# Span-length fractions reported as telemetry, same reporting range as GBV-Span.
REPORTED_SPAN_LENGTHS = (1, 2, 3, 4)

_DEFAULT_ROW_CHUNK_ATOMS = 32
_ROW_CHUNK_FLAG = "MP_OPD_GRASS_ROW_CHUNK"
_DEFAULT_VOCAB_CHUNK = 16384
_VOCAB_CHUNK_FLAG = "MP_OPD_GRASS_VOCAB_CHUNK"
_EXACT_FP32_FLAG = "MP_OPD_GRASS_EXACT_FP32"

# Relative floor under which a negative ``D_c``/``V_c`` is float round-off from a
# PSD Gram rather than the implementation diagnostic the method note warns about.
PATHOLOGY_RELATIVE_TOL = 1e-6

# Credit-conservation guard of section 18.1, relative to the response's total
# weighted credit. The floor sits above fp32 rounding because the coefficients the
# loss actually multiplies by are the float32 rates the credit path already keeps.
CREDIT_CONSERVATION_REL_TOL = 1e-6


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name}={raw!r} must be an integer") from error
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}={raw!r} is not a boolean value")


def grass_row_chunk_atoms() -> int:
    """Row atoms per Gram accumulation launch, overridable per host.

    A memory/latency knob, not a recipe knob: it changes how the same ``H`` is
    computed, never which ``H``. Smaller chunks shrink the redundant work inside
    the column window but launch more kernels.
    """
    return _env_int(_ROW_CHUNK_FLAG, _DEFAULT_ROW_CHUNK_ATOMS)


def grass_vocab_chunk() -> int:
    """Vocabulary chunk used only when no selected log-prob is supplied."""
    return _env_int(_VOCAB_CHUNK_FLAG, _DEFAULT_VOCAB_CHUNK)


@contextlib.contextmanager
def _exact_fp32():
    """Force non-TF32 fp32 matmul inside the block and restore the global flag.

    ``torch.backends.cuda.matmul.allow_tf32`` is process-global, so the previous
    value is restored on the way out and on the exception path; nothing outside
    this block observes the change. Torch renamed the switch to
    ``fp32_precision = "ieee"``; both spellings are handled because the B200 image
    and a CPU-only local torch do not necessarily agree on which one exists.
    """
    if not _env_flag(_EXACT_FP32_FLAG, True):
        yield
        return
    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "fp32_precision"):
        previous = matmul.fp32_precision
        matmul.fp32_precision = "ieee"

        def restore() -> None:
            matmul.fp32_precision = previous

    else:
        previous = matmul.allow_tf32
        matmul.allow_tf32 = False

        def restore() -> None:
            matmul.allow_tf32 = previous

    try:
        yield
    finally:
        restore()


def _as_vector(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector")
    return value


def _symmetric_parts(rate: torch.Tensor, weight: torch.Tensor, max_span_length: int):
    """Validate the credit pair and return the float64 views the tables need."""
    rate = _as_vector("rate", rate)
    weight = _as_vector("weight", weight)
    if rate.shape != weight.shape:
        raise ValueError("rate and weight must share one shape")
    n = rate.numel()
    if n == 0:
        raise ValueError("at least one atom is required")
    if not torch.isfinite(rate).all():
        raise ValueError("rates must be finite")
    if not torch.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("weights must be finite and positive")
    length = min(int(max_span_length), n)
    if length < 1:
        raise ValueError("max_span_length must be positive")
    return rate.detach().to(torch.float64), weight.detach().to(torch.float64), n, length


# ---------------------------------------------------------------------------
# Section 10: robust credit-noise scale
# ---------------------------------------------------------------------------


class GrassNoiseEstimator:
    """MAD-of-adjacent-differences credit-noise scale with a bias-corrected EMA.

    ``z_i = (r_{i+1} - r_i) / sqrt(1/w_i + 1/w_{i+1})`` is standardised to unit
    scale wherever the latent credit is locally constant, so ``1.4826 * MAD(z)``
    is a robust estimate of the credit-noise standard deviation. Genuine credit
    change-points inflate the tails of ``z`` and the median ignores them, which is
    why this is a median and not a variance.

    The EMA is bias corrected (``ema / (1 - rho^t)``) so the first batch after a
    cold start does not report a deflated scale - a deflated scale is precisely
    the failure mode that collapses GRASS to atomic supervision.

    State is training-level (one estimator per run), not per-response: it is
    updated once per valid response inside the micro-batch loop and lives in the
    algorithm instance, so it survives across steps and across a resume.
    """

    def __init__(
        self,
        rho: float = 0.99,
        *,
        min_adjacent_pairs: int = 8,
        initial_variance: float = 0.0,
    ) -> None:
        rho = float(rho)
        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must lie in [0, 1)")
        if not math.isfinite(float(initial_variance)) or float(initial_variance) < 0:
            raise ValueError("initial_variance must be finite and nonnegative")
        self.rho = rho
        self.min_adjacent_pairs = int(min_adjacent_pairs)
        if self.min_adjacent_pairs < 1:
            raise ValueError("min_adjacent_pairs must be positive")
        self.initial_variance = float(initial_variance)
        self._ema = 0.0
        self._updates = 0

    @property
    def sigma2(self) -> float:
        """Bias-corrected ``sigma^2``; 0.0 until the first accepted batch."""
        if self._updates == 0:
            return self.initial_variance
        return self._ema / (1.0 - self.rho ** self._updates)

    def state_dict(self) -> dict:
        return {
            "rho": self.rho,
            "ema": self._ema,
            "updates": self._updates,
            "min_adjacent_pairs": self.min_adjacent_pairs,
            "initial_variance": self.initial_variance,
        }

    def load_state_dict(self, state: dict) -> None:
        if float(state.get("rho", self.rho)) != self.rho:
            raise ValueError("checkpointed GRASS rho does not match the configured rho")
        self._ema = float(state["ema"])
        self._updates = int(state["updates"])
        self.min_adjacent_pairs = int(
            state.get("min_adjacent_pairs", self.min_adjacent_pairs)
        )
        self.initial_variance = float(
            state.get("initial_variance", self.initial_variance)
        )

    def observe(
        self,
        rate: torch.Tensor,
        weight: torch.Tensor,
        same_group: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Batch diagnostics of section 17.3 for one response, without updating.

        ``same_group[i]`` marks the adjacency pair ``(i, i+1)`` as staying inside one
        statistical group. GRASS-DP has no such grouping; GRASS-Chunk must exclude
        every pair that crosses an alignment chunk boundary, because a chunk
        boundary is exactly where the latent credit is allowed to jump.
        """
        z = self._standardized_differences(rate, weight, same_group)
        if z.numel() == 0:
            return {
                "valid_pairs": 0.0,
                "sigma_batch": self.sigma2 ** 0.5,
                "sigma_batch_variance": self.sigma2,
                "z_abs_mean": 0.0,
                "z_abs_quantile": 0.0,
            }
        mad = 1.4826 * (z - z.median()).abs().median()
        return {
            "valid_pairs": float(z.numel()),
            "sigma_batch": float(mad),
            "sigma_batch_variance": float(mad * mad),
            "z_abs_mean": float(z.abs().mean()),
            "z_abs_quantile": float(z.abs().quantile(0.99)),
        }

    def update(
        self,
        rate: torch.Tensor,
        weight: torch.Tensor,
        same_group: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Fold one response's adjacent differences into the running scale.

        Returns the batch-level diagnostics of section 17.3. A batch with too few
        valid adjacent pairs cannot identify a scale, so the previous estimate is
        retained instead of folding sampling noise into the EMA.
        """
        diagnostics = self.observe(rate, weight, same_group)
        if int(diagnostics["valid_pairs"]) < self.min_adjacent_pairs:
            return diagnostics
        batch_variance = float(diagnostics["sigma_batch_variance"])
        if batch_variance <= 0.0:
            return diagnostics
        self._ema = self.rho * self._ema + (1.0 - self.rho) * batch_variance
        self._updates += 1
        return diagnostics

    @staticmethod
    def _standardized_differences(
        rate: torch.Tensor,
        weight: torch.Tensor,
        same_group: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r = _as_vector("rate", rate).detach().to(torch.float64)
        w = _as_vector("weight", weight).detach().to(torch.float64)
        if r.numel() != w.numel() or r.numel() < 2:
            return r.new_zeros(0)
        if (w <= 0).any() or not torch.isfinite(r).all() or not torch.isfinite(w).all():
            raise ValueError("rate and weight must be finite with positive weights")
        scale = (1.0 / w[:-1] + 1.0 / w[1:]).clamp_min(1e-300).sqrt()
        z = (r[1:] - r[:-1]) / scale
        if same_group is not None:
            if same_group.numel() != z.numel():
                raise ValueError("same_group must mark one flag per adjacent pair")
            z = z[same_group.detach().to(torch.bool)]
        return z[torch.isfinite(z)]


# ---------------------------------------------------------------------------
# Section 6: exact banded output-head atom Gram
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GrassHeadGram:
    """Banded atom Gram plus the numerical-quality read-out of section 17.5.

    ``gram`` is ``[n, n]`` with the band ``|i - j| <= L_max - 1`` filled and
    everything else exactly zero, so ``gram[a:b, a:b]`` is the exact ``H_c`` of
    that candidate span. ``diagonal_only`` reproduces the GRASS-Diag ablation of
    section 15 by zeroing the off-diagonal band.
    """

    gram: torch.Tensor
    diagonal_only: bool
    symmetry_error: float
    token_count: int
    atom_count: int


def _log_partition_from_selected(
    logits: torch.Tensor, labels: torch.Tensor, selected_log_prob: torch.Tensor
) -> torch.Tensor:
    """Recover ``log Z`` from ``log p(y)`` so the probabilities match the credit path.

    Both routes derive ``log Z`` from the selected logit, so this reproduces the
    production credit operator's probabilities rather than a second, slightly
    different normalisation.
    """
    selected_logit = logits.gather(1, labels.long().unsqueeze(1)).squeeze(1).to(torch.float64)
    return selected_logit - selected_log_prob.detach().to(torch.float64)


def _window_quantities(
    logits: torch.Tensor,
    log_partition: torch.Tensor,
    labels: torch.Tensor,
    softcap: float | None,
    token_weight: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q_t, v_t)`` for one token window: the scaled probability and label scale.

    ``q_t = p_t * m_t * (1 - (z_t/sc)^2)`` is the whole-vocabulary part of the
    head gradient row. The softcap factor is ``d/du [sc * tanh(u/sc)]`` and
    ``tanh(u/sc) = z/sc``, so it is recoverable from the returned logits alone -
    no second ``[tokens, vocab] x [vocab, hidden]`` matmul through the head weight
    is needed.

    ``v_t = m_t * (1 - (z_t[y_t]/sc)^2)`` is the *label* half: the ``e_y`` term of
    ``delta_t`` is scaled by the head Jacobian too, and it is scaled by ``s_t[y_t]``
    rather than by ``q_t[y_t]``. Conflating the two would put ``p_t[y_t]`` where
    ``1`` belongs and shift every diagonal block by ``-2 p_t[y_t] + 1``.
    """
    window = logits.to(torch.float32)
    probabilities = torch.exp(window - log_partition.to(torch.float32).unsqueeze(-1))
    weight = (
        torch.ones(window.shape[0], dtype=torch.float32, device=window.device)
        if token_weight is None
        else token_weight.to(torch.float32)
    )
    if softcap is not None:
        normalized = (window / float(softcap)).clamp(-1.0, 1.0)
        jacobian = (1.0 - normalized * normalized).clamp_min(0.0)
        probabilities = probabilities * jacobian
    else:
        # Same 2-D shape as the squashing branch on purpose: the label half below
        # gathers along the vocabulary axis, and a 1-D placeholder made that gather
        # raise IndexError on every unsquashed head - which is every head except
        # Gemma-2's. Probabilities are not multiplied by it, so nothing is paid.
        jacobian = torch.ones_like(window)
    probabilities = probabilities * weight.unsqueeze(-1)
    label_scale = weight * jacobian.gather(1, labels.unsqueeze(1)).squeeze(1)
    return probabilities, label_scale


def _delta_correction(
    probabilities: torch.Tensor,
    label_scale: torch.Tensor,
    column_labels: torch.Tensor,
    column_index: torch.Tensor,
    row_labels: torch.Tensor,
) -> torch.Tensor:
    """The part of ``delta_t^T delta_s`` that the stored ``q_t^T q_s`` does not carry.

    With ``delta_t = q_t - v_t e_{y_t}`` the identity is

        delta_t^T delta_s
            = q_t^T q_s - v_s q_t[y_s] - v_t q_s[y_t] + v_t v_s 1[y_t == y_s]

    so the vocabulary product alone is not ``delta^T delta``. Gathering ``q_t[y_s]``
    costs one ``[rows, window]`` gather rather than a second vocabulary pass.

    Dropping this term silently turns the Gram into a probability Gram, which
    understates the true logit-gradient geometry exactly on the confident tokens
    that dominate it: the diagonal block would come out as ``||p_t||^2`` instead of
    ``1 - 2 p_t[y_t] + p_t[y_t]^2``.
    """
    rows = row_labels.numel()
    columns = probabilities.shape[0]
    vocab = probabilities.shape[1]
    row_scale = label_scale[column_index].unsqueeze(-1)
    column_scale = label_scale.unsqueeze(0)
    # q_t[y_s] for every (row, column) pair: one gather across the window. The row
    # axis is addressed through column_index, not through 0..rows-1 - the two agree
    # only for a chunk whose first row is the window's first row, and using the
    # window position there left every chunk but the first one wrong.
    at_column_labels = probabilities[
        column_index.unsqueeze(-1), column_labels.unsqueeze(0).expand(rows, -1)
    ]
    # q_s[y_t]: a flat gather at (the window position of column s, the label of row t).
    # Both operands must broadcast to the full [rows, window] block - indexing by the
    # row's own position instead would collapse this term onto the diagonal and leave
    # every off-diagonal block wrong.
    window_position = torch.arange(columns, device=probabilities.device)
    at_row_labels = probabilities.reshape(-1).gather(
        0,
        (window_position.unsqueeze(0) * vocab + row_labels.unsqueeze(-1))
        .expand(rows, columns)
        .reshape(-1),
    ).view(rows, columns)
    same_label = row_labels.unsqueeze(-1) == column_labels.unsqueeze(0)
    # Each half of the correction carries the scale of the operand whose label it
    # reads: -v_s q_t[y_s] - v_t q_s[y_t]. Pairing them the other way round is
    # invisible whenever every token weight is equal, which is the only case the
    # earlier suite exercised, and wrong by a factor of m_t/m_s as soon as two atoms
    # carry different weights.
    return (
        -(column_scale * at_column_labels + row_scale * at_row_labels)
        + row_scale * column_scale * same_label
    )


def _atom_sums(values: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """Sum each atom's token rows into one ``[atoms, rows]`` block.

    ``bounds`` is the ``[atoms + 1]`` prefix-token-count vector, so atom ``i``
    owns rows ``bounds[i]:bounds[i+1]``.
    """
    stacked = torch.cat(
        (values.new_zeros((1, values.shape[1])), values.cumsum(0)), dim=0
    )
    index = bounds.to(torch.int64)
    return stacked[index[1:]] - stacked[index[:-1]]


def atom_head_gram(
    logits: torch.Tensor,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    atom_ranges: Sequence[tuple[int, int]],
    max_span_length: int,
    *,
    selected_log_prob: torch.Tensor | None = None,
    softcap: float | None = None,
    token_weight: torch.Tensor | None = None,
    head_bias: bool = False,
    diagonal_only: bool = False,
    row_chunk_atoms: int | None = None,
    vocab_chunk: int | None = None,
) -> GrassHeadGram:
    """Exact banded ``H_ij = <grad_W l_i, grad_W l_j>_F`` of the student LM head.

    ``logits``/``hidden``/``labels`` are the atom-covered student token rows in
    the credit path's order. ``hidden`` must be the *actual* input of the output
    head - the post-norm decoder state - not a pooled or pre-norm one.

    Only atom pairs that can co-occur inside a candidate span are evaluated: row
    atoms are processed in chunks, the column window is widened by ``L_max - 1``
    atoms on each side, and the in-band blocks are added into one flat
    accumulator. Every output entry is written by exactly one chunk (row atoms are
    partitioned across chunks), so the accumulation is a deterministic sum rather
    than an atomic race, and the accumulator's extra slot absorbs out-of-band
    entries without a host-side compaction.
    """
    if logits.ndim != 2:
        raise ValueError("student logits must be [tokens,vocab]")
    if hidden.ndim != 2:
        raise ValueError("student hidden states must be [tokens,hidden]")
    tokens, vocab = logits.shape
    if hidden.shape[0] != tokens:
        raise ValueError("hidden states and logits must cover the same token axis")
    if labels.numel() != tokens:
        raise ValueError("labels must align with the token axis")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= vocab):
        raise ValueError("labels must be within [0, vocab)")
    if not atom_ranges:
        raise ValueError("at least one atom range is required")
    cursor = 0
    for start, end in atom_ranges:
        if start != cursor:
            raise ValueError(
                "atom ranges must tile the token axis from 0 without gaps; GRASS "
                "resolves each atom's head gradient over exactly its own tokens"
            )
        if not start < end <= tokens:
            raise ValueError("atom ranges must stay inside the token axis")
        cursor = end
    covered = cursor
    if covered == 0:
        raise ValueError("atom ranges must cover at least one token")
    n = len(atom_ranges)
    length = min(int(max_span_length), n)
    if length < 1:
        raise ValueError("max_span_length must be positive")
    chunk_atoms = int(row_chunk_atoms) if row_chunk_atoms else grass_row_chunk_atoms()

    if selected_log_prob is not None:
        selected_log_prob = _as_vector("selected_log_prob", selected_log_prob)
        if selected_log_prob.numel() != covered:
            raise ValueError("selected_log_prob must align with the atom-covered token axis")
    elif int(vocab_chunk or grass_vocab_chunk()) < 1:
        raise ValueError("vocab_chunk must be positive")
    if token_weight is not None:
        token_weight = _as_vector("token_weight", token_weight)
        if token_weight.numel() != covered:
            raise ValueError("token_weight must align with the atom-covered token axis")

    slot_list = [0]
    for _start, end in atom_ranges:
        slot_list.append(int(end))
    slots = torch.tensor(slot_list, dtype=torch.int64, device=logits.device)
    atom_index = torch.arange(n, device=logits.device)
    covered_logits = logits[:covered].detach()
    covered_labels = labels[:covered].detach().long()

    # The Gram is a *selector* input, never a differentiable quantity: letting it
    # carry a graph would backpropagate the variance estimate into the student.
    with torch.no_grad(), _exact_fp32():
        if selected_log_prob is not None:
            log_partition = _log_partition_from_selected(
                covered_logits, covered_labels, selected_log_prob
            )
        else:
            chunk = int(vocab_chunk) if vocab_chunk else grass_vocab_chunk()
            log_partition = _logsumexp_chunked(covered_logits, chunk).to(torch.float64)
        accumulator = torch.zeros(n * n + 1, dtype=torch.float32, device=logits.device)
        dump = n * n
        hidden_fp32 = hidden[:covered].detach().to(torch.float32)
        for first_atom in range(0, n, chunk_atoms):
            last_atom = min(first_atom + chunk_atoms, n)
            window_first = max(0, first_atom - (length - 1))
            window_last = min(n, last_atom + (length - 1))
            low, high = slot_list[window_first], slot_list[window_last]
            row_low, row_high = slot_list[first_atom], slot_list[last_atom]
            probabilities, label_scale = _window_quantities(
                covered_logits[low:high],
                log_partition[low:high],
                covered_labels[low:high],
                softcap,
                None if token_weight is None else token_weight[low:high],
            )
            hidden_window = hidden_fp32[low:high]
            rows = probabilities[row_low - low : row_high - low]
            hidden_rows = hidden_window[row_low - low : row_high - low]
            window_labels = covered_labels[low:high]
            column_index = torch.arange(
                row_low - low, row_high - low, device=logits.device
            )
            vocabulary_gram = rows @ probabilities.T + _delta_correction(
                probabilities,
                label_scale,
                window_labels,
                column_index,
                window_labels[row_low - low : row_high - low],
            )
            block = vocabulary_gram * (hidden_rows @ hidden_window.T)
            if head_bias:
                # A head bias adds <sum_t delta_t, sum_s delta_s> to every block.
                # The term is constant across the token block, so per-atom sums
                # expanded back to token rows reproduce it exactly. Bounds are made
                # window-local because both operands start at their own slice.
                delta = probabilities.scatter(
                    1, window_labels.unsqueeze(1), -label_scale.unsqueeze(1)
                )
                row_bounds = slots[first_atom : last_atom + 1] - row_low
                window_bounds = slots[window_first : window_last + 1] - low
                atom_rows = _atom_sums(delta[row_low - low : row_high - low], row_bounds).repeat_interleave(
                    row_bounds[1:] - row_bounds[:-1], dim=0
                )
                atom_columns = _atom_sums(delta, window_bounds).repeat_interleave(
                    window_bounds[1:] - window_bounds[:-1], dim=0
                )
                block = block + atom_rows @ atom_columns.T
            row_counts = (slots[first_atom + 1 : last_atom + 1] - slots[first_atom:last_atom])
            column_counts = (
                slots[window_first + 1 : window_last + 1] - slots[window_first:window_last]
            )
            row_atoms = torch.repeat_interleave(atom_index[first_atom:last_atom], row_counts)
            column_atoms = torch.repeat_interleave(
                atom_index[window_first:window_last], column_counts
            )
            flat_index = row_atoms.unsqueeze(-1) * n + column_atoms.unsqueeze(0)
            band = (column_atoms.unsqueeze(0) - row_atoms.unsqueeze(-1)).abs() <= (length - 1)
            if diagonal_only:
                band = band & (row_atoms.unsqueeze(-1) == column_atoms.unsqueeze(0))
            target = torch.where(
                band, flat_index, torch.full_like(flat_index, dump)
            )
            accumulator.index_add_(0, target.reshape(-1), block.reshape(-1))

    gram = accumulator[:dump].view(n, n)
    symmetry_error = float(gram.sub(gram.T).abs().max() / gram.abs().max().clamp_min(1e-300))
    gram = (gram + gram.T) * 0.5
    return GrassHeadGram(
        gram=gram,
        diagonal_only=bool(diagonal_only),
        symmetry_error=symmetry_error,
        token_count=int(covered),
        atom_count=n,
    )


# ---------------------------------------------------------------------------
# Sections 7-9: candidate-span SURE statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GrassSpanCosts:
    """Per-candidate GRASS tables and the pathology tolerance of section 17.7.

    Every table has shape ``[n, L]`` indexed ``[start, length - 1]``.
    ``distortion`` and ``variance`` hold the **raw** ``D_c`` and ``V_c`` as
    computed; clipping ``alpha`` never rewrites them, because a silently clamped
    diagnostic is exactly how a broken Gram stays invisible.
    """

    costs: torch.Tensor
    valid: torch.Tensor
    alpha: torch.Tensor
    distortion: torch.Tensor
    variance: torch.Tensor
    trace: torch.Tensor
    sigma2: float
    eps_d: float
    negative_tol: float
    degenerate: bool


def _shrunk_strength(
    distortion: torch.Tensor, variance: torch.Tensor, eps_d: float
) -> torch.Tensor:
    """``clip(V/D, 0, 1)`` with the documented degenerate branches of section 9.

    The division is only evaluated where ``D > eps_d``; elsewhere the branch is
    ``1`` when ``V > 0`` (no observed heterogeneity but real predicted variance)
    and ``0`` otherwise, so a span that is constant *and* noise-free stays atomic
    instead of collapsing onto a 0/0.
    """
    usable = distortion > eps_d
    safe = torch.where(usable, distortion, torch.ones_like(distortion))
    ratio = (variance / safe).clamp(0.0, 1.0)
    return torch.where(
        usable, ratio, (variance > 0).to(distortion.dtype)
    )


def grass_span_costs(
    rate: torch.Tensor,
    weight: torch.Tensor,
    gram: torch.Tensor,
    sigma2: float,
    max_span_length: int,
    *,
    eps_d: float = 0.0,
    negative_tol_rel: float = PATHOLOGY_RELATIVE_TOL,
) -> GrassSpanCosts:
    """SURE-excess cost of every candidate span, vectorised over start atoms.

    With ``P_c = 1 w^T / W`` and ``A_c = I - P_c`` both statistics of the method
    note have closed forms that need no per-span matrix product:

        D_c = (r - rbar*1)^T H_c (r - rbar*1)
        V_c = sigma^2 * ( sum_i H_ii / w_i  -  (1^T H_c 1) / W_c )
            = tr(H_c Sigma_c) - sigma^2 * (1^T H_c 1) / W_c

    The second identity follows from ``A_c Sigma_c = sigma^2 (diag(1/w) - 11^T/W)``,
    which is PSD by Cauchy-Schwarz. So with an exact PSD head Gram ``V_c >= 0``
    holds by construction and a materially negative value is an implementation or
    noise-model failure, never a quantity to tune away.
    """
    r, w, n, length = _symmetric_parts(rate, weight, max_span_length)
    if gram.ndim != 2 or gram.shape[0] != n or gram.shape[1] != n:
        raise ValueError("gram must be [n,n] and align with the credit vectors")
    if not torch.isfinite(gram).all():
        raise ValueError("the head Gram must be finite")
    sigma2 = float(sigma2)
    if not math.isfinite(sigma2) or sigma2 < 0:
        raise ValueError("sigma2 must be finite and nonnegative")
    eps_d = float(eps_d)
    if eps_d < 0 or not math.isfinite(eps_d):
        raise ValueError("eps_d must be finite and nonnegative")
    negative_tol_rel = float(negative_tol_rel)
    if negative_tol_rel < 0 or not math.isfinite(negative_tol_rel):
        raise ValueError("negative_tol_rel must be finite and nonnegative")

    h = gram.detach().to(torch.float64)
    diagonal = torch.diagonal(h).contiguous()
    weight_prefix = torch.cat((h.new_zeros(1), w.cumsum(0)))
    weighted_rate_prefix = torch.cat((h.new_zeros(1), (w * r).cumsum(0)))

    costs = r.new_full((n, length), float("inf"))
    alpha_table = r.new_zeros((n, length))
    distortion_table = r.new_zeros((n, length))
    variance_table = r.new_zeros((n, length))
    trace_table = r.new_zeros((n, length))
    valid = torch.zeros((n, length), dtype=torch.bool, device=r.device)
    offsets = torch.arange(length, device=r.device).view(1, -1)

    for offset in range(length):
        width = offset + 1
        count = n - offset
        if count <= 0:
            continue
        starts = torch.arange(count, device=r.device)
        span_index = starts.unsqueeze(-1) + offsets[:, :width]
        span_weight = weight_prefix[span_index + 1] - weight_prefix[span_index]
        span_rate = (
            weighted_rate_prefix[span_index + 1] - weighted_rate_prefix[span_index]
        ) / span_weight
        deviation = r[span_index] - span_rate
        block = h[span_index.unsqueeze(-1), span_index.unsqueeze(-2)]
        distortion = (deviation.unsqueeze(-1) * block * deviation.unsqueeze(-2)).sum(
            dim=(1, 2)
        )
        trace = sigma2 * (diagonal[span_index] / w[span_index]).sum(dim=1)
        # tr(H_c A_c Sigma_c) with A_c = I - 1 w^T/W_c and Sigma_c = sigma^2 diag(1/w)
        # is sigma^2 [sum_i H_ii/w_i - (1/W_c) sum_ij H_ij], and that entry sum runs
        # over the span on *both* axes. Summing the rows alone folds in the entries
        # pointing at atoms outside the span, which a candidate Gram leaves nonzero
        # whenever its band reaches past the span.
        cross = block.sum(dim=(1, 2))
        variance = trace - sigma2 * cross / span_weight.sum(dim=1)
        strength = _shrunk_strength(distortion, variance, eps_d)
        # Section 14: the cost is built from the *zeroed* D, not the raw one, so
        # costs are reproducible from the reported table. A D that is negative or
        # below the floor is not silently used as a small negative number.
        usable_distortion = torch.where(
            distortion > eps_d, distortion, torch.zeros_like(distortion)
        )

        distortion_table[:count, offset] = distortion
        variance_table[:count, offset] = variance
        alpha_table[:count, offset] = strength
        trace_table[:count, offset] = trace
        costs[:count, offset] = (
            strength * strength * usable_distortion - 2.0 * strength * variance
        )
        valid[:count, offset] = True

    trace_max = float(trace_table.max()) if bool(valid.any()) else 0.0
    return GrassSpanCosts(
        costs=costs,
        valid=valid,
        alpha=alpha_table,
        distortion=distortion_table,
        variance=variance_table,
        trace=trace_table,
        sigma2=sigma2,
        eps_d=eps_d,
        negative_tol=negative_tol_rel * trace_max,
        degenerate=sigma2 <= 0.0,
    )


# ---------------------------------------------------------------------------
# Sections 11 and 14: dynamic programming and soft shrinkage
# ---------------------------------------------------------------------------


def grass_partition(
    tables: GrassSpanCosts,
) -> tuple[tuple[tuple[int, int], ...], torch.Tensor, torch.Tensor]:
    """Minimum-excess-risk contiguous partition with its selection margins.

    Returns ``(partition, total_cost, margins)`` where ``margins[j]`` is
    ``best - second best`` over the admissible predecessors at position ``j``.
    A near-zero margin means the segmentation is a coin flip on costs that
    themselves carry sampling noise - the section 17.8 signature of unstable
    boundaries. It is reported, never used to change the decision.
    """
    n, length = tables.costs.shape
    if n == 0:
        empty = tables.costs.new_zeros(())
        return (), empty, tables.costs.new_zeros((1,))
    # The dynamic program runs on host floats from one device->host copy of the
    # cost table. The obvious device version is ~2n synchronisations - one per
    # (position, span length) for the validity test and one per position for
    # argmax - which on a 1000-atom response is a thousand round trips and is the
    # dominant per-sample cost in the whole mode. The transfer is O(nL) floats,
    # which is the same order as the span table itself.
    cost_columns = [
        tables.costs.detach()[:, offset].to("cpu").tolist() for offset in range(length)
    ]
    valid_columns = [
        tables.valid.detach()[:, offset].to("cpu").tolist() for offset in range(length)
    ]
    best = [math.inf] * (n + 1)
    best[0] = 0.0
    margin = [math.inf] * (n + 1)
    margin[0] = 0.0
    back = [-1] * (n + 1)
    for end in range(1, n + 1):
        winner = math.inf
        runner_up = math.inf
        winner_start = -1
        for span_length in range(1, min(length, end) + 1):
            start = end - span_length
            if not valid_columns[span_length - 1][start]:
                continue
            total = best[start] + cost_columns[span_length - 1][start]
            if total < winner:
                runner_up, winner, winner_start = winner, total, start
            elif total < runner_up:
                runner_up = total
        if winner_start < 0:
            continue
        best[end] = winner
        back[end] = winner_start
        # Ascending span length with a strict comparison makes ties fall to the
        # shortest admissible span, matching the sibling gbv_partition rule.
        if runner_up < math.inf:
            margin[end] = winner - runner_up
    if back[n] < 0 or not math.isfinite(best[n]):
        raise ValueError("no valid full-cover partition")
    parts = []
    end = n
    while end:
        start = back[end]
        parts.append((start, end))
        end = start
    parts.reverse()
    return (
        tuple(parts),
        tables.costs.new_tensor(best[n]),
        torch.tensor(margin, dtype=torch.float64, device=tables.costs.device),
    )


def _partition_index(
    partition: Sequence[tuple[int, int]], n: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(span lengths, span id per atom, span start columns)`` for one partition.

    Built from host-side Python integers, so it costs no device synchronisation:
    the partition itself comes out of the dynamic program, which never needed a
    device value to decide.
    """
    cursor = 0
    lengths: list[int] = []
    for start, end in partition:
        if start != cursor or not start < end <= n:
            raise ValueError("partition must cover the atoms once, contiguously, in order")
        lengths.append(end - start)
        cursor = end
    if cursor != n:
        raise ValueError("partition does not cover all atoms")
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64, device=device)
    span_ids = torch.repeat_interleave(
        torch.arange(len(lengths), device=device), lengths_tensor
    )
    starts_tensor = torch.tensor(
        [start for start, _end in partition], dtype=torch.int64, device=device
    )
    return lengths_tensor, span_ids, starts_tensor


def grass_shrink(
    rate: torch.Tensor,
    weight: torch.Tensor,
    partition: Sequence[tuple[int, int]],
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``r~_i = (1 - alpha_c) r_i + alpha_c rbar_c`` and report conservation.

    ``alpha`` is indexed ``[start, length - 1]``, the same layout as the cost
    tables, so the strength the loss uses is exactly the one the objective chose
    for that span. The returned residual is the largest per-span violation of
    ``sum_i w_i r~_i == sum_i w_i r_i``, which section 18.1 makes mandatory.

    The whole shrink is one vectorised pass: reading the selected strengths one
    span at a time would cost one device synchronisation per span, which on a
    thousand-atom response is the dominant cost of the selector.
    """
    r, w, n, _length = _symmetric_parts(rate, weight, int(rate.numel()))
    if alpha.ndim != 2 or alpha.shape[0] != n:
        raise ValueError("alpha must be [n,L] and align with the credit vectors")
    lengths, span_ids, starts = _partition_index(partition, n, r.device)
    if lengths.numel() == 0:
        return r.clone(), r.new_zeros(())
    strengths = alpha[starts, lengths - 1]
    if not bool(((strengths >= 0.0) & (strengths <= 1.0)).all()):
        raise ValueError("the selected shrinkage strength must lie in [0, 1]")
    span_count = lengths.numel()
    span_weight = torch.zeros(span_count, dtype=torch.float64, device=r.device).index_add_(
        0, span_ids, w
    )
    weighted_rate = torch.zeros(span_count, dtype=torch.float64, device=r.device).index_add_(
        0, span_ids, w * r
    )
    pooled = (weighted_rate / span_weight)[span_ids]
    per_atom = strengths[span_ids]
    shrunk = (1.0 - per_atom) * r + per_atom * pooled
    # Conservation is exact in exact arithmetic: the shrink only redistributes a
    # span's credit, so the per-span weighted sum of the shift must vanish.
    shift = torch.zeros(span_count, dtype=torch.float64, device=r.device).index_add_(
        0, span_ids, w * (shrunk - r)
    )
    return shrunk, shift.abs().max()


def grass_credit_residual(
    rate: torch.Tensor, weight: torch.Tensor, shrunk: torch.Tensor
) -> torch.Tensor:
    """``|sum_i w_i r~_i - sum_i w_i r_i|`` for a whole response."""
    r, w, _n, _length = _symmetric_parts(rate, weight, int(rate.numel()))
    if shrunk.shape != r.shape:
        raise ValueError("shrunk credits must align with the atomic credits")
    return (w * shrunk.to(torch.float64) - w * r).sum().abs()


def grass_span_ids(
    partition: Sequence[tuple[int, int]], n: int, device: torch.device
) -> torch.Tensor:
    """Span index of every atom under ``partition``; used to mask the Gram."""
    lengths, span_ids, _starts = _partition_index(partition, n, device)
    return span_ids


def hard_pooled_credits(
    base: torch.Tensor, weight: torch.Tensor, partition: Sequence[tuple[int, int]]
) -> torch.Tensor:
    """Broadcast one pooled rate inside each span - the GBV / Fixed-k coefficients.

    Kept here so the shadow baselines of section 17.12 are built by the same code
    the compared modes use, not by a re-derivation that could drift from it.
    """
    r, w, n, _length = _symmetric_parts(base / weight, weight, int(base.numel()))
    lengths, span_ids, _starts = _partition_index(partition, n, r.device)
    if lengths.numel() == 0:
        return r.clone()
    span_count = lengths.numel()
    span_weight = torch.zeros(span_count, dtype=torch.float64, device=r.device).index_add_(0, span_ids, w)
    weighted = torch.zeros(span_count, dtype=torch.float64, device=r.device).index_add_(0, span_ids, w * r)
    return (weighted / span_weight)[span_ids]


def grass_update_energy(
    gram: torch.Tensor,
    span_ids: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor | None = None,
) -> torch.Tensor:
    """``left^T (H restricted to same-span pairs) right`` in the head geometry.

    Restricting to same-span pairs is what makes this comparable across selectors:
    the banded Gram also holds pairs the selector deliberately left in different
    spans, and counting those would attribute to every method an update change it
    did not make. ``right`` defaults to ``left``.
    """
    masked = gram.to(torch.float64) * (span_ids.unsqueeze(-1) == span_ids.unsqueeze(0))
    left = left.to(torch.float64)
    target = left if right is None else right.to(torch.float64)
    return left @ masked @ target


def grass_cosine(numerator: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Cosine of two non-negative head-update energies given their cross term."""
    denominator = (left.clamp_min(0.0) * right.clamp_min(0.0)).clamp_min(1e-300).sqrt()
    return numerator / denominator


# ---------------------------------------------------------------------------
# Section 17: Tier-A / Tier-B telemetry
# ---------------------------------------------------------------------------


def _band_cosines(gram: torch.Tensor, max_span_length: int) -> torch.Tensor:
    """Gradient cosines ``H_ij / sqrt(H_ii H_jj)`` for every in-band pair.

    Section 17.5 asks for ``d = 1 .. L_max - 1``: the widest pair the dynamic
    program can place inside one span is ``d = L_max - 1``, and it is precisely
    that pair whose pooling decides ``alpha``. Stopping at ``d = 2`` would omit
    the bandwidth the method is actually about.
    """
    n = gram.shape[0]
    length = min(int(max_span_length), n)
    if length < 2:
        return gram.new_zeros(0)
    diagonal = torch.diagonal(gram).to(torch.float64)
    parts = []
    for offset in range(1, length):
        if n - offset <= 0:
            continue
        rows = torch.arange(n - offset, device=gram.device).unsqueeze(-1)
        columns = rows + offset
        denominator = (diagonal[rows] * diagonal[columns]).clamp_min(1e-300).sqrt()
        parts.append((gram[rows, columns].to(torch.float64) / denominator).reshape(-1))
    if not parts:
        return gram.new_zeros(0)
    cosine = parts[0] if len(parts) == 1 else torch.cat(parts)
    return cosine.reshape(-1)


def grass_partition_metrics(
    tables: GrassSpanCosts,
    partition: Sequence[tuple[int, int]],
    rate: torch.Tensor,
    weight: torch.Tensor,
    gram: torch.Tensor,
    shrunk: torch.Tensor,
    margins: torch.Tensor,
    max_span_length: int,
    *,
    conservation_error: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Detached, always-on scalars describing one selected partition.

    Everything here is a read-out; none of it feeds back into the shrinkage, so a
    telemetry bug cannot change the training update. Names follow the existing
    ``mp_opd_gbv_*`` convention so one dashboard can carry both selectors.
    """
    n = tables.costs.shape[0]
    r, w, _n, _length = _symmetric_parts(rate, weight, int(rate.numel()))
    if shrunk.shape != r.shape:
        raise ValueError("shrunk credits must align with the atomic credits")
    lengths_tensor, span_ids, starts = _partition_index(partition, n, r.device)
    span_count = int(lengths_tensor.numel())
    columns = lengths_tensor - 1
    # One batched gather for every per-span scalar; a Python loop over spans costs
    # one device synchronisation each and buys no numerical accuracy.
    selected_cost = tables.costs[starts, columns]
    selected_distortion = tables.distortion[starts, columns]
    selected_variance = tables.variance[starts, columns]
    strengths = tables.alpha[starts, columns]

    shrunk64 = shrunk.to(torch.float64)
    # Head-update energies, summed over the selected spans only. Masking the Gram
    # by "same selected span" drops pairs the banded Gram does contain but that
    # the partition did not choose; summing per-span quadratic forms instead would
    # cost one tiny kernel launch per span.
    same_span = span_ids.unsqueeze(-1) == span_ids.unsqueeze(0)
    masked = gram.to(torch.float64) * same_span
    atomic_energy = r @ masked @ r
    grass_energy = shrunk64 @ masked @ shrunk64
    cross_energy = r @ masked @ shrunk64
    total_cost = selected_cost.sum()

    delta = shrunk64 - r
    pre_variation = float(((r[1:] - r[:-1]) ** 2).sum()) if n > 1 else 0.0
    post_variation = float(((shrunk64[1:] - shrunk64[:-1]) ** 2).sum()) if n > 1 else 0.0
    cosine = _band_cosines(gram.to(torch.float64), max_span_length)
    margin_body = margins[1:] if margins.numel() > 1 else margins
    # Positions with a single admissible predecessor carry no margin; counting
    # them as a near-tie would make the fraction a constant of the topology.
    decidable = margin_body[torch.isfinite(margin_body)]
    pre_variance = float(r.var(unbiased=False)) if n > 1 else 0.0
    metrics = {
        "mp_opd_grass_total_cost": total_cost,
        "mp_opd_grass_predicted_risk_reduction": -total_cost,
        "mp_opd_grass_predicted_risk_reduction_per_atom": -total_cost / n,
        "mp_opd_grass_distortion_term": selected_distortion.sum(),
        "mp_opd_grass_variance_term": selected_variance.sum(),
        "mp_opd_grass_sigma2": tables.costs.new_tensor(tables.sigma2),
        "mp_opd_grass_degenerate": tables.costs.new_tensor(float(tables.degenerate)),
        "mp_opd_grass_selected_span_count": total_cost.new_tensor(float(span_count)),
        "mp_opd_grass_selected_span_length_mean": lengths_tensor.to(torch.float64).mean(),
        "mp_opd_grass_span_length_max": lengths_tensor.max().to(torch.float64),
        "mp_opd_grass_alpha_mean": strengths.mean(),
        "mp_opd_grass_alpha_atomic_fraction": (strengths < 0.1).to(torch.float64).mean(),
        "mp_opd_grass_alpha_partial_fraction": (
            (strengths >= 0.1) & (strengths <= 0.9)
        ).to(torch.float64).mean(),
        "mp_opd_grass_alpha_hard_fraction": (strengths > 0.9).to(torch.float64).mean(),
        "mp_opd_grass_credit_l2_change": delta.norm(),
        "mp_opd_grass_credit_relative_l2_change": delta.norm() / r.norm().clamp_min(1e-300),
        "mp_opd_grass_credit_sign_change_fraction": (r * shrunk64 < 0).to(torch.float64).mean(),
        "mp_opd_grass_credit_adjacent_variation_ratio": total_cost.new_tensor(
            post_variation / pre_variation if pre_variation > 0 else 1.0
        ),
        "mp_opd_grass_credit_variance_ratio": total_cost.new_tensor(
            float(shrunk64.var(unbiased=False) / pre_variance) if pre_variance > 0 else 1.0
        ),
        "mp_opd_grass_credit_conservation_error": conservation_error.detach(),
        "mp_opd_grass_head_update_energy_atomic": atomic_energy,
        "mp_opd_grass_head_update_energy_shrunk": grass_energy,
        "mp_opd_grass_head_update_cosine": cross_energy
        / (atomic_energy.clamp_min(1e-300) * grass_energy.clamp_min(1e-300)).sqrt(),
        "mp_opd_grass_neighbour_gradient_cosine_mean": cosine.mean()
        if cosine.numel()
        else selected_cost.new_zeros(()),
        "mp_opd_grass_neighbour_gradient_cosine_negative_fraction": (cosine < 0)
        .to(torch.float64)
        .mean()
        if cosine.numel()
        else selected_cost.new_zeros(()),
        "mp_opd_grass_dp_margin_median": decidable.median()
        if decidable.numel()
        else selected_cost.new_zeros(()),
        "mp_opd_grass_dp_near_tie_fraction": (decidable.abs() < 1e-12)
        .to(torch.float64)
        .mean()
        if decidable.numel()
        else selected_cost.new_zeros(()),
    }
    for reported in REPORTED_SPAN_LENGTHS:
        fraction = float((lengths_tensor == reported).to(torch.float64).mean())
        metrics[f"mp_opd_grass_span_{reported}_fraction"] = total_cost.new_tensor(fraction)
    return {key: value.detach() for key, value in metrics.items()}


def grass_candidate_metrics(
    tables: GrassSpanCosts,
    selected: Sequence[tuple[int, int]] | None = None,
) -> dict[str, torch.Tensor]:
    """Sections 17.6 / 17.7 aggregate read-out over all candidates, and the selected ones.

    Both populations are reported because they localise different faults: sane
    candidates with pathological selected spans point at the DP competition,
    pathological candidates point at the local SURE statistics or the Gram.
    """
    valid = tables.valid
    if not bool(valid.any()):
        return {}
    # The reporting buckets are indexed against the table's own width, not against a
    # separately recomputed `length`: the two can disagree whenever a caller supplies
    # a table built with a different max_span_length, and indexing past the end is a
    # crash in a telemetry function.
    length = tables.costs.shape[1]
    distortion = tables.distortion[valid]
    variance = tables.variance[valid]
    alpha = tables.alpha[valid]
    cost = tables.costs[valid]
    trace = tables.trace[valid]
    tolerance = tables.costs.new_tensor(tables.negative_tol)
    normalized = -cost / trace.clamp_min(1e-300)
    # gamma = V/D is only a ratio where D is a real number. With the 1e-300 floor
    # the mean would be pinned at ~0 for every D_c == 0 span, hiding exactly the
    # "no observed heterogeneity but real predicted variance" population that
    # drives the degenerate alpha = 1 branch, so those are counted instead.
    positive_distortion = distortion > 0
    gamma = torch.where(
        positive_distortion,
        variance / distortion.clamp_min(1e-300),
        torch.zeros_like(variance),
    )
    zero_with_variance = (~positive_distortion) & (variance > 0)
    metrics = {
        "mp_opd_grass_candidate_count": tables.costs.new_tensor(float(int(valid.sum()))),
        "mp_opd_grass_candidate_distortion_mean": distortion.mean(),
        "mp_opd_grass_candidate_variance_mean": variance.mean(),
        "mp_opd_grass_candidate_alpha_mean": alpha.mean(),
        "mp_opd_grass_candidate_cost_mean": cost.mean(),
        "mp_opd_grass_candidate_negative_cost_fraction": (cost < 0).to(torch.float64).mean(),
        "mp_opd_grass_candidate_gamma_mean": gamma.mean(),
        "mp_opd_grass_candidate_zero_distortion_positive_variance_fraction": zero_with_variance
        .to(torch.float64)
        .mean(),
        "mp_opd_grass_candidate_trace_mean": trace.mean(),
        "mp_opd_grass_candidate_normalized_gain_mean": normalized.mean(),
        "mp_opd_grass_candidate_normalized_gain_max": normalized.max(),
        "mp_opd_grass_pathology_negative_d_fraction": (distortion < -tolerance)
        .to(torch.float64)
        .mean(),
        "mp_opd_grass_pathology_negative_v_fraction": (variance < -tolerance)
        .to(torch.float64)
        .mean(),
        "mp_opd_grass_pathology_alpha_low_fraction": (alpha <= 1e-12).to(torch.float64).mean(),
        "mp_opd_grass_pathology_alpha_high_fraction": (alpha >= 1.0 - 1e-12)
        .to(torch.float64)
        .mean(),
    }
    for reported in REPORTED_SPAN_LENGTHS:
        # The table is only `length` wide, and length = min(max_span_length, n).
        # A harness that defaults max_span_length to 2, or a 3-atom response, must
        # report fewer buckets - never index past the end of the table.
        if reported > length:
            continue
        column = tables.valid[:, reported - 1]
        if not bool(column.any()):
            continue
        metrics[f"mp_opd_grass_candidate_span_{reported}_count"] = tables.costs.new_tensor(
            float(int(column.sum()))
        )
        metrics[f"mp_opd_grass_candidate_span_{reported}_alpha_mean"] = tables.alpha[
            column, reported - 1
        ].mean()
    if selected:
        index = torch.tensor(
            [(start, end - start - 1) for start, end in selected],
            dtype=torch.int64,
            device=tables.costs.device,
        )
        rows, columns = index[:, 0], index[:, 1]
        metrics["mp_opd_grass_selected_distortion_mean"] = tables.distortion[rows, columns].mean()
        metrics["mp_opd_grass_selected_variance_mean"] = tables.variance[rows, columns].mean()
        metrics["mp_opd_grass_selected_cost_mean"] = tables.costs[rows, columns].mean()
    return {key: value.detach() for key, value in metrics.items()}


def grass_gram_diagnostics(gram: torch.Tensor, max_span_length: int) -> dict[str, torch.Tensor]:
    """Section 17.5 numerical quality of the head Gram itself.

    The exact head Gram is PSD up to round-off, so a material negative eigenvalue
    is an implementation bug until proven otherwise - which is why the
    minimum eigenvalue is reported rather than assumed.
    """
    n = gram.shape[0]
    length = min(int(max_span_length), n)
    diagonal = torch.diagonal(gram).to(torch.float64)
    metrics = {
        "mp_opd_grass_gram_diagonal_mean": diagonal.mean(),
        "mp_opd_grass_gram_diagonal_max": diagonal.max(),
    }
    cosine = _band_cosines(gram, max_span_length)
    metrics["mp_opd_grass_gram_offdiagonal_mean"] = cosine.mean() if cosine.numel() else diagonal.new_zeros(())
    metrics["mp_opd_grass_gram_offdiagonal_negative_fraction"] = (
        (cosine < 0).to(torch.float64).mean() if cosine.numel() else diagonal.new_zeros(())
    )
    if length >= 2:
        # Every contiguous window of the widest admissible width. A wider block is
        # the harder PSD case, so the worst case is what is sampled rather than an
        # arbitrary width; narrower spans are a sub-block of one of these by
        # principal submatrix ordering.
        rows = torch.arange(n - length + 1, device=gram.device)
        offsets = torch.arange(length, device=gram.device)
        # One [starts, length] grid of atom ids, used as both axes of the gather, so
        # entry (c, i, j) is H[start_c + i, start_c + j]: a stack of contiguous
        # blocks. Two different grids broadcast to incompatible shapes, and a single
        # grid on both axes builds a non-square matrix that eigvalsh rejects.
        grid = rows.view(-1, 1) + offsets.view(1, -1)
        block = gram[grid.unsqueeze(-1), grid.unsqueeze(-2)].to(torch.float64)
        smallest = torch.linalg.eigvalsh(block).min(dim=1).values
        metrics["mp_opd_grass_gram_max_span_min_eigenvalue"] = smallest.min()
        metrics["mp_opd_grass_gram_max_span_negative_fraction"] = (
            (smallest < -1e-8 * smallest.abs().clamp_min(1e-300).max())
            .to(torch.float64)
            .mean()
        )
    return {key: value.detach() for key, value in metrics.items()}


def gram_local_blocks(
    gram: torch.Tensor, partition: Sequence[tuple[int, int]]
) -> list[torch.Tensor]:
    """``H_c`` for each selected span - the Tier-C raw-matrix export of section 17.16.

    Kept as a plain function so the Tier-C writer can save exactly the objects the
    algorithm used, without the algorithm having to know about file layout.
    """
    return [gram[start:end, start:end].detach().clone() for start, end in partition]