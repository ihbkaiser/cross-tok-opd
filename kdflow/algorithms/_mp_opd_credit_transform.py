"""Cross-atom credit operators for MP-OPD: ``A = K r`` over atomic rate credits.

Atomic MP-OPD attaches the realized-path rate credit ``r_i = b_i / w_i`` to atom
``i`` and trains with ``sum_i stopgrad(r_i) NLL_i``. In operator language that is
the identity, ``K = I``. GBV-Span instead pooled credits inside a selected span
(``K = P``), which is what this module deliberately does **not** reimplement.

The operators here test a different hypothesis: the credit assignment of Atomic
may be too *local*, because the decision at atom ``i`` influences the quality of
later continuation atoms. Every operator in this module therefore pulls credit
from neighbours of ``i`` and is written so that the experiments of
``cross_atom_credit_experiment_spec.md`` can run through one interface:

* ``identity``      -- Atomic itself, ``A_i = r_i``;
* ``forward``       -- ``A_i = (1 - lam) r_i + lam r_{i+1}`` (Experiment C);
* ``backward``      -- matched anti-causal control;
* ``shuffle``       -- matched non-directional control, marginal preserving;
* ``causal_kernel`` -- static forward/backward/symmetric kernels (Experiment D);
* ``external``      -- offline oracle-ish ``r_i + alpha * A_hat_future`` (Experiment B).

Contract kept intact by design: ``b_i``, ``w_i`` and ``r_i`` are never redefined
here, and no credit may cross a boundary Atomic already treats as invalid. A
masked atom is neither a source nor a destination: it keeps its own atomic credit
and its neighbours fall back to atomic for that direction. The same rule applies
at sequence boundaries, so a packed batch never leaks credit across sequences.

These operators are pure functions of detached credits. They select no spans, run
no dynamic program and read no gradients; they only re-weight the same atoms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch

IDENTITY_CODE = 0
FORWARD_CODE = 1
BACKWARD_CODE = 2
SHUFFLE_CODE = 3
CAUSAL_KERNEL_CODE = 4
EXTERNAL_CODE = 5

CREDIT_TRANSFORM_CHOICES = (
    "identity",
    "forward",
    "backward",
    "shuffle",
    "causal_kernel",
    "external",
)
CREDIT_KERNEL_CHOICES = ("uniform", "exponential")
CREDIT_DIRECTION_CHOICES = ("forward", "backward", "symmetric")
CREDIT_SCALE_MATCH_CHOICES = ("raw", "rms")

DIRECTION_CODES = {"forward": 0.0, "backward": 1.0, "symmetric": 2.0}
KERNEL_CODES = {"uniform": 0.0, "exponential": 1.0}
SCALE_MATCH_CODES = {"raw": 0.0, "rms": 1.0}

METRIC_PREFIX = "mp_opd_credit_"
_EPS = 1e-12


@dataclass(frozen=True)
class AtomCreditBatch:
    """Flattened atom credits for one packed batch, in sequence order.

    Invariants (validated, cheaply): every tensor is one-dimensional with the same
    length; ``valid_mask`` is boolean; ``seq_ids`` is non-decreasing so each
    sequence occupies one contiguous block; ``atom_positions[i]`` is the rank of
    atom ``i`` inside its own sequence.
    """

    rate_credit: torch.Tensor  # r_i
    base_credit: torch.Tensor  # b_i
    token_count: torch.Tensor  # w_i
    valid_mask: torch.Tensor
    seq_ids: torch.Tensor
    atom_positions: torch.Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rate = self.rate_credit
        if rate.ndim != 1:
            raise ValueError("rate_credit must be a flat [atoms] tensor")
        for name in ("base_credit", "token_count", "valid_mask", "seq_ids", "atom_positions"):
            value = getattr(self, name)
            if value.shape != rate.shape:
                raise ValueError(f"{name} must share the rate_credit shape")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean")
        if self.seq_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("seq_ids must be an integer tensor")
        if rate.numel() == 0:
            return
        if bool((self.seq_ids[1:] < self.seq_ids[:-1]).any().item()):
            raise ValueError("seq_ids must be non-decreasing: pack sequences contiguously")
        index = torch.arange(rate.numel(), device=rate.device)
        is_start = torch.ones_like(self.seq_ids, dtype=torch.bool)
        is_start[1:] = self.seq_ids[1:] != self.seq_ids[:-1]
        group_start = torch.cummax(torch.where(is_start, index, torch.zeros_like(index)), 0).values
        if not bool(torch.equal(index - group_start, self.atom_positions)):
            raise ValueError("atom_positions must be the within-sequence rank of each atom")


@dataclass(frozen=True)
class CreditTransformOutput:
    effective_credit: torch.Tensor  # A_i
    diagnostics: dict[str, torch.Tensor]


@runtime_checkable
class CreditTransform(Protocol):
    def __call__(
        self,
        batch: AtomCreditBatch,
        *,
        training: bool,
    ) -> CreditTransformOutput:  # pragma: no cover - interface only
        ...


def _index(batch: AtomCreditBatch) -> torch.Tensor:
    return torch.arange(batch.rate_credit.numel(), device=batch.rate_credit.device)


def _shifted_index(batch: AtomCreditBatch, offset: int) -> torch.Tensor:
    n = batch.rate_credit.numel()
    return (_index(batch) + int(offset)).clamp(min=0, max=max(n - 1, 0))


def neighbor_reach(batch: AtomCreditBatch, offset: int) -> torch.Tensor:
    """``True`` where ``r[i + offset]`` is reachable from atom ``i``.

    Reachability requires the whole chain ``i -> i + offset`` to stay inside the
    tensor, inside one sequence, and entirely on valid atoms. ``offset == 0`` is
    reachable exactly on valid atoms.
    """
    n = batch.rate_credit.numel()
    if n == 0:
        return batch.valid_mask.clone()
    index = _index(batch)
    sequence = batch.seq_ids
    valid = batch.valid_mask
    reach = valid.clone()
    step = 0 if offset == 0 else (1 if offset > 0 else -1)
    for distance in range(1, abs(int(offset)) + 1):
        target = index + step * distance
        inside = (target >= 0) & (target < n)
        safe = target.clamp(min=0, max=n - 1)
        reach = reach & inside & valid[safe] & (sequence[safe] == sequence)
    return reach


def accumulate_dtype(reference: torch.Tensor) -> torch.dtype:
    """Widest of float32/float64 for a reduction over credits.

    bf16/bf16-like credits reduce in float32 (training credits are float32 already);
    float64 stays float64 so tests can pin exact arithmetic.
    """
    return torch.float64 if reference.dtype == torch.float64 else torch.float32


def _effective_stats(
    batch: AtomCreditBatch, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    selected = values.detach().float()[batch.valid_mask]
    count = int(selected.numel())
    if count == 0:
        zero = batch.rate_credit.new_zeros(())
        return zero, zero, zero, 0
    return selected.mean(), selected.std(unbiased=False), selected.pow(2).mean().sqrt(), count


def _safe_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Correlation, reported as 0 rather than NaN when either side is constant."""
    if left.numel() == 0:
        return left.new_zeros(())
    a = left.float() - left.float().mean()
    b = right.float() - right.float().mean()
    energy = a.norm() * b.norm()
    return torch.where(energy > _EPS, (a * b).sum() / energy.clamp_min(_EPS), a.new_zeros(())).detach()


def _neighbor_agreement(batch: AtomCreditBatch) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Mean ``r_i r_{i+1}`` and mean sign agreement over adjacent valid pairs."""
    n = batch.rate_credit.numel()
    if n < 2:
        zero = batch.rate_credit.new_zeros(())
        return zero, zero, 0
    reach = neighbor_reach(batch, 1)
    if not bool(reach.any().item()):
        zero = batch.rate_credit.new_zeros(())
        return zero, zero, 0
    rate = batch.rate_credit.detach().float()
    left = rate[reach]
    right = rate[_shifted_index(batch, 1)][reach]
    product = (left * right).mean()
    agreement = ((left > 0) == (right > 0)).float().mean()
    return product.detach(), agreement.detach(), int(left.numel())


def credit_diagnostics(
    batch: AtomCreditBatch,
    effective: torch.Tensor,
    *,
    name: str,
    code: int,
    parameters: Mapping[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Logging fields of the credit operator (``cross_atom_credit`` spec section 11).

    Every value is a finite scalar tensor: the trainer fails closed on a
    non-finite metric, so empty inputs and zero-variance credits must report a
    concrete zero rather than a NaN.
    """
    device = batch.rate_credit.device
    dtype = torch.float32
    rate = batch.rate_credit.detach()
    valid = batch.valid_mask
    atomic = rate[valid].float()
    applied = effective.detach()[valid].float()
    mean_atomic, std_atomic, rms_atomic, count = _effective_stats(batch, rate)
    mean_eff, std_eff, rms_eff, _ = _effective_stats(batch, effective)
    if count:
        delta = (applied - atomic).abs()
        mean_abs_delta = delta.mean()
        transfer_fraction = (delta > 0).float().mean()
        sign_flip = ((applied > 0) != (atomic > 0)).float().mean()
        correlation = _safe_correlation(applied, atomic)
    else:
        zero = torch.zeros((), device=device, dtype=dtype)
        mean_abs_delta = transfer_fraction = sign_flip = correlation = zero
    product, agreement, pairs = _neighbor_agreement(batch)
    metrics: dict[str, torch.Tensor] = {
        f"{METRIC_PREFIX}transform_code": torch.tensor(float(code), device=device, dtype=dtype),
        f"{METRIC_PREFIX}valid_atom_count": torch.tensor(float(count), device=device, dtype=dtype),
        f"{METRIC_PREFIX}mean_atomic": mean_atomic.to(device=device, dtype=dtype),
        f"{METRIC_PREFIX}std_atomic": std_atomic.to(device=device, dtype=dtype),
        f"{METRIC_PREFIX}rms_atomic": rms_atomic.to(device=device, dtype=dtype),
        f"{METRIC_PREFIX}mean_effective": mean_eff.to(device=device, dtype=dtype),
        f"{METRIC_PREFIX}std_effective": std_eff.to(device=device, dtype=dtype),
        f"{METRIC_PREFIX}rms_effective": rms_eff.to(device=device, dtype=dtype),
        f"{METRIC_PREFIX}corr_effective_atomic": correlation.to(device=device, dtype=dtype),
        f"{METRIC_PREFIX}mean_abs_delta": mean_abs_delta.to(device=device, dtype=dtype),
        f"{METRIC_PREFIX}transfer_fraction": transfer_fraction.to(device=device, dtype=dtype),
        f"{METRIC_PREFIX}sign_flip_fraction": sign_flip.to(device=device, dtype=dtype),
        f"{METRIC_PREFIX}neighbor_product_mean": product.to(device=device, dtype=dtype),
        f"{METRIC_PREFIX}neighbor_sign_agreement": agreement.to(device=device, dtype=dtype),
        f"{METRIC_PREFIX}neighbor_pair_count": torch.tensor(
            float(pairs), device=device, dtype=dtype
        ),
    }
    for key, value in (parameters or {}).items():
        metrics[f"{METRIC_PREFIX}{key}"] = torch.tensor(
            float(value), device=device, dtype=dtype
        )
    return metrics


class CreditTransformBase:
    """Shared plumbing: no-grad evaluation, shape check, uniform telemetry."""

    name = "identity"
    code = IDENTITY_CODE

    def parameter_metrics(self) -> dict[str, float]:
        return {}

    def _effective_credit(self, batch: AtomCreditBatch) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, batch: AtomCreditBatch, *, training: bool = False) -> CreditTransformOutput:
        with torch.no_grad():
            effective = self._effective_credit(batch)
        if effective.shape != batch.rate_credit.shape:
            raise ValueError(
                f"{type(self).__name__} returned {tuple(effective.shape)} for "
                f"rate_credit {tuple(batch.rate_credit.shape)}"
            )
        return CreditTransformOutput(
            effective_credit=effective,
            diagnostics=credit_diagnostics(
                batch,
                effective,
                name=self.name,
                code=self.code,
                parameters=self.parameter_metrics(),
            ),
        )


class IdentityCreditTransform(CreditTransformBase):
    """Atomic MP-OPD: the credit operator is the identity, ``A_i = r_i``."""

    name = "identity"
    code = IDENTITY_CODE

    def _effective_credit(self, batch: AtomCreditBatch) -> torch.Tensor:
        return batch.rate_credit.clone()


def _neighbor_effective(
    batch: AtomCreditBatch,
    offset: int,
    lam: float,
    *,
    convex: bool,
    source: torch.Tensor | None = None,
) -> torch.Tensor:
    rate = batch.rate_credit
    if rate.numel() == 0:
        return rate.clone()
    reach = neighbor_reach(batch, offset)
    values = rate if source is None else source
    origin = values[_shifted_index(batch, offset)]
    if convex:
        transferred = (1.0 - lam) * rate + lam * origin
    else:
        transferred = rate + lam * origin
    return torch.where(reach, transferred, rate)


class ForwardNeighborCreditTransform(CreditTransformBase):
    """``A_i = (1 - lam) r_i + lam r_{i+1}`` with atomic fallback at boundaries."""

    name = "forward"
    code = FORWARD_CODE

    def __init__(self, lam: float, convex: bool = True):
        if not 0.0 <= float(lam) <= 1.0:
            raise ValueError("forward credit lambda must be in [0, 1]")
        self.lam = float(lam)
        self.convex = bool(convex)

    def parameter_metrics(self) -> dict[str, float]:
        return {
            "lambda": self.lam,
            "convex": float(self.convex),
            "horizon": 1.0,
            "direction_code": DIRECTION_CODES["forward"],
        }

    def _effective_credit(self, batch: AtomCreditBatch) -> torch.Tensor:
        return _neighbor_effective(batch, 1, self.lam, convex=self.convex)


class BackwardNeighborCreditTransform(CreditTransformBase):
    """Matched anti-causal control: ``A_i = (1 - lam) r_i + lam r_{i-1}``."""

    name = "backward"
    code = BACKWARD_CODE

    def __init__(self, lam: float, convex: bool = True):
        if not 0.0 <= float(lam) <= 1.0:
            raise ValueError("backward credit lambda must be in [0, 1]")
        self.lam = float(lam)
        self.convex = bool(convex)

    def parameter_metrics(self) -> dict[str, float]:
        return {
            "lambda": self.lam,
            "convex": float(self.convex),
            "horizon": 1.0,
            "direction_code": DIRECTION_CODES["backward"],
        }

    def _effective_credit(self, batch: AtomCreditBatch) -> torch.Tensor:
        return _neighbor_effective(batch, -1, self.lam, convex=self.convex)


def _shuffle_partner(batch: AtomCreditBatch, seed: int) -> torch.Tensor:
    """Random partner inside each sequence, as a bijection over valid atoms.

    Valid atoms of one sequence are ordered by a seeded random key and displaced by
    half the group size. The induced map is a permutation of the sequence's valid
    atoms, so ``r[partner]`` has exactly the marginal distribution of ``r``: the
    control changes the *assignment* of credit, never its multiset or its scale.
    Masked atoms are never touched and never act as a source, and no partner is
    taken from another sequence. (A group of size 1 maps to itself.)
    """
    index = _index(batch)
    valid_positions = index[batch.valid_mask]
    count = int(valid_positions.numel())
    partner = index.clone()
    if count == 0:
        return partner
    keys = torch.rand(
        count,
        generator=torch.Generator(device="cpu").manual_seed(int(seed)),
        dtype=torch.float64,
    ).to(batch.rate_credit.device)
    # Sequence-major, then ascending random key: each sequence's valid atoms become
    # one contiguous group.
    by_key = torch.argsort(keys, stable=True)
    by_sequence = torch.argsort(batch.seq_ids[valid_positions[by_key]], stable=True)
    ordered = valid_positions[by_key[by_sequence]]
    position = torch.arange(count, device=index.device)
    ordered_sequence = batch.seq_ids[ordered]
    starts = torch.ones_like(ordered_sequence, dtype=torch.bool)
    if count > 1:
        starts[1:] = ordered_sequence[1:] != ordered_sequence[:-1]
    group = torch.cumsum(starts.to(torch.long), 0) - 1
    # `group_start[j]` is the first ordered position of j's own group.
    group_start = torch.cummax(
        torch.where(starts, position, torch.zeros_like(position)), 0
    ).values
    group_size = torch.zeros(int(group[-1]) + 1, device=index.device, dtype=position.dtype)
    group_size.scatter_add_(0, group, torch.ones_like(position))
    within = position - group_start
    size = group_size[group]
    # `offset` is 0 for a singleton group (identity) and floor(size/2) otherwise.
    offset = torch.where(size > 1, size // 2, torch.zeros_like(size))
    target = group_start + (within + offset) % size.clamp_min(1)
    partner[ordered] = ordered[target]
    return partner


class ShuffledNeighborCreditTransform(CreditTransformBase):
    """Non-directional control: ``A_i = (1 - lam) r_i + lam r_{pi(i)}``.

    ``pi`` is drawn inside the same sequence only, so no credit crosses a sequence
    boundary, and it is a permutation of the valid atoms, so the marginal credit
    distribution is preserved exactly (section 4.4 of the spec).
    """

    name = "shuffle"
    code = SHUFFLE_CODE

    def __init__(self, lam: float, seed: int = 43, convex: bool = True):
        if not 0.0 <= float(lam) <= 1.0:
            raise ValueError("shuffle credit lambda must be in [0, 1]")
        self.lam = float(lam)
        self.seed = int(seed)
        self.convex = bool(convex)

    def parameter_metrics(self) -> dict[str, float]:
        return {
            "lambda": self.lam,
            "convex": float(self.convex),
            "horizon": 1.0,
            "shuffle_seed": float(self.seed),
            "direction_code": DIRECTION_CODES["forward"],
        }

    def _effective_credit(self, batch: AtomCreditBatch) -> torch.Tensor:
        rate = batch.rate_credit
        if rate.numel() == 0:
            return rate.clone()
        partner = _shuffle_partner(batch, self.seed)
        source = rate[partner]
        if self.convex:
            shuffled = (1.0 - self.lam) * rate + self.lam * source
        else:
            shuffled = rate + self.lam * source
        # A masked atom is never a destination; it keeps its own atomic credit.
        return torch.where(batch.valid_mask, shuffled, rate)


def kernel_weights(kind: str, horizon: int, decay: float) -> torch.Tensor:
    """Static causal kernel weights ``lambda_k``, ``k = 0 .. horizon - 1``."""
    if kind not in CREDIT_KERNEL_CHOICES:
        raise ValueError(f"unsupported credit kernel: {kind}")
    if int(horizon) < 1:
        raise ValueError("credit kernel horizon must be at least 1")
    if not 0.0 < float(decay) <= 1.0:
        raise ValueError("credit kernel decay must be in (0, 1]")
    distance = torch.arange(int(horizon), dtype=torch.float64)
    if kind == "uniform":
        return torch.full((int(horizon),), 1.0 / float(horizon), dtype=torch.float64)
    weights = float(decay) ** distance
    return weights / weights.sum()


class CausalKernelCreditTransform(CreditTransformBase):
    """Static kernel ``A_i = sum_k lambda_k r_{i + s k}`` with matched controls.

    ``direction`` selects the offset family: ``forward`` uses ``k = 0..H-1``,
    ``backward`` mirrors it, and ``symmetric`` averages the two (same weights, so
    the comparison isolates temporal direction rather than kernel mass). The
    denominator is the sum of the weights actually reachable, which keeps the
    credit scale controlled and makes boundary atoms fall back to atomic.
    """

    name = "causal_kernel"
    code = CAUSAL_KERNEL_CODE

    def __init__(
        self,
        horizon: int = 2,
        kind: str = "uniform",
        decay: float = 0.5,
        direction: str = "forward",
    ):
        if direction not in CREDIT_DIRECTION_CHOICES:
            raise ValueError(f"unsupported credit kernel direction: {direction}")
        self.horizon = int(horizon)
        self.kind = str(kind)
        self.decay = float(decay)
        self.direction = str(direction)
        self.weights = kernel_weights(self.kind, self.horizon, self.decay)

    def parameter_metrics(self) -> dict[str, float]:
        return {
            "lambda": float(self.weights[1]) if self.horizon > 1 else 0.0,
            "decay": self.decay,
            "horizon": float(self.horizon),
            "kernel_code": KERNEL_CODES[self.kind],
            "direction_code": DIRECTION_CODES[self.direction],
            "weight_mass": float(self.weights.sum()),
        }

    def _directional(self, batch: AtomCreditBatch, sign: int) -> torch.Tensor:
        rate = batch.rate_credit
        n = rate.numel()
        if n == 0:
            return rate.clone()
        accumulator = accumulate_dtype(rate)
        values = rate.to(accumulator)
        numerator = torch.zeros_like(values)
        denominator = torch.zeros_like(values)
        for step in range(self.horizon):
            offset = sign * step
            weight = self.weights[step].to(accumulator)
            reach = neighbor_reach(batch, offset)
            source = values[_shifted_index(batch, offset)]
            numerator = numerator + weight * torch.where(reach, source, torch.zeros_like(values))
            denominator = denominator + weight * reach.to(accumulator)
        combined = torch.where(
            denominator > 0, numerator / denominator.clamp_min(_EPS), torch.zeros_like(values)
        )
        return combined.to(rate.dtype)

    def _effective_credit(self, batch: AtomCreditBatch) -> torch.Tensor:
        rate = batch.rate_credit
        if self.direction == "forward":
            combined = self._directional(batch, 1)
        elif self.direction == "backward":
            combined = self._directional(batch, -1)
        else:
            combined = 0.5 * (self._directional(batch, 1) + self._directional(batch, -1))
        # A masked atom keeps its atomic credit and never acts as a source.
        return torch.where(batch.valid_mask, combined, rate)


class ExternalCreditAugmentationTransform(CreditTransformBase):
    """Offline augmentation ``A_i = r_i + alpha * A_hat_future_i`` (Experiment B).

    ``A_hat_future`` must be supplied by the caller as ``metadata['future_advantage']``
    aligned with the flattened atoms. ``scale_match='rms'`` rescales the augmented
    credit to the atomic RMS so an apparent effect cannot come from a larger credit
    or gradient magnitude (spec section 3.4).
    """

    name = "external"
    code = EXTERNAL_CODE

    def __init__(self, alpha: float, scale_match: str = "raw"):
        if not float(alpha) >= 0.0 or not torch.isfinite(torch.tensor(float(alpha))):
            raise ValueError("external credit alpha must be finite and non-negative")
        if scale_match not in CREDIT_SCALE_MATCH_CHOICES:
            raise ValueError(f"unsupported credit scale_match: {scale_match}")
        self.alpha = float(alpha)
        self.scale_match = str(scale_match)

    def parameter_metrics(self) -> dict[str, float]:
        return {
            "lambda": self.alpha,
            "scale_match_code": SCALE_MATCH_CODES[self.scale_match],
            "horizon": 0.0,
        }

    def _effective_credit(self, batch: AtomCreditBatch) -> torch.Tensor:
        advantage = batch.metadata.get("future_advantage")
        if advantage is None:
            raise RuntimeError(
                "external credit transform requires metadata['future_advantage']"
            )
        advantage = torch.as_tensor(advantage, device=batch.rate_credit.device)
        if advantage.shape != batch.rate_credit.shape:
            raise ValueError("future_advantage must align with the flattened atoms")
        advantage = torch.where(
            batch.valid_mask, advantage.detach().to(batch.rate_credit.dtype),
            torch.zeros_like(batch.rate_credit),
        )
        augmented = batch.rate_credit + self.alpha * advantage
        if self.scale_match == "rms":
            selected = batch.valid_mask
            if bool(selected.any().item()):
                atomic_rms = batch.rate_credit[selected].float().pow(2).mean().sqrt()
                augmented_rms = augmented[selected].float().pow(2).mean().sqrt()
                if float(augmented_rms) > _EPS:
                    augmented = augmented * (atomic_rms / augmented_rms).to(augmented.dtype)
        return augmented


def production_credit_batch(
    rate_credit: torch.Tensor,
    base_credit: torch.Tensor,
    token_count: torch.Tensor,
    metadata: Mapping[str, Any] | None = None,
) -> AtomCreditBatch:
    """One sample's atoms as the trainer presents them: one sequence, all valid.

    Production calls the operator once per sample, so the packed-batch axes are
    degenerate. Atomizer-rejected samples never reach the credit path, and masked
    EOS tokens live outside every atom, exactly as in the Atomic credit path. The
    trainer and the offline probes share this constructor so a probe cannot pass on
    a batch geometry the trainer never builds.
    """
    n = rate_credit.numel()
    return AtomCreditBatch(
        rate_credit=rate_credit,
        base_credit=base_credit,
        token_count=token_count,
        valid_mask=torch.ones_like(rate_credit, dtype=torch.bool),
        seq_ids=torch.zeros_like(rate_credit, dtype=torch.long),
        atom_positions=torch.arange(n, device=rate_credit.device),
        metadata=dict(metadata or {}),
    )


@dataclass(frozen=True)
class CreditTransformSpec:
    """Resolved, loggable description of one credit operator."""

    name: str
    code: int
    parameters: Mapping[str, float]
    transform: CreditTransform

    def invocation_record(self) -> dict[str, Any]:
        return {
            "credit_transform": self.name,
            "credit_transform_code": self.code,
            "credit_transform_parameters": dict(self.parameters),
        }


def build_credit_transform(
    name: str,
    *,
    lam: float = 0.25,
    convex: bool = True,
    horizon: int = 2,
    kernel: str = "uniform",
    decay: float = 0.5,
    direction: str = "forward",
    shuffle_seed: int = 43,
    alpha: float = 0.25,
    scale_match: str = "raw",
) -> CreditTransformSpec:
    if name not in CREDIT_TRANSFORM_CHOICES:
        raise ValueError(f"unsupported mp_opd_credit_transform: {name}")
    if name == "identity":
        transform: CreditTransform = IdentityCreditTransform()
    elif name == "forward":
        transform = ForwardNeighborCreditTransform(lam, convex=convex)
    elif name == "backward":
        transform = BackwardNeighborCreditTransform(lam, convex=convex)
    elif name == "shuffle":
        transform = ShuffledNeighborCreditTransform(lam, seed=shuffle_seed, convex=convex)
    elif name == "causal_kernel":
        transform = CausalKernelCreditTransform(
            horizon=horizon, kind=kernel, decay=decay, direction=direction
        )
    else:
        transform = ExternalCreditAugmentationTransform(alpha, scale_match=scale_match)
    parameters = transform.parameter_metrics()
    return CreditTransformSpec(
        name=name,
        code=int(getattr(transform, "code", IDENTITY_CODE)),
        parameters=parameters,
        transform=transform,
    )


def credit_transform_from_args(kd: Any) -> CreditTransformSpec:
    """Build the configured operator from the MP-OPD argument namespace."""
    return build_credit_transform(
        str(getattr(kd, "mp_opd_credit_transform", "identity")),
        lam=float(getattr(kd, "mp_opd_credit_lambda", 0.25)),
        convex=bool(getattr(kd, "mp_opd_credit_convex", True)),
        horizon=int(getattr(kd, "mp_opd_credit_horizon", 2)),
        kernel=str(getattr(kd, "mp_opd_credit_kernel", "uniform")),
        decay=float(getattr(kd, "mp_opd_credit_decay", 0.5)),
        direction=str(getattr(kd, "mp_opd_credit_direction", "forward")),
        shuffle_seed=int(getattr(kd, "mp_opd_credit_shuffle_seed", 43)),
        alpha=float(getattr(kd, "mp_opd_credit_alpha", 0.25)),
        scale_match=str(getattr(kd, "mp_opd_credit_scale_match", "raw")),
    )
