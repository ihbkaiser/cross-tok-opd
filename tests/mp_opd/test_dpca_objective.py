"""DPCA objective: parity with the upstream formula, and the two properties that
matter scientifically.

The reference implementation here is a naive per-token transcription of
``verl.trainer.ppo.core_algos.compute_policy_loss_opd`` (ivanniu/On-Policy-Distill
@ 927a8264). Vectorising that function is only legitimate if the result matches the
transcription, so the vectorised path is pinned against it rather than against a
hand-written expectation.

Two claims get their own tests because they are the ones most likely to be reported
as working when they are not:

* the clip only bites when the behaviour and current policies actually differ, so a
  run with one optimizer update per rollout iteration reports ``clipfrac == 0``;
* chunk boundaries must not leak, or the semantic prior silently reads across atoms.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kdflow.algorithms._mp_opd_dpca import (
    DPCAConfig,
    dpca_atom_advantages,
    dpca_atom_coverage,
    dpca_metrics_to_tensors,
    dpca_policy_loss,
    rollout_temperature_log_probs,
)

CONFIG = DPCAConfig()


def _reference_policy_loss(prior, current, advantages, config=CONFIG):
    """Transcription of compute_policy_loss_opd, one token at a time."""
    losses = []
    clipfrac = 0.0
    clipfrac_lower = 0.0
    for old, new, adv in zip(prior.tolist(), current.tolist(), advantages.tolist()):
        ratio = min(20.0, max(-20.0, new - old))
        ratio = torch.tensor(ratio).exp().item()
        pg1 = -adv * ratio
        lo = 1.0 - config.clip_ratio_low
        hi = 1.0 + config.clip_ratio_high
        pg2 = -adv * min(max(ratio, lo), hi)
        clip1 = max(pg1, pg2)
        if pg2 > pg1:
            clipfrac += 1.0
        pg3 = -adv * config.clip_ratio_c
        clip2 = min(pg3, clip1)
        if clip1 > pg3 and adv < 0:
            clipfrac_lower += 1.0
        losses.append(clip2 if adv < 0 else clip1)
    n = float(len(losses))
    return sum(losses) / n, clipfrac / n, clipfrac_lower / n


def test_matches_upstream_transcription():
    torch.manual_seed(0)
    prior = torch.randn(37) * 2.0
    current = prior + torch.randn(37) * 0.6
    advantages = torch.randn(37) * 3.0

    loss, metrics = dpca_policy_loss(prior, current, advantages, torch.ones(37), CONFIG)
    ref, ref_clip, ref_lower = _reference_policy_loss(prior, current, advantages)

    assert loss.item() == pytest.approx(ref, rel=1e-6)
    assert metrics["mp_opd_dpca_clipfrac"] == pytest.approx(ref_clip, rel=1e-6)
    assert metrics["mp_opd_dpca_clipfrac_lower"] == pytest.approx(ref_lower, rel=1e-6)


def test_clip_is_inert_when_the_policy_has_not_moved():
    """One update per rollout iteration evaluates a single, un-updated policy.

    This is the R1 canary observation generalised: ``ppo_kl`` and ``clip_fraction``
    were both exactly zero for ten consecutive steps. A test that pinned clipping as
    active would be pinning a mechanism this trainer cannot exercise.
    """
    torch.manual_seed(1)
    prior = torch.randn(64) * 2.0
    advantages = torch.randn(64) * 3.0

    loss, metrics = dpca_policy_loss(prior, prior.clone(), advantages, torch.ones(64), CONFIG)

    assert metrics["mp_opd_dpca_ppo_kl"] == 0.0
    assert metrics["mp_opd_dpca_clipfrac"] == 0.0
    assert metrics["mp_opd_dpca_clipfrac_lower"] == 0.0
    assert metrics["mp_opd_dpca_ratio_mean"] == pytest.approx(1.0)
    assert loss.item() == pytest.approx((-advantages).mean().item(), rel=1e-6)


def test_clip_engages_once_the_policy_moves():
    prior = torch.zeros(16)
    current = torch.full((16,), 5.0)
    advantages = torch.ones(16)

    _, metrics = dpca_policy_loss(prior, current, advantages, torch.ones(16), CONFIG)

    # ratio = e^5 is far above 1 + clip_ratio_high, so every token is clipped.
    assert metrics["mp_opd_dpca_clipfrac"] == 1.0


def test_semantic_prior_matches_the_paper_formula():
    """A_i = (L_T / L_S - 1) * log p_i, with L_T and L_S summed inside a chunk."""
    prior = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    counts = torch.tensor([2, 2])
    # L_S is summed from prior: atom 0 -> -3, atom 1 -> -7
    teacher = torch.tensor([-6.0, -10.5])   # ratios 2.0 and 1.5

    advantages = dpca_atom_advantages(teacher, counts, prior, adv_clamp=None)

    expected = torch.cat(
        [
            (teacher[0] / -3.0 - 1.0) * prior[0:2],
            (teacher[1] / -7.0 - 1.0) * prior[2:4],
        ]
    )
    assert torch.allclose(advantages, expected, atol=1e-5)
    assert torch.allclose(advantages, torch.tensor([-1.0, -2.0, -1.5, -2.0]), atol=1e-5)


def test_chunks_do_not_leak_across_atoms():
    """Two atoms with different ratios must not share one denominator.

    Dividing a whole chunk's teacher mass by a single token's log-probability
    (``L_T / log p_i``) instead of the chunk's own sum (``L_T / L_S``) is the exact
    failure this pins: it still produces finite, plausible-looking advantages.
    """
    prior = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    counts = torch.tensor([2, 2])
    teacher = torch.tensor([-1.0, -6.0])   # L_S = -2 both; ratios 0.5 and 3.0

    advantages = dpca_atom_advantages(teacher, counts, prior, adv_clamp=None)

    assert torch.allclose(advantages, torch.tensor([0.5, 0.5, -2.0, -2.0]), atol=1e-6)
    assert not torch.allclose(advantages[0:2], advantages[2:4])


def test_zero_prior_chunk_spreads_the_teacher_budget_uniformly():
    """L_S == 0 is the degenerate chunk upstream handles explicitly."""
    prior = torch.zeros(3)
    counts = torch.tensor([3])
    teacher = torch.tensor([-3.0])

    advantages = dpca_atom_advantages(teacher, counts, prior, adv_clamp=None)

    # target = L_T / n_chunk = -1 per token, so A = -1 - 0.
    assert torch.allclose(advantages, torch.full((3,), -1.0), atol=1e-6)


def test_clamp_saturates_and_is_measurable():
    prior = torch.tensor([-1.0, -1.0])
    counts = torch.tensor([1, 1])
    teacher = torch.tensor([-100.0, 0.0])      # L_S = -1 both; ratios 100 and 0

    raw = dpca_atom_advantages(teacher, counts, prior, adv_clamp=None)
    clamped = dpca_atom_advantages(teacher, counts, prior, adv_clamp=10.0)

    # token 0: teacher far more negative than the student -> advantage -99
    assert raw[0].item() == pytest.approx(-99.0)
    # token 1: teacher exactly matches -> ratio 0 -> advantage +1
    assert raw[1].item() == pytest.approx(1.0)
    assert raw.abs().max().item() > 10.0
    assert clamped.abs().max().item() == 10.0
    assert clamped[0].item() == -10.0
    assert clamped[1].item() == 1.0


def test_counts_that_do_not_cover_the_prior_fail_closed():
    with pytest.raises(ValueError):
        dpca_atom_advantages(
            torch.tensor([-6.0]), torch.tensor([2]), torch.tensor([-1.0]), None
        )


def test_mismatched_shapes_fail_closed():
    with pytest.raises(ValueError):
        dpca_atom_advantages(
            torch.zeros(5), torch.tensor([1, 1, 1, 1]), torch.zeros(4), None
        )


def test_atom_coverage_is_full_mask_width_and_narrows_on_use():
    """Coverage is built at the unfiltered mask width, then used to narrow.

    The mask carries a synthetic EOS the atomizer excluded, so coverage is
    strictly narrower than the tensors it selects from. If the width were taken
    from an already-narrowed tensor, the two would disagree by exactly that gap.
    """
    full_width = 5
    covered = dpca_atom_coverage([(0, 1), (1, 2), (2, 3)], full_width, torch.device("cpu"))

    assert covered.numel() == full_width
    assert covered.sum().item() == 3

    full = torch.arange(full_width, dtype=torch.float32)
    assert torch.equal(full[covered], torch.tensor([0.0, 1.0, 2.0]))


def test_atom_coverage_marks_every_token_of_a_multi_token_atom():
    spans = [(0, 1), (1, 4), (6, 7)]
    covered = dpca_atom_coverage(spans, 8, torch.device("cpu"))
    assert covered.tolist() == [True, True, True, True, False, False, True, False]


def test_atom_coverage_fails_closed_when_nothing_is_covered():
    with pytest.raises(RuntimeError):
        dpca_atom_coverage([], 4, torch.device("cpu"))


def test_chunk_total_comes_from_the_prior_not_a_separate_array():
    """L_S must be summed from the same array that supplies log p_i.

    Upstream sums ``prior_log_probs`` for the chunk denominator and reuses that
    array for the per-token term. Passing an independent per-atom total is the
    failure mode where both halves are individually plausible and the ratio is
    quietly formed from two different conventions.
    """
    prior = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    counts = torch.tensor([2, 2])
    teacher = torch.tensor([-6.0, -7.0])   # ratios 2.0 and 1.0

    advantages = dpca_atom_advantages(teacher, counts, prior, adv_clamp=None)

    # L_S = [-3, -7]; ratios [2.0, 1.0]; A = (ratio - 1) * log p_i.
    # A positive L_T/L_S raises the chunk's target above log p_i, and since
    # log p_i is negative that makes the advantage more negative. The same sign
    # shows up in test_semantic_prior_matches_the_paper_formula.
    expected = torch.tensor([-1.0, -2.0, 0.0, 0.0])
    assert torch.allclose(advantages, expected, atol=1e-5)


def test_tempered_log_probs_match_a_reference_softmax():
    """The current policy must be rebuilt on pi_T, like verl's actor does.

    Dividing the logits by the rollout temperature before the log-softmax is what
    puts the ratio's two sides on the same distribution. Comparing against an
    explicit float64 softmax pins both the scaling and the gather.
    """
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 0.5, 0.5]])
    labels = torch.tensor([0, 2])

    got = rollout_temperature_log_probs(logits, labels, 0.6)

    reference = torch.log_softmax(logits.double() / 0.6, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    assert torch.allclose(got.double(), reference, atol=1e-5)


def test_tempered_log_probs_differ_from_the_raw_ones():
    """Guards the fix itself: at T != 1 the two distributions must not coincide."""
    logits = torch.tensor([[2.0, 0.0, -1.0]])
    labels = torch.tensor([0])

    tempered = rollout_temperature_log_probs(logits, labels, 0.6)
    raw = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    assert not torch.allclose(tempered, raw, atol=1e-3)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_temperature_fails_closed(temperature):
    with pytest.raises(ValueError):
        rollout_temperature_log_probs(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long), temperature)


def test_ppo_kl_keeps_the_upstream_sign():
    """Upstream reports masked_mean(-negative_approx_kl).

    Dropping the negation makes the reported number the exact opposite of the
    reference runs, which is easy to misread as "the policy moved the wrong way".
    """
    prior = torch.tensor([-1.0, -1.0])
    log_probs = torch.tensor([-2.0, -2.0])   # negative_approx_kl = -1 per token
    advantages = torch.ones(2)

    _, metrics = dpca_policy_loss(prior, log_probs, advantages, torch.ones(2), CONFIG)

    assert metrics["mp_opd_dpca_ppo_kl"] == pytest.approx(1.0)


def test_policy_metrics_are_floats_not_tensors():
    """dpca_policy_loss reports Python floats on purpose.

    They are what get aggregated per sample, so the tensor conversion has to be a
    separate, explicit step. If this ever returns tensors, extra_sums' ``value.
    new_zeros(())`` accumulator changes shape silently.
    """
    prior = torch.tensor([-1.0, -2.0])
    log_probs = torch.tensor([-1.2, -2.4])
    advantages = torch.tensor([0.5, -0.5])
    _, metrics = dpca_policy_loss(
        prior, log_probs, advantages, torch.ones(2), DPCAConfig()
    )
    assert metrics, "policy loss reported no metrics"
    assert all(isinstance(v, float) for v in metrics.values()), metrics


def test_scalar_metrics_are_materialized_on_the_requested_device():
    """The conversion must honour the device it is handed.

    A bare ``torch.as_tensor(float)`` lands on CPU no matter what device is
    requested, which is exactly the bug that made training_step's finite check
    fail on torch.stack with the loss-derived metrics still on CUDA.
    """
    metrics = {"a": 1.0, "b": -0.5}
    tensors = dpca_metrics_to_tensors(metrics, torch.device("cpu"))
    assert set(tensors) == set(metrics)
    assert all(isinstance(v, torch.Tensor) for v in tensors.values())
    assert all(v.device.type == "cpu" for v in tensors.values())
    assert tensors["b"].item() == pytest.approx(-0.5)