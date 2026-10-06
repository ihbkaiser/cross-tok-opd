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
    dpca_policy_loss,
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
    # atom 0: L_S = -3, L_T = -6 -> ratio 2; atom 1: L_S = -7, L_T = -10.5 -> ratio 1.5
    student_old = torch.tensor([-3.0, -7.0])
    teacher = torch.tensor([-6.0, -10.5])

    advantages = dpca_atom_advantages(student_old, teacher, counts, prior, adv_clamp=None)

    expected = torch.cat(
        [(teacher[0] / student_old[0] - 1.0) * prior[0:2], (teacher[1] / student_old[1] - 1.0) * prior[2:4]]
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
    student_old = torch.tensor([-2.0, -2.0])   # L_S equal, L_T differ
    teacher = torch.tensor([-1.0, -6.0])       # ratios 0.5 and 3.0

    advantages = dpca_atom_advantages(student_old, teacher, counts, prior, adv_clamp=None)

    assert torch.allclose(advantages, torch.tensor([0.5, 0.5, -2.0, -2.0]), atol=1e-6)
    assert not torch.allclose(advantages[0:2], advantages[2:4])


def test_zero_prior_chunk_spreads_the_teacher_budget_uniformly():
    """L_S == 0 is the degenerate chunk upstream handles explicitly."""
    prior = torch.zeros(3)
    counts = torch.tensor([3])
    student_old = torch.tensor([0.0])
    teacher = torch.tensor([-3.0])

    advantages = dpca_atom_advantages(student_old, teacher, counts, prior, adv_clamp=None)

    # target = L_T / n_chunk = -1 per token, so A = -1 - 0.
    assert torch.allclose(advantages, torch.full((3,), -1.0), atol=1e-6)


def test_clamp_saturates_and_is_measurable():
    prior = torch.tensor([-1.0, -1.0])
    counts = torch.tensor([1, 1])
    student_old = torch.tensor([-1.0, -1.0])
    teacher = torch.tensor([-100.0, 0.0])      # ratios 100 and 0

    raw = dpca_atom_advantages(student_old, teacher, counts, prior, adv_clamp=None)
    clamped = dpca_atom_advantages(student_old, teacher, counts, prior, adv_clamp=10.0)

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
            torch.tensor([-3.0]), torch.tensor([-6.0]), torch.tensor([2]), torch.tensor([-1.0]), None
        )


def test_mismatched_shapes_fail_closed():
    with pytest.raises(ValueError):
        dpca_atom_advantages(
            torch.zeros(4), torch.zeros(5), torch.tensor([1, 1, 1, 1]), torch.zeros(4), None
        )