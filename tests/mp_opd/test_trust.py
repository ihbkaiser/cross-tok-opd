"""TRUST v1.0 - trusted reference update scaling.

The tests here are written against the method note rather than against the
implementation, so an implementation that happens to be self-consistent still
fails. Where a property is a direct consequence of the calibration geometry -
kappa never exceeding one, an opposing mismatch calibrating to zero, an empty
side leaving the strict loss whole - it is asserted directly instead of through
a stored expectation.
"""

from __future__ import annotations

import pytest
import torch

from kdflow.algorithms._mp_opd_trust import (
    TRUST_EPS_G,
    strict_mask,
    trust_batch_metrics,
    trust_calibrate,
    trust_response_loss,
    trust_response_metrics,
)


def _psd_gram(m: int, seed: int) -> torch.Tensor:
    # Guaranteed PSD by construction (A A^T), like the numpy audit: a merely
    # symmetric matrix need not satisfy Cauchy-Schwarz, and the kappa bound
    # below is a statement about true Grams, which always are PSD.
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(m, m, generator=generator, dtype=torch.float32)
    return raw @ raw.T + 0.1 * torch.eye(m, dtype=torch.float32)


# ---------------------------------------------------------------------------
# The strict/mismatch partition
# ---------------------------------------------------------------------------


def test_strict_mask_files_boundary_types_and_fails_closed():
    mask = strict_mask(["one_to_one", "multi_token", "one_to_one"])
    assert mask.tolist() == [True, False, True]
    with pytest.raises(ValueError):
        strict_mask(["one_to_one", "aligned_span"])


# ---------------------------------------------------------------------------
# The calibration itself
# ---------------------------------------------------------------------------


def test_opposing_mismatch_calibrates_to_zero():
    # Negative Gram off-diagonal with positive credits: the mismatch update
    # fights the strict one, so no amount of it is trusted.
    gram = torch.tensor([[1.0, -0.95], [-0.95, 1.0]], dtype=torch.float32)
    result = trust_calibrate(
        torch.tensor([1.0]), torch.tensor([1.0]),
        gram[:1, :1], gram[1:, 1:], gram[:1, 1:],
    )
    assert result.lam == 0.0
    assert result.calibrated is True
    assert result.kappa == 0.0


def test_aligned_pair_gives_the_hand_computed_lambda():
    # dot = 1*0.5*0.9 = 0.45, gm2 = 0.25*1 = 0.25, so lam = 1.8;
    # kappa = 1.8*0.5/1.0 = 0.9.
    gram = torch.tensor([[1.0, 0.9], [0.9, 1.0]], dtype=torch.float32)
    result = trust_calibrate(
        torch.tensor([1.0]), torch.tensor([0.5]),
        gram[:1, :1], gram[1:, 1:], gram[:1, 1:],
    )
    assert result.lam == pytest.approx(1.8)
    assert result.kappa == pytest.approx(0.9)
    assert result.cosine == pytest.approx(0.9)
    assert result.calibrated is True


def test_empty_sides_leave_lambda_at_zero():
    empty = torch.zeros(0, dtype=torch.float32)
    one = torch.tensor([1.0])
    eye1 = torch.eye(1, dtype=torch.float32)
    no_cross = torch.zeros(0, 1, dtype=torch.float32)
    no_strict = trust_calibrate(empty, one, torch.zeros(0, 0), eye1, no_cross)
    assert no_strict.lam == 0.0
    assert no_strict.calibrated is False
    assert no_strict.has_strict is False
    assert no_strict.has_mismatch is True
    no_mismatch = trust_calibrate(one, empty, eye1, torch.zeros(0, 0), torch.zeros(1, 0))
    assert no_mismatch.lam == 0.0
    assert no_mismatch.calibrated is False


def test_degenerate_mismatch_norm_disables_calibration():
    gram = torch.eye(2, dtype=torch.float32)
    result = trust_calibrate(
        torch.tensor([1.0]), torch.tensor([0.0]),
        gram[:1, :1], gram[1:, 1:], gram[:1, 1:],
    )
    assert result.gm_norm2 == 0.0
    assert result.lam == 0.0
    assert result.calibrated is False


def test_kappa_never_exceeds_one_on_random_problems():
    generator = torch.Generator().manual_seed(7)
    for trial in range(200):
        n = int(torch.randint(1, 6, (), generator=generator))
        m = int(torch.randint(1, 6, (), generator=generator))
        gram = _psd_gram(n + m, 1000 + trial)
        qs = torch.randn(n, generator=generator, dtype=torch.float32)
        qm = torch.randn(m, generator=generator, dtype=torch.float32)
        result = trust_calibrate(qs, qm, gram[:n, :n], gram[n:, n:], gram[:n, n:])
        # Tolerance is fp32 roundoff headroom, not doubt about the bound: a
        # genuine violation from a non-PSD input overshoots by order one.
        assert result.kappa <= 1.0 + 1e-4
        assert result.lam >= 0.0


def test_both_vanishing_updates_compare_equal():
    result = trust_calibrate(
        torch.tensor([0.0]), torch.tensor([0.0]),
        torch.eye(1, dtype=torch.float32),
        torch.eye(1, dtype=torch.float32),
        torch.zeros(1, 1, dtype=torch.float32),
    )
    assert result.update_cosine == 1.0
    assert result.update_norm_ratio == 0.0


def test_non_finite_inputs_fail_fast():
    gram = torch.eye(2, dtype=torch.float32)
    bad = torch.tensor([float("inf")])
    with pytest.raises(ValueError):
        trust_calibrate(bad, torch.tensor([1.0]), gram[:1, :1], gram[1:, 1:], gram[:1, 1:])
    with pytest.raises(ValueError):
        trust_calibrate(
            torch.tensor([1.0]), torch.tensor([1.0]),
            torch.full((1, 1), float("nan")), gram[1:, 1:], gram[:1, 1:],
        )


def test_shape_mismatches_fail_fast():
    gram = torch.eye(4, dtype=torch.float32)
    # Cross block (2, 2) where (2, 1) belongs.
    with pytest.raises(ValueError):
        trust_calibrate(torch.ones(2), torch.ones(1), gram[:2, :2], gram[2:3, 2:3], gram[:2, :2])
    # Strict block (1, 1) where (2, 2) belongs.
    with pytest.raises(ValueError):
        trust_calibrate(torch.ones(2), torch.ones(1), gram[:1, :1], gram[2:3, 2:3], gram[:2, 2:3])


def test_negative_eps_g_is_rejected():
    gram = torch.eye(2, dtype=torch.float32)
    with pytest.raises(ValueError):
        trust_calibrate(
            torch.tensor([1.0]), torch.tensor([1.0]),
            gram[:1, :1], gram[1:, 1:], gram[:1, 1:], eps_g=-1.0,
        )


# ---------------------------------------------------------------------------
# The loss: strict kept whole, mismatch scaled by the detached lambda
# ---------------------------------------------------------------------------


def _hand_loss(nll, rate, is_strict, lam):
    strict = sum(float(r) * float(v) for r, v, s in zip(rate, nll, is_strict) if s)
    mismatch = sum(float(r) * float(v) for r, v, s in zip(rate, nll, is_strict) if not s)
    return strict + lam * mismatch


def test_response_loss_matches_the_hand_combination():
    nll = torch.tensor([0.5, 1.5, 2.5, 0.25], dtype=torch.float32)
    rate = torch.tensor([1.0, -0.5, 2.0, 1.5], dtype=torch.float32)
    is_strict = torch.tensor([True, False, True, False])
    gram = _psd_gram(4, 11)
    result = trust_calibrate(
        rate[is_strict], rate[~is_strict],
        gram[is_strict][:, is_strict], gram[~is_strict][:, ~is_strict],
        gram[is_strict][:, ~is_strict],
    )
    loss = trust_response_loss(nll, rate, is_strict, ~is_strict, result)
    assert float(loss) == pytest.approx(
        _hand_loss(nll.tolist(), rate.tolist(), is_strict.tolist(), result.lam)
    )


def test_mismatch_free_response_keeps_the_strict_loss_whole():
    nll = torch.tensor([0.5, 1.5], dtype=torch.float32)
    rate = torch.tensor([1.0, 2.0], dtype=torch.float32)
    is_strict = torch.tensor([True, True])
    gram = torch.eye(2, dtype=torch.float32)
    result = trust_calibrate(rate, torch.zeros(0), gram, torch.zeros(0, 0), torch.zeros(2, 0))
    assert result.lam == 0.0
    loss = trust_response_loss(nll, rate, is_strict, ~is_strict, result)
    assert float(loss) == pytest.approx(0.5 * 1.0 + 1.5 * 2.0)


def test_overlapping_sides_are_rejected():
    nll = torch.ones(2, dtype=torch.float32)
    rate = torch.ones(2, dtype=torch.float32)
    both = torch.tensor([True, True])
    gram = torch.eye(2, dtype=torch.float32)
    result = trust_calibrate(rate, rate, gram, gram, gram)
    with pytest.raises(ValueError):
        trust_response_loss(nll, rate, both, both, result)


def test_gradients_flow_only_through_the_atom_nll():
    # Hand-built PSD Gram: dot = 0.5*2.0 + 0.5*(-1.0) = 0.5, gm2 = 3.0,
    # so lam = 1/6 exactly.
    gram = torch.tensor(
        [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]], dtype=torch.float32
    )
    nll = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32, requires_grad=True)
    rate = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32)
    is_strict = torch.tensor([True, False, False])
    result = trust_calibrate(
        rate[is_strict], rate[~is_strict],
        gram[is_strict][:, is_strict], gram[~is_strict][:, ~is_strict],
        gram[is_strict][:, ~is_strict],
    )
    assert result.lam == pytest.approx(1.0 / 6.0)
    loss = trust_response_loss(nll, rate, is_strict, ~is_strict, result)
    grad = torch.autograd.grad(loss, nll)[0]
    assert grad.tolist() == pytest.approx([1.0, 2.0 / 6.0, -1.0 / 6.0])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_response_metrics_carry_counts_and_scope():
    gram = torch.tensor([[1.0, 0.9], [0.9, 1.0]], dtype=torch.float32)
    result = trust_calibrate(
        torch.tensor([1.0]), torch.tensor([0.5]),
        gram[:1, :1], gram[1:, 1:], gram[:1, 1:],
    )
    raw = trust_response_metrics(
        result, torch.tensor([True, False]), torch.tensor([1.0, 3.0]), scope="response"
    )
    assert raw["strict_span_count"] == 1.0
    assert raw["mismatch_span_count"] == 1.0
    assert raw["strict_atom_count"] == 1.0
    assert raw["strict_token_count"] == 1.0
    assert raw["mismatch_token_count"] == 3.0
    assert raw["strict_span_fraction"] == 0.5
    assert raw["has_both"] == 1.0
    assert raw["zero_strict"] == 0.0
    assert raw["lambda"] == pytest.approx(1.8)
    assert raw["lambda_gt1"] == 1.0
    assert raw["lambda_zero"] == 0.0
    assert raw["calibrated"] == 1.0
    assert raw["cosine_negative"] == 0.0
    assert raw["scope"] == 0.0
    with pytest.raises(ValueError):
        trust_response_metrics(result, torch.tensor([True]), torch.tensor([1.0]), scope="chunk")


def test_batch_metrics_are_singletons_with_batch_scope():
    gram = torch.tensor([[1.0, 0.9], [0.9, 1.0]], dtype=torch.float32)
    result = trust_calibrate(
        torch.tensor([1.0]), torch.tensor([0.5]),
        gram[:1, :1], gram[1:, 1:], gram[:1, 1:],
    )
    counts = {
        "strict_span_count": 2.0, "mismatch_span_count": 3.0,
        "strict_atom_count": 2.0, "mismatch_atom_count": 3.0,
        "strict_token_count": 4.0, "mismatch_token_count": 9.0,
        "strict_span_fraction": 0.4, "mismatch_span_fraction": 0.6,
    }
    out = trust_batch_metrics(result, counts)
    assert out["scope"] == 1.0
    assert out["batch_strict_span_count"] == 2.0
    assert out["batch_mismatch_span_fraction"] == 0.6
    assert out["lambda"] == pytest.approx(1.8)
    assert out["batch_calibrated"] == 1.0
    assert out["cosine"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Gram correspondence: the block arithmetic must match materialized head gradients
# ---------------------------------------------------------------------------


def test_gram_blocks_match_materialized_head_gradients():
    # Section 10.7: dot_sm, gs_norm2 and gm_norm2 from Gram blocks must equal
    # the same quantities from explicit G_i = sum_t delta_t h_t^T.
    generator = torch.Generator().manual_seed(3)
    tokens, vocab, hidden = 6, 5, 4
    logits = torch.randn(tokens, vocab, generator=generator, dtype=torch.float32)
    state = torch.randn(tokens, hidden, generator=generator, dtype=torch.float32)
    labels = torch.randint(0, vocab, (tokens,), generator=generator)
    probs = torch.softmax(logits, dim=-1)
    delta = probs.clone()
    delta[torch.arange(tokens), labels] -= 1.0
    # Two atoms: strict tokens 0..1, mismatch tokens 2..5.
    grads = []
    for start, end in ((0, 2), (2, 6)):
        grads.append(delta[start:end].T @ state[start:end])
    vec = torch.stack([g.reshape(-1) for g in grads])
    gram = vec @ vec.T
    qs = torch.tensor([1.0])
    qm = torch.tensor([-0.5])
    result = trust_calibrate(qs, qm, gram[:1, :1], gram[1:, 1:], gram[:1, 1:])
    gs_direct = vec[:1] * 1.0
    gm_direct = vec[1:] * -0.5
    assert result.gs_norm2 == pytest.approx(float((gs_direct @ gs_direct.T).sum()))
    assert result.gm_norm2 == pytest.approx(float((gm_direct @ gm_direct.T).sum()))
    assert result.dot == pytest.approx(float((gs_direct @ gm_direct.T).sum()))
    assert result.lam == pytest.approx(max(0.0, float((gs_direct @ gm_direct.T).sum()) / float((gm_direct @ gm_direct.T).sum())))


def test_full_call_pattern_through_chunk_head_gram():
    # The exact call mp_opd makes: one chunk over all atoms, fp32 Gram out.
    from kdflow.algorithms._mp_opd_grass_chunk import chunk_head_gram

    generator = torch.Generator().manual_seed(9)
    tokens, vocab, hidden = 5, 7, 4
    logits = torch.randn(tokens, vocab, generator=generator, dtype=torch.float32)
    state = torch.randn(tokens, hidden, generator=generator, dtype=torch.float32)
    labels = torch.randint(0, vocab, (tokens,), generator=generator)
    token_nll = torch.rand(tokens, generator=generator, dtype=torch.float32)
    atom_ranges = ((0, 2), (2, 5))
    head = chunk_head_gram(
        logits, state, labels, atom_ranges, [(0, 2)],
        selected_log_prob=-token_nll,
        softcap=None, head_bias=False, diagonal_only=False,
    )
    assert tuple(head.gram.shape) == (2, 2)
    assert bool(torch.isfinite(head.gram).all())
    rate = torch.tensor([1.0, -0.5])
    is_strict = torch.tensor([True, False])
    result = trust_calibrate(
        rate[is_strict], rate[~is_strict],
        head.gram[is_strict][:, is_strict], head.gram[~is_strict][:, ~is_strict],
        head.gram[is_strict][:, ~is_strict],
    )
    assert result.has_strict and result.has_mismatch
    assert result.lam >= 0.0
