"""Independent counterexamples from the October algorithm audit."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kdflow.algorithms._mp_opd_grass_span import (
    GrassNoiseEstimator, atom_head_gram, grass_span_costs, grass_shrink,
)
from kdflow.algorithms._mp_opd_grass_chunk import (
    atom_chunk_ids_from_tokens, chunk_partition, chunk_boundary_diagnostics,
    grass_chunk_tables, grass_chunk_metrics, chunk_shadow_metrics, run_chunk_assignment,
)
from kdflow.algorithms._mp_opd_align import align_chunk, _row_relative_change
from kdflow.algorithms._mp_opd_trust_batch import TrustHeadAccumulator
from kdflow.algorithms._mp_opd_trust import trust_calibrate, trust_calibrate_scalars


def test_trust_rejects_invalid_geometry_instead_of_clamping_it_to_a_lambda():
    one = torch.ones(1)
    with pytest.raises(ValueError, match="negative squared norm"):
        trust_calibrate(one, one, -torch.eye(1), torch.eye(1), torch.zeros(1, 1))
    with pytest.raises(ValueError, match="Cauchy-Schwarz"):
        trust_calibrate(one, one, torch.eye(1), torch.eye(1), torch.full((1, 1), 2.))
    with pytest.raises(ValueError, match="nonnegative"):
        trust_calibrate_scalars(-1., 1., 0., 1, 1)


def test_grass_two_atom_counterexample_changes_the_actual_loss_gradient():
    r = torch.tensor([0., 10.], dtype=torch.float64)
    w = torch.ones(2, dtype=torch.float64)
    tables = grass_span_costs(r, w, torch.eye(2), 1., 2)
    assert float(tables.distortion[0, 1]) == pytest.approx(50.)
    assert float(tables.variance[0, 1]) == pytest.approx(1.)
    assert float(tables.alpha[0, 1]) == pytest.approx(.02)
    e, _ = grass_shrink(r, w, [(0, 2)], tables.alpha)
    nll = torch.ones(2, dtype=torch.float64, requires_grad=True)
    (e.detach()*nll).sum().backward()
    assert torch.allclose(nll.grad, torch.tensor([.1, 9.9], dtype=torch.float64))


def test_zero_mad_is_an_accepted_observation_not_a_missing_observation():
    estimator = GrassNoiseEstimator(rho=.99, min_adjacent_pairs=8)
    r = torch.tensor([0., 1., 0., 1., 0., 2., 0., 2., 0.])
    first = estimator.update(r, torch.ones(9))["sigma_batch_variance"]
    for _ in range(3):
        estimator.update(torch.zeros(9), torch.ones(9))
    assert estimator.state_dict()["updates"] == 4
    assert estimator.sigma2 == pytest.approx(first*.01*.99**3/(1-.99**4))


def test_confident_token_gram_and_bias_match_explicit_gradients():
    p = torch.tensor([[.9999, .0001], [.9998, .0002]], dtype=torch.float32)
    h = torch.tensor([[1., 2.], [3., -1.]])
    labels = torch.zeros(2, dtype=torch.long)
    for bias in (False, True):
        result = atom_head_gram(p.log(), h, labels, [(0, 2)], 1, head_bias=bias)
        delta = p.double().clone(); delta[:, 0] -= 1
        g = delta.T @ h.double()
        expected = g.square().sum()
        if bias:
            expected += delta.sum(0).square().sum()
        assert float(result.gram[0, 0]) > 0
        assert float(result.gram[0, 0]) == pytest.approx(float(expected), rel=.002)
        assert math.isfinite(result.symmetry_error)


def test_chunk_identity_cosines_use_one_common_geometry():
    r = torch.ones(2)
    h = torch.tensor([[1., .5], [.5, 1.]])
    w = torch.ones(2)
    tables = grass_chunk_tables(r, w, h, [(0, 2)], 0.)
    metrics = grass_chunk_metrics(tables, r, w, h, r, torch.tensor(0.),
                                  assignment=run_chunk_assignment(2, 2),
                                  noncontiguous_splits=0, source="run")
    assert float(metrics["mp_opd_grass_chunk_head_update_cosine"]) == pytest.approx(1.)
    shadow = chunk_shadow_metrics(h, r, w, [(0, 2)], r)
    assert float(shadow["mp_opd_grass_chunk_shadow_atomic_head_cosine_to_grass"]) == pytest.approx(1.)


def test_singleton_boundary_and_missing_cross_gram_are_not_measured_zeros():
    r = torch.tensor([0., 10., 11.])
    metrics = chunk_boundary_diagnostics(r, torch.ones(3), torch.eye(3),
                                        [(0, 1), (1, 3)], gram_is_block_diagonal=True)
    assert metrics["adjacent_pairs_within"] == 1
    assert metrics["adjacent_pairs_cross"] == 1
    assert metrics["within_abs_diff_mean"] == 1
    assert metrics["cross_abs_diff_mean"] == 10
    assert metrics["cross_gradient_cosine_pairs_undefined"] == 1


def test_partially_unaligned_atom_remains_a_singleton():
    assignment = atom_chunk_ids_from_tokens([-1, 7, 7], [(0, 2), (2, 3)])
    partition, _ = chunk_partition(assignment)
    assert partition == ((0, 1), (1, 2))
    assert assignment.unaligned == 1


def test_align_rounding_of_a_true_psd_gram_never_produces_a_complex_metric():
    g = torch.tensor([[-.0001824775814887203, -1.], [.8298368729994624, 0.],
                      [-.5480125239723292, .0001]], dtype=torch.float64)
    result = align_chunk(torch.ones(3), (g@g.T).float())
    assert isinstance(result.row_relative_change_mean, float)
    assert math.isfinite(result.row_relative_change_mean)
    bad = torch.tensor([[1., -1., -1.], [-1., 1., -2.], [-1., -2., 1.]])
    with pytest.raises(ValueError, match="negative change norm"):
        _row_relative_change(torch.ones(3), bad, torch.tensor([[1., 1., 3.], [0., 1., 0.], [0., 0., 1.]]))


@pytest.mark.parametrize("softcap,bias", [(None, False), (30., False), (30., True)])
def test_streamed_full_batch_head_updates_match_autograd(softcap, bias):
    torch.manual_seed(72)
    weight = torch.randn(7, 3, requires_grad=True)
    b = torch.randn(7, requires_grad=True) if bias else None
    collector = TrustHeadAccumulator(softcap=softcap, head_bias=bias, vocab_chunk=3)
    strict_loss = mismatch_loss = 0
    for _ in range(3):
        h = torch.randn(5, 3)
        logits = h @ weight.T + (b if b is not None else 0)
        if softcap:
            logits = softcap*torch.tanh(logits/softcap)
        labels = torch.tensor([0, 1, 6, 1, 3])
        lp = logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)
        rates = torch.tensor([.3, -.2, .8])
        is_strict = torch.tensor([True, False, True])
        ranges = [(0, 1), (1, 3), (3, 5)]
        collector.add_response(logits, h, labels, ranges, rates, is_strict, lp)
        strict_loss += -.3*lp[0] - .8*lp[3:].sum()
        mismatch_loss += .2*lp[1:3].sum()
    params = (weight, b) if bias else (weight,)
    gs = torch.autograd.grad(strict_loss, params, retain_graph=True)
    gm = torch.autograd.grad(mismatch_loss, params)
    assert torch.allclose(collector.gs, gs[0], atol=2e-6, rtol=2e-5)
    assert torch.allclose(collector.gm, gm[0], atol=2e-6, rtol=2e-5)
    if bias:
        assert torch.allclose(collector.bs, gs[1], atol=2e-6)
        assert torch.allclose(collector.bm, gm[1], atol=2e-6)
    expected = max(0., sum(float((s*m).sum()) for s, m in zip(gs, gm)) /
                   sum(float(m.square().sum()) for m in gm))
    result = collector.finalize()
    assert result.lam == pytest.approx(expected, rel=2e-5, abs=2e-6)
    assert result.n_strict == 6 and result.n_mismatch == 3


def test_full_batch_calibration_is_not_microbatch_calibration():
    assert trust_calibrate_scalars(4., .25, 1., 2, 2).lam == 4
    one = trust_calibrate_scalars(1., 1., 1., 1, 1).lam
    two = trust_calibrate_scalars(1., .25, -.5, 1, 1).lam
    assert 2 + one*1 + two*(-.5) == 3


def test_pair_affordability_guard_runs_before_concatenation(monkeypatch):
    from test_numeric_fallback import _make_algo, _atom
    import kdflow.algorithms.mp_opd as module
    algo = _make_algo("trust_b")
    monkeypatch.setattr(module, "gram_affordable", lambda *args: False)
    monkeypatch.setattr(torch, "cat", lambda *args, **kwargs: pytest.fail("allocated before guard"))
    atoms = [[_atom("one_to_one", 0, 2)]]*2
    with pytest.raises(ValueError, match="before concatenation"):
        algo._trust_pair_gram(0, 1, atoms, [torch.ones(2, 3)]*2,
                             [torch.ones(2, 2)]*2, [torch.zeros(2, dtype=torch.long)]*2,
                             [torch.ones(2)]*2)


def test_native_chunker_does_not_require_an_unused_vocab_projection():
    from test_numeric_fallback import _make_algo
    algo = _make_algo("grass_chunk")
    algo.student_tokenizer.convert_ids_to_tokens = lambda ids: [str(i) for i in ids]
    algo.teacher_tokenizer.convert_ids_to_tokens = algo.student_tokenizer.convert_ids_to_tokens
    aligner = algo._build_grass_aligner()
    output = aligner.align(torch.tensor([[0, 1, 2]]), torch.tensor([[0, 1, 2]]))
    assert (output.student_chunk_id >= 0).all()
