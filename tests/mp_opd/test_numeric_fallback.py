"""Numeric guardrail downgrades: warnings and atomic fallbacks, not dead runs.

The geometry modes (grass, grass_chunk, align, trust_r) route through one
guarded block in `_partition_loss`:

* non-finite mode inputs raise immediately with a diagnosis (no finite loss
  can come out of them, and the call would poison stateful estimators);
* a mode failure on finite inputs falls back to the atomic credits and counts
  itself loudly via `mp_opd_numeric_fallback_fraction`;
* CUDA out-of-memory is never caught: it is a resource failure, and a fallback
  loop around it would retry until the node burns.

These tests construct the algorithm with plain namespaces (no
DistillationArguments, which would pull transformers) so they run on the same
CPU image as the other suites.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from kdflow.algorithms.mp_opd import MetaPartitionedOPD


class FakeTokenizer:
    def __init__(self):
        self.eos_token_id = None

    def decode(self, ids, **_kwargs):
        return ""

    def get_added_vocab(self):
        return {}


def _make_algo(mode="grass"):
    kd = SimpleNamespace(
        kd_ratio=1.0,
        mp_opd_mode=mode,
        # Production default: fixed-run baseline, so __init__ never tries to
        # build the audited xtoken aligner in these tests.
        mp_opd_grass_chunk_source="run",
        mp_opd_max_span_length=2,
        mp_opd_fixed_span_length=2,
        mp_opd_min_span_length=1,
        mp_opd_partition_temperature=1.0,
        mp_opd_random_seed=43,
        xtoken_max_comb_len=4,
        xtoken_projection_path=None,
        xtoken_projection_sha256=None,
        mp_opd_energy_checkpoint=None,
        mp_opd_energy_hidden_dim=8,
        mp_opd_energy_layers=1,
        mp_opd_energy_lr=0.001,
    )
    args = SimpleNamespace(
        kd=kd,
        train=SimpleNamespace(max_norm=1.0),
        data=SimpleNamespace(max_len=512),
        rollout=SimpleNamespace(temperature=0.6),
    )
    strategy = SimpleNamespace(args=args, ring_attn_group=None)
    return MetaPartitionedOPD(
        strategy, object(), torch.nn.Linear(2, 3), FakeTokenizer(), FakeTokenizer()
    )


def _atom(boundary_type, start, end):
    return SimpleNamespace(
        boundary_type=boundary_type,
        student_start=start,
        student_end=end,
        student_token_count=end - start,
    )


def _credits(rate, n_tokens=3, vocab=5):
    nll = torch.tensor([0.5, 1.5], dtype=torch.float32, requires_grad=True)
    return (
        SimpleNamespace(
            rate=torch.tensor(rate, dtype=torch.float32),
            weight=torch.ones(2),
            base_credit=torch.ones(2),
            current_nll=nll,
            student_token_nll=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        ),
        nll,
    )


def _atoms():
    return [_atom("one_to_one", 0, 2), _atom("multi_token", 2, 3)]


def test_geometry_failure_falls_back_to_atomic_with_flag():
    algo = _make_algo("grass")
    credits, nll = _credits([1.0, -0.5])
    atoms = _atoms()
    # No logits: _grass_loss raises RuntimeError, the wrapper must catch it and
    # fall back instead of killing the run.
    loss, metrics = algo._partition_loss(
        credits,
        atoms,
        {},
        0,
        student_logits=None,
        student_labels=torch.tensor([1, 2, 3]),
        student_hidden=torch.randn(3, 4),
    )
    assert float(metrics["mp_opd_numeric_fallback_fraction"]) == 1.0
    assert bool(torch.isfinite(loss))
    # Atomic baseline on the same inputs: (rate.detach() * nll).sum().
    assert float(loss.detach()) == pytest.approx(1.0 * 0.5 + (-0.5) * 1.5)
    grad = torch.autograd.grad(loss, nll)[0]
    assert grad.tolist() == pytest.approx([1.0, -0.5])


def test_non_finite_inputs_raise_instead_of_falling_back():
    algo = _make_algo("align")
    credits, _ = _credits([1.0, float("inf")])
    with pytest.raises(FloatingPointError, match="non-finite inputs"):
        algo._partition_loss(
            credits,
            _atoms(),
            {},
            0,
            student_logits=torch.randn(3, 5),
            student_labels=torch.tensor([1, 2, 3]),
            student_hidden=torch.randn(3, 4),
        )


def test_healthy_path_reports_zero_fallback():
    algo = _make_algo("trust_r")
    credits, _ = _credits([1.0, 0.5])
    loss, metrics = algo._partition_loss(
        credits,
        _atoms(),
        {},
        0,
        student_logits=torch.randn(3, 5),
        student_labels=torch.tensor([1, 2, 3]),
        student_hidden=torch.randn(3, 4),
    )
    assert float(metrics["mp_opd_numeric_fallback_fraction"]) == 0.0
    assert bool(torch.isfinite(loss))
    assert float(metrics["mp_opd_trust_scope"]) == 0.0


def _stash_response(n_atoms, boundaries, n_token_rows, seed):
    generator = torch.Generator().manual_seed(seed)
    atoms = [
        _atom("one_to_one" if i % 2 == 0 else "multi_token", start, end)
        for i, (start, end) in enumerate(boundaries)
    ]
    credits = SimpleNamespace(
        rate=torch.randn(n_atoms, generator=generator, dtype=torch.float32),
        weight=torch.ones(n_atoms),
        base_credit=torch.ones(n_atoms),
        current_nll=torch.randn(n_atoms, generator=generator, dtype=torch.float32),
        student_token_nll=torch.randn(n_token_rows, generator=generator, dtype=torch.float32),
    )
    return (
        credits,
        atoms,
        torch.randn(n_token_rows, 5, generator=generator, dtype=torch.float32),
        torch.randint(0, 5, (n_token_rows,), generator=generator),
        torch.randn(n_token_rows, 4, generator=generator, dtype=torch.float32),
    )


def test_trust_b_concatenates_covered_prefixes_only():
    # Each response's token rows extend past its atoms (trailing masked
    # tokens). Concatenating whole rows would leave those as gaps between the
    # rebased ranges and the Gram routine would fail closed; only the covered
    # prefixes may join the shared axis.
    algo = _make_algo("trust_b")
    stash = [
        _stash_response(2, [(0, 2), (2, 4)], 6, seed=11),
        _stash_response(2, [(0, 1), (1, 3)], 4, seed=12),
    ]
    loss, metrics = algo._trust_b_loss(stash)
    assert bool(torch.isfinite(loss))
    assert torch.isfinite(metrics["mp_opd_trust_lambda"])
    assert "mp_opd_numeric_fallback_fraction" not in metrics
    assert float(metrics["mp_opd_trust_scope"]) == 1.0
