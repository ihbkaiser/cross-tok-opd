"""Phase 0.5 tests: the real Gemma<->Qwen tokenizer mismatch and the w_i > 1 path.

Brief section 17 asks for six Phase 0.5 tests. Two of them (a real cross-tokenizer
identity regression, and "an example with w_i > 1") need the actual tokenizers, which
are only present inside the B200 image with the staged volumes mounted; those tests
therefore read ``CA_STUDENT_DIR`` / ``CA_TEACHER_DIR`` and skip with an explicit message
when the fixture is absent. The rest run anywhere.

The distinction this file keeps: a test that *skips* is not a test that passed. The
in-image run is what closes the gate, and the GPU probe in
``experiments/modal/cross_atom_phase05_modal.py`` is what closes it with real logits.
"""
import json
import os
from pathlib import Path

import pytest
import torch

from kdflow.algorithms._mp_opd_atoms import SimCTAtomizer
from kdflow.algorithms._mp_opd_credit import hard_partition_loss
from kdflow.algorithms._mp_opd_credit_transform import (
    IdentityCreditTransform,
    build_credit_transform,
    credit_diagnostics,
    production_credit_batch,
)

ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / 'experiments/mp_opd/cross_atom_phase05_texts.json'
STUDENT_DIR = os.environ.get('CA_STUDENT_DIR')
TEACHER_DIR = os.environ.get('CA_TEACHER_DIR')


def corpus():
    return json.loads(CORPUS_PATH.read_text())['items']


def corpus_without_real_tokenizers():
    return {item['id']: dict(item) for item in corpus()}


def real_atomizer():
    if not (STUDENT_DIR and TEACHER_DIR):
        pytest.skip('CA_STUDENT_DIR / CA_TEACHER_DIR are not set: no real tokenizer fixture')
    transformers = pytest.importorskip('transformers')
    student = transformers.AutoTokenizer.from_pretrained(STUDENT_DIR)
    teacher = transformers.AutoTokenizer.from_pretrained(TEACHER_DIR)
    return SimCTAtomizer(student, teacher), student, teacher


def atomize_corpus():
    atomizer, student, teacher = real_atomizer()
    rows = []
    for item in corpus():
        response = item['response']
        stu = student.encode(response, add_special_tokens=False)
        tea = teacher.encode(response, add_special_tokens=False)
        if student.eos_token_id is not None and (not stu or stu[-1] != student.eos_token_id):
            stu = stu + [student.eos_token_id]
        if teacher.eos_token_id is not None and (not tea or tea[-1] != teacher.eos_token_id):
            tea = tea + [teacher.eos_token_id]
        rows.append((item['id'], atomizer.atomize(stu, tea, sample_id=item['id']), stu, tea))
    return rows


def legacy_atomic_loss(current_nll, base, weight):
    partition = tuple((index, index + 1) for index in range(current_nll.numel()))
    return hard_partition_loss(current_nll, base, weight, partition)


# ---------------------------------------------------------------------------
# The corpus itself
# ---------------------------------------------------------------------------


def test_corpus_covers_the_required_structure_and_is_at_least_sixteen_items():
    items = corpus()
    assert len(items) >= 16
    tasks = {item['task'] for item in items}
    # arithmetic reasoning, symbolic math, code, code with indentation/punctuation,
    # short natural language, and multi-byte Unicode.
    assert {'gsm8k', 'math500', 'mbpp', 'lcb_v6_raw', 'natural', 'unicode'} <= tasks
    assert len({item['id'] for item in items}) == len(items)
    assert any('\n' in item['response'] for item in items)
    assert any(any(ord(ch) > 127 for ch in item['response']) for item in items)
    assert all(item['prompt'] and item['response'] for item in items)


# ---------------------------------------------------------------------------
# Section 17.1/17.2 -- real cross-tokenizer atoms and a real w_i > 1 atom
# ---------------------------------------------------------------------------


def test_real_cross_tokenizer_atomization_produces_w_gt_1_atoms():
    rows = atomize_corpus()
    ok = [(name, result) for name, result, _stu, _tea in rows if result.valid]
    assert ok, 'no corpus item atomized: ' + json.dumps(
        {name: result.failure_reason for name, result, _s, _t in rows if not result.valid})
    weights = [atom.student_token_count for _name, result in ok for atom in result.atoms]
    assert max(weights) > 1, 'cross-tokenizer atoms still all have w_i = 1'
    assert any(atom.boundary_type == 'multi_token' for _n, result in ok for atom in result.atoms)


def test_real_cross_tokenizer_terminal_handling_is_accounted_for():
    for name, result, stu, tea in atomize_corpus():
        if not result.valid:
            continue
        assert result.covered_student_events + result.masked_student_eos == len(stu), name
        assert result.covered_teacher_events + result.masked_teacher_eos == len(tea), name


def test_real_cross_tokenizer_identity_regression_on_real_weights():
    """The real ``w_i`` vector drives the b/w division; the credits are synthetic.

    Full-path equivalence with real logits is established by the GPU probe, not here.
    This test isolates the arithmetic: for every atom the singleton partition must
    reproduce ``sum_i (b_i / w_i) * nll_i`` through the identity operator.
    """
    rows = atomize_corpus()
    generator = torch.Generator().manual_seed(1234)
    checked = 0
    for name, result, _stu, _tea in rows:
        if not result.valid:
            continue
        weight = torch.tensor([atom.student_token_count for atom in result.atoms],
                              dtype=torch.float32)
        base = torch.randn(weight.numel(), generator=generator)
        nll = torch.rand(weight.numel(), generator=generator) * 4.0
        rate = base / weight
        batch = production_credit_batch(rate, base, weight)
        identity_loss = float(
            (IdentityCreditTransform()(batch, training=True).effective_credit * nll).sum())
        reference = float(legacy_atomic_loss(nll, base, weight))
        assert identity_loss == pytest.approx(reference, rel=1e-5, abs=1e-5), name
        checked += 1
    assert checked > 0


def test_w_gt_1_arithmetic_is_exact_for_an_explicit_example():
    base = torch.tensor([2.0, -6.0, 7.5])
    weight = torch.tensor([2.0, 3.0, 5.0])
    rate = base / weight
    assert torch.allclose(rate, torch.tensor([1.0, -2.0, 1.5]))
    nll = torch.tensor([1.0, 2.0, 3.0])
    batch = production_credit_batch(rate, base, weight)
    identity = float((IdentityCreditTransform()(batch, training=True).effective_credit * nll).sum())
    assert identity == pytest.approx(1.0 * 1.0 + -2.0 * 2.0 + 1.5 * 3.0)
    assert identity == pytest.approx(float(legacy_atomic_loss(nll, base, weight)))


# ---------------------------------------------------------------------------
# Section 17.3/17.4 -- masked atoms and invalid boundaries
# ---------------------------------------------------------------------------


def test_masked_atom_stays_atomic_on_a_real_length_vector():
    rows = [result for _n, result, _s, _t in atomize_corpus() if result.valid]
    length = max(len(result.atoms) for result in rows)
    rate = torch.linspace(-1.0, 1.0, length)
    valid = torch.ones(length, dtype=torch.bool)
    valid[length // 2] = False
    from kdflow.algorithms._mp_opd_credit_transform import AtomCreditBatch
    index = torch.arange(length)
    batch = AtomCreditBatch(
        rate_credit=rate,
        base_credit=rate.clone(),
        token_count=torch.ones(length),
        valid_mask=valid,
        seq_ids=torch.zeros(length, dtype=torch.long),
        atom_positions=index,
    )
    for name in ('forward', 'backward', 'shuffle'):
        applied = build_credit_transform(name, lam=0.5, shuffle_seed=7)(
            batch, training=True).effective_credit
        assert float(applied[length // 2]) == float(rate[length // 2]), name
    forward = build_credit_transform('forward', lam=0.5)(batch, training=True).effective_credit
    assert float(forward[length // 2 - 1]) == float(rate[length // 2 - 1])


def test_no_transfer_across_the_terminal_boundary():
    length = 9
    rate = torch.linspace(-1.0, 1.0, length)
    from kdflow.algorithms._mp_opd_credit_transform import AtomCreditBatch
    batch = AtomCreditBatch(
        rate_credit=rate,
        base_credit=rate.clone(),
        token_count=torch.ones(length),
        valid_mask=torch.ones(length, dtype=torch.bool),
        seq_ids=torch.zeros(length, dtype=torch.long),
        atom_positions=torch.arange(length),
    )
    for name in ('forward', 'causal_kernel'):
        applied = build_credit_transform(name, lam=0.25, horizon=3)(batch, training=True)
        assert float(applied.effective_credit[-1]) == pytest.approx(float(rate[-1])), name


# ---------------------------------------------------------------------------
# Section 17.6 -- non-finite telemetry must not be swallowed
# ---------------------------------------------------------------------------


def test_non_finite_credit_propagates_into_telemetry():
    """A non-finite credit must stay visible so the trainer's finite guard fires.

    Silently zeroing it would turn a broken run into a plausible-looking one.
    """
    rate = torch.tensor([1.0, float('nan'), 3.0])
    batch = production_credit_batch(rate, rate.clone(), torch.ones(3))
    diagnostics = IdentityCreditTransform()(batch, training=True).diagnostics
    assert not all(bool(torch.isfinite(value).all()) for value in diagnostics.values())
    trainer_guard = torch.stack([torch.isfinite(value.detach()).all()
                                 for value in diagnostics.values()])
    assert not bool(trainer_guard.all())


def test_zero_variance_credits_do_not_trip_the_finite_guard():
    rate = torch.full((5,), 0.25)
    batch = production_credit_batch(rate, rate.clone(), torch.ones(5))
    diagnostics = credit_diagnostics(batch, rate, name='identity', code=0)
    assert all(bool(torch.isfinite(value).all()) for value in diagnostics.values())
