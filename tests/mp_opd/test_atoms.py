from pathlib import Path

import pytest

from kdflow.algorithms._mp_opd_atoms import AtomizationFailure, SimCTAtomizer


class FakeTokenizer:
    def __init__(self, pieces, eos_token_id=99, added=None):
        self.pieces = pieces
        self.eos_token_id = eos_token_id
        self._added = added or {}

    def decode(self, ids, **_kwargs):
        return "".join(self.pieces[int(value)] for value in ids)

    def get_added_vocab(self):
        return self._added


class PrefixTokenizer(FakeTokenizer):
    def decode(self, ids, **_kwargs):
        return self.pieces[tuple(int(value) for value in ids)]


def test_atom_coverage_order_no_overlap_and_exact_bytes():
    student = FakeTokenizer({1: "a", 2: "b", 3: "c", 99: "<eos>"})
    teacher = FakeTokenizer({10: "ab", 11: "c", 98: "<end>"}, eos_token_id=98)
    result = SimCTAtomizer(student, teacher).atomize([1, 2, 3, 99], [10, 11, 98], sample_id="s")
    assert result.valid
    assert [(a.student_start, a.student_end) for a in result.atoms] == [(0, 2), (2, 3)]
    assert [(a.teacher_start, a.teacher_end) for a in result.atoms] == [(0, 1), (1, 2)]
    assert [(a.byte_start, a.byte_end) for a in result.atoms] == [(0, 2), (2, 3)]
    assert result.covered_student_events == 3 and result.covered_teacher_events == 2
    assert result.masked_student_eos == result.masked_teacher_eos == 1


@pytest.mark.parametrize(
    ("student_pieces", "teacher_pieces", "expected"),
    [
        ({1: "\ufffd"}, {2: "x"}, AtomizationFailure.REPLACEMENT_CHARACTER.value),
        ({1: "é"}, {2: "e\u0301"}, AtomizationFailure.NORMALIZATION_MISMATCH.value),
        ({1: ""}, {2: ""}, AtomizationFailure.EMPTY_DECODE.value),
    ],
)
def test_atomizer_fails_closed(student_pieces, teacher_pieces, expected):
    result = SimCTAtomizer(FakeTokenizer(student_pieces), FakeTokenizer(teacher_pieces)).atomize(
        [next(iter(student_pieces))], [next(iter(teacher_pieces))], sample_id="bad"
    )
    assert not result.valid and result.failure_reason == expected


def test_unsupported_added_token_fails_closed():
    result = SimCTAtomizer(
        FakeTokenizer({1: "x"}, added={"<x>": 1}), FakeTokenizer({2: "x"})
    ).atomize([1], [2], sample_id="added")
    assert not result.valid
    assert result.failure_reason == AtomizationFailure.UNSUPPORTED_ADDED_TOKEN.value


def test_transient_replacement_prefix_is_skipped_as_a_boundary():
    student = PrefixTokenizer({(1,): "\ufffd", (1, 2): "é"})
    teacher = PrefixTokenizer({(10,): "é"})
    result = SimCTAtomizer(student, teacher).atomize([1, 2], [10], sample_id="utf8")
    assert result.valid
    assert len(result.atoms) == 1
    assert result.atoms[0].student_token_count == 2
    assert result.atoms[0].teacher_token_count == 1
    assert result.atoms[0].byte_end == len("é".encode("utf-8"))


def test_empty_response_behavior_is_explicit():
    result = SimCTAtomizer(FakeTokenizer({}), FakeTokenizer({})).atomize([], [], sample_id="empty")
    assert not result.valid and result.failure_reason == AtomizationFailure.EMPTY_RESPONSE.value


def test_real_llama_qwen_tokenizer_pair_exact_response_bytes():
    tokenizers = pytest.importorskip("tokenizers")
    llama_path = Path("/home/tung/.cache/mp-opd-tokenizers/llama-3.2-1b-tokenizer.json")
    qwen_path = Path(
        "/home/tung/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
        "snapshots/a09a35458c702b33eeacc393d103063234e8bc28/tokenizer.json"
    )
    if not llama_path.is_file() or not qwen_path.is_file():
        pytest.skip("real tokenizer cache fixture is unavailable")

    class Wrapper:
        eos_token_id = None
        def __init__(self, path):
            self.inner = tokenizers.Tokenizer.from_file(str(path))
        def decode(self, ids, **kwargs):
            return self.inner.decode(ids, skip_special_tokens=False)
        def get_added_vocab(self):
            return {}

    text = "The quick brown fox jumps over 13 lazy dogs.\n"
    llama = Wrapper(llama_path)
    qwen = Wrapper(qwen_path)
    llama_ids = llama.inner.encode(text, add_special_tokens=False).ids
    qwen_ids = qwen.inner.encode(text, add_special_tokens=False).ids
    result = SimCTAtomizer(llama, qwen).atomize(llama_ids, qwen_ids, sample_id="real")
    assert result.valid
    assert result.atoms
    assert result.atoms[-1].byte_end == len(text.encode("utf-8"))
