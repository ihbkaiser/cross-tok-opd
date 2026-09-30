"""Random partitioning with an inclusive lower bound; min_length=1 must not move."""
import ast as _ast
import random as _random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kdflow.algorithms.mp_opd import random_partition

ARGS = ROOT / "kdflow/arguments/distillation_args.py"


def _historical(n, max_length, seed):
    """The pre-knob draw, kept here so the default path stays pinned to it."""
    generator = _random.Random(int(seed))
    parts, cursor = [], 0
    while cursor < n:
        length = generator.randint(1, min(max_length, n - cursor))
        parts.append((cursor, cursor + length))
        cursor += length
    return tuple(parts)


@pytest.mark.parametrize("n,max_length,seed", [(0, 4, 1), (1, 4, 7), (2, 2, 3), (10, 3, 5),
                                               (17, 5, 11), (40, 2, 13), (63, 5, 29)])
def test_default_reproduces_the_historical_draw(n, max_length, seed):
    assert random_partition(n, max_length, seed) == _historical(n, max_length, seed)
    assert random_partition(n, max_length, seed, 1) == _historical(n, max_length, seed)


@pytest.mark.parametrize("seed", range(40))
def test_min_two_drops_single_atom_spans_except_a_short_tail(seed):
    parts = random_partition(30, 5, seed, 2)
    lengths = [end - start for start, end in parts]
    assert sum(lengths) == 30
    assert parts[0][0] == 0 and parts[-1][1] == 30
    assert all(end - start <= 5 for start, end in parts)
    assert all(length >= 2 for length in lengths[:-1])
    assert lengths[-1] >= 1


def test_exact_length_when_min_equals_max():
    assert [e - s for s, e in random_partition(11, 3, 5, 3)] == [3, 3, 3, 2]


def test_tiny_remainder_is_covered_not_rejected():
    assert random_partition(1, 5, 3, 2) == ((0, 1),)
    assert random_partition(4, 5, 3, 4) == ((0, 4),)


def test_same_seed_gives_the_same_partition():
    assert random_partition(25, 4, 99, 2) == random_partition(25, 4, 99, 2)


@pytest.mark.parametrize("n,max_length,min_length", [(5, 0, 1), (5, -1, 1), (5, 3, 0), (5, 3, -2), (5, 3, 4)])
def test_invalid_bounds_raise(n, max_length, min_length):
    with pytest.raises(ValueError):
        random_partition(n, max_length, 1, min_length)


def test_args_field_declares_one_as_its_default():
    """Source-level check so it holds in environments without transformers."""
    tree = _ast.parse(ARGS.read_text())
    field = None
    for node in _ast.walk(tree):
        if isinstance(node, _ast.AnnAssign) and getattr(node.target, "id", "") == "mp_opd_min_span_length":
            field = node
    assert field is not None, "mp_opd_min_span_length is not declared"
    assert _ast.literal_eval({kw.arg: kw.value for kw in field.value.keywords}["default"]) == 1
    source = ARGS.read_text()
    assert "mp_opd_min_span_length must be at least 1" in source


def test_args_dataclass_exposes_the_field_when_transformers_is_present():
    pytest.importorskip("transformers")
    from kdflow.arguments.distillation_args import DistillationArguments
    assert DistillationArguments.__dataclass_fields__["mp_opd_min_span_length"].default == 1
