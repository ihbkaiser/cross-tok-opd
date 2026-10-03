"""Regression tests for the MP-OPD student forward position ids.

`kdflow.models.model.forward_position_ids` decides whether the non-packing forward
passes explicit `position_ids` to the HuggingFace model. That choice is not cosmetic:
on the MP-OPD parity capture `failure-zusi05xj` the same row forwarded inside a
right-padded batch differed from the row forwarded alone by up to 15.5 logits when
`position_ids` was passed, and by exactly 0.0 when it was not. The capture's longest
row is 4096 tokens = the full padded width, so its mask is all ones and its
`cumsum - 1` *is* `arange`, which is why the effect was invisible to inspection.

These tests are CPU-only and need no model: they pin the decision rule.
"""

from __future__ import annotations

import pytest
import torch

from kdflow.models.model import forward_position_ids


def test_no_mask_yields_no_positions():
    assert forward_position_ids(None) is None


def test_all_ones_mask_yields_no_positions():
    # A batch with no padding at all: HF's default already numbers it 0..n-1.
    mask = torch.ones(2, 8, dtype=torch.long)
    assert forward_position_ids(mask) is None


def test_right_padded_mask_yields_no_positions():
    # Prefix-ones is exactly right-padding: the real tokens already sit at 0..n-1,
    # and the masked tail needs no numbering.
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.long)
    assert forward_position_ids(mask) is None


def test_left_padded_mask_keeps_explicit_positions():
    # With left padding HF's default would number the pads too, so the real tokens
    # would start above 0. The explicit cumsum form restores 0..n-1, with the pads
    # forced to 1 exactly as HF itself does.
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0]], dtype=torch.long)
    positions = forward_position_ids(mask)
    assert positions is not None
    assert positions.tolist() == [[1, 1, 0, 1, 2, 3], [1, 0, 1, 2, 1, 1]]


def test_single_row_prefix_mask_yields_no_positions():
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long)
    assert forward_position_ids(mask) is None


@pytest.mark.parametrize("dtype", [torch.long, torch.bool])
def test_prefix_detection_is_dtype_agnostic(dtype):
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    assert forward_position_ids(mask.to(dtype)) is None


def test_capture_mask_shape_yields_no_positions():
    """The incident's exact mask geometry must take the no-positions path.

    `failure-zusi05xj` has 4 rows padded to width 4096 with attention-mask sums
    506 / 224 / 4096 / 644, and all four masks are plain prefixes. The longest row
    is exactly the width, so its mask is all ones. If this ever returned explicit
    positions the parity guard would start tripping again on the capped row.
    """
    width = 4096
    lengths = [506, 224, 4096, 644]
    mask = torch.zeros(4, width, dtype=torch.long)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1
    assert forward_position_ids(mask) is None


def test_mask_with_a_hole_is_not_treated_as_prefix():
    """A gap in the ones is not right padding, so the explicit form is kept.

    This is the guard against over-eagerly dropping positions: only a genuine
    prefix may skip them, because HF's default numbering is only correct then.
    """
    mask = torch.tensor([[1, 1, 0, 1, 1, 0]], dtype=torch.long)
    positions = forward_position_ids(mask)
    assert positions is not None
    assert positions.tolist() == [[0, 1, 1, 2, 3, 1]]
