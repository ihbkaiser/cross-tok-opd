import pytest
import torch
from kdflow.ray.train.teacher_actor import chunk_batch_indices, teacher_input_trajectories


def test_chunk_batch_indices_preserves_order_and_tail() -> None:
    indices = list(range(19))
    chunks = chunk_batch_indices(indices, 8)
    assert chunks == [list(range(8)), list(range(8, 16)), list(range(16, 19))]
    assert [index for chunk in chunks for index in chunk] == indices


def test_chunk_batch_indices_clamps_nonpositive_size() -> None:
    assert chunk_batch_indices([3, 4], 0) == [[3], [4]]


@pytest.mark.parametrize("exact_mode", [False, True])
def test_teacher_text_uses_mask_ids_even_in_legacy_simct(exact_mode):
    batches = [{"tea_input_ids": torch.tensor([[0, 3, 4], [5, 6, 0]]),
                "tea_attn_mask": torch.tensor([[0, 1, 1], [1, 1, 0]])}]
    assert teacher_input_trajectories(batches, exact_mode=exact_mode) == [[3, 4], [5, 6]]


def test_teacher_multimodal_preserves_legacy_path_and_exact_rejection():
    assert teacher_input_trajectories([], image_data=[object()]) is None
    with pytest.raises(ValueError, match="text-only"):
        teacher_input_trajectories([], image_data=[object()], exact_mode=True)
