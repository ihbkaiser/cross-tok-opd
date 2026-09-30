import torch

from kdflow.models.ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor


def test_torch_unpad_and_pad_round_trip_without_flash_attn_2() -> None:
    sequences = torch.tensor([[11, 12, 0], [21, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    unpadded, position_ids, rolled, pad_len, indices = unpad_and_slice_tensor(
        sequences,
        attention_mask,
        ring_attn_group=None,
    )

    assert unpadded.tolist() == [[11, 12, 21]]
    assert position_ids.tolist() == [[0, 1, 0]]
    assert rolled.tolist() == [[12, 0, 0]]
    assert pad_len == 0
    assert indices.tolist() == [0, 1, 3]
    assert gather_and_pad_tensor(
        unpadded,
        ring_attn_group=None,
        ring_attn_pad_len=0,
        indices=indices,
        batch=2,
        seqlen=3,
    ).tolist() == sequences.tolist()
