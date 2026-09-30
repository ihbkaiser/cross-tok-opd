from kdflow.ray.train.teacher_actor import chunk_batch_indices


def test_chunk_batch_indices_preserves_order_and_tail() -> None:
    indices = list(range(19))
    chunks = chunk_batch_indices(indices, 8)
    assert chunks == [list(range(8)), list(range(8, 16)), list(range(16, 19))]
    assert [index for chunk in chunks for index in chunk] == indices


def test_chunk_batch_indices_clamps_nonpositive_size() -> None:
    assert chunk_batch_indices([3, 4], 0) == [[3], [4]]
