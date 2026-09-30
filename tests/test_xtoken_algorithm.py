import pytest
import torch

from kdflow.algorithms.xtoken import _Fp32SparseMM, _chunk_average
from kdflow.arguments.distillation_args import DistillationArguments


def test_xtoken_arguments_require_audited_projection():
    with pytest.raises(ValueError, match="xtoken_projection_path"):
        DistillationArguments(kd_algorithm="xtoken")


def test_chunk_average_ignores_negative_chunk_ids():
    values = torch.tensor([[1.0, 3.0], [9.0, 9.0], [3.0, 5.0]])
    averages, counts = _chunk_average(values, torch.tensor([0, -1, 0]), 2)
    torch.testing.assert_close(averages, torch.tensor([[2.0, 4.0], [0.0, 0.0]]))
    torch.testing.assert_close(counts, torch.tensor([2.0, 0.0]))


def test_sparse_projection_is_value_and_gradient_preserving():
    indices = torch.tensor([[0, 1, 2], [0, 1, 0]])
    values = torch.tensor([1.0, 0.25, 0.75])
    matrix = torch.sparse_coo_tensor(indices, values, (3, 2)).coalesce()
    dense = torch.tensor([[0.2], [0.3], [0.5]], requires_grad=True)
    projected = _Fp32SparseMM.apply(matrix, dense)
    torch.testing.assert_close(projected, torch.tensor([[0.575], [0.075]]))
    projected.sum().backward()
    torch.testing.assert_close(dense.grad, torch.tensor([[1.0], [0.25], [0.75]]))
