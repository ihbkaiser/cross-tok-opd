"""Reduce sparse telemetry with the same collective schema on every rank."""
import torch
import torch.distributed as dist


def reduce_sparse_metrics(values, strategy, counts=None):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return values
    schemas = [None] * dist.get_world_size()
    dist.all_gather_object(schemas, list(values))
    keys = sorted(set().union(*schemas))
    weights = {k: float((counts or {}).get(k, 1)) if k in values else 0.0 for k in keys}
    device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
    packed = torch.tensor([
        [float(values.get(k, 0.0)) * weights[k] for k in keys],
        [weights[k] for k in keys],
    ], dtype=torch.float64, device=device)
    reduced = strategy.all_reduce(packed, op="sum")
    return {k: float(reduced[0, i] / reduced[1, i])
            for i, k in enumerate(keys) if float(reduced[1, i]) > 0}
