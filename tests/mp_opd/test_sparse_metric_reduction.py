"""Real two-process Gloo collectives with differing telemetry keys and counts."""
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kdflow.metric_reduction import reduce_sparse_metrics


def _worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method="file://"+rendezvous, rank=rank, world_size=2)
    try:
        def all_reduce(tensor, op):
            assert op == "sum"
            dist.all_reduce(tensor)
            return tensor
        values = ({"loss": 2., "reason": 1., "lambda": .4} if rank == 0 else
                  {"lambda": .8, "loss": 8.})
        counts = ({"loss": 2, "reason": 1, "lambda": 2} if rank == 0 else
                  {"loss": 1, "lambda": 1})
        result = reduce_sparse_metrics(values, SimpleNamespace(all_reduce=all_reduce), counts)
        torch.save(result, Path(output)/f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_sparse_schemas_reduce_by_name_and_occurrence_count(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path/"rendezvous"), str(tmp_path)), nprocs=2, join=True)
    first = torch.load(tmp_path/"rank0.pt", weights_only=True)
    second = torch.load(tmp_path/"rank1.pt", weights_only=True)
    assert first == second
    assert first == {"lambda": (2*.4+.8)/3, "loss": 4., "reason": 1.}
