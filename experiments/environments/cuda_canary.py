"""Two-rank CUDA/BF16/NCCL and target import probe; not a training result."""
import datetime
import importlib
import json
import os
import sys

import torch
import torch.distributed as dist

target = sys.argv[1]
rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
assert torch.cuda.is_bf16_supported()
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=90))
try:
    x = torch.tensor([rank + 1.0], device='cuda')
    dist.all_reduce(x)
    assert x.item() == 3.0
    q = torch.randn(2, 64, 8, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    if target == 'simct':
        from flash_attn import flash_attn_func
        y = flash_attn_func(q, q, q, causal=True)
        importlib.import_module('sglang')
        importlib.import_module('sgl_kernel')
        importlib.import_module('kdflow')
    else:
        y = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
        importlib.import_module('nemo_automodel')
        importlib.import_module('nemo_rl.algorithms.xtoken_off_policy_distillation')
        importlib.import_module('nemo_rl.models.policy.workers.dtensor_policy_worker_v2')
    y.float().square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(q.grad).all()
    torch.cuda.synchronize()
    print(json.dumps(dict(target=target, rank=rank, torch=torch.__version__,
                          cuda=torch.version.cuda, device=torch.cuda.get_device_name(rank),
                          nccl_sum=x.item(), finite_backward=True)), flush=True)
finally:
    dist.destroy_process_group()
