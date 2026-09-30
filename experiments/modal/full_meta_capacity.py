"""B200 full-size Gemma capacity stress test, NOT a campaign benchmark.

Uses cached base weights, synthetic token NLL and a small differentiable energy
to exercise the production full-meta AdamW/VJP. No teacher/rollout/partition DP.
Two inner microbatches exercise accumulation; this is not a B64 timing claim.
"""
import argparse
import importlib.util
import json
import statistics
import time
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from transformers import AutoModelForCausalLM

p = argparse.ArgumentParser()
p.add_argument('--micro', type=int, required=True)
p.add_argument('--meta-micro', type=int, default=4)
p.add_argument('--length', type=int, default=1024)
p.add_argument('--steps', type=int, default=1)
p.add_argument('--batch', type=int, default=0, help='0: two-microbatch capacity probe; 64: timing')
p.add_argument('--attention', choices=('eager', 'sdpa'), default='eager')
p.add_argument('--offload-adam-moments', action='store_true')
a = p.parse_args()
if a.steps < 1 or a.micro < 1 or 64 % a.micro or 16 % a.meta_micro:
    raise ValueError('Invalid steps or microbatch divisors')
batch = a.batch or 2*a.micro
if batch % a.micro: raise ValueError('Batch must divide into whole microbatches')
spec = importlib.util.spec_from_file_location('full_meta', '/opt/overlay/kdflow/algorithms/_mp_opd_full_meta.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
torch.cuda.set_device(0)
dist.init_process_group('nccl', init_method='tcp://127.0.0.1:29721', rank=0, world_size=1)
report = dict(micro=a.micro, meta_micro=a.meta_micro, length=a.length,
    batch=batch, requested_steps=a.steps, steps=[],
    attention=a.attention,
    offload_adam_moments=a.offload_adam_moments,
    scope='synthetic full-size student/meta timing; excludes teacher, rollout and partition DP')
try:
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained('/assets/student', local_files_only=True,
        dtype=torch.float32, attn_implementation=a.attention).cuda().train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.model.layers: fully_shard(layer, mp_policy=policy)
    fully_shard(model, mp_policy=policy)
    params = tuple(model.parameters())
    energy = torch.nn.Linear(1, 1, device='cuda')
    opt = torch.optim.AdamW(params, lr=1e-6, betas=(.9,.98), eps=1e-8, weight_decay=0.)
    eo = torch.optim.AdamW(energy.parameters(), lr=1e-3, weight_decay=0.)
    # Include optimizer momentum memory even on the first measured update.
    for param in params:
        opt.state[param].update(step=torch.tensor(16.),
            exp_avg=torch.zeros_like(param), exp_avg_sq=torch.zeros_like(param))
    ids = torch.randint(10, 10000, (max(a.micro, a.meta_micro), a.length), device='cuda')
    def loss(size, weighted):
        logits = model(ids[:size]).logits.float()
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
            ids[:size].roll(-1, 1).flatten())
        return nll * energy(torch.ones(1,1,device='cuda')).sigmoid().squeeze() if weighted else nll
    def refresh():
        for m in model.modules():
            if isinstance(m, FSDPModule): m.reshard()
    torch.cuda.reset_peak_memory_stats()
    for step in range(a.steps):
        start = time.perf_counter()
        bridge = module.ForwardParameterBridge(model)
        try:
            metrics = module.full_meta_step(params, opt, energy, eo,
                [lambda: loss(a.micro, True)/(batch//a.micro) for _ in range(batch//a.micro)],
                [lambda: loss(a.meta_micro, False)/(16//a.meta_micro) for _ in range(16//a.meta_micro)],
                max_norm=1., parameter_grad=bridge.grad, refresh_parameters=refresh,
                offload_adam_moments=a.offload_adam_moments)
        finally:
            bridge.close()
        for _ in range(batch//a.micro):
            (loss(a.micro, True)/(batch//a.micro)).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.)
        opt.step(); opt.zero_grad(set_to_none=True); refresh(); torch.cuda.synchronize()
        item = dict(step=step+1, seconds=time.perf_counter()-start, metrics=metrics)
        report['steps'].append(item)
        print('CAPACITY_STEP '+json.dumps(item), flush=True)
    warm = report['steps'][2:]
    report.update(status='pass', warm_median_seconds=statistics.median(x['seconds'] for x in warm) if warm else None)
except torch.OutOfMemoryError:
    report.update(status='oom')
except Exception as error:
    report.update(status='error', error_type=type(error).__name__, error=str(error))
    raise
finally:
    report.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    print('CAPACITY_RESULT '+json.dumps(report), flush=True)
    dist.destroy_process_group()
