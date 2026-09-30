"""Tiny GPU diagnostic of the production full-meta function under FSDP2."""
import json
import os
import importlib.util
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
spec = importlib.util.spec_from_file_location('full_meta', '/opt/overlay/kdflow/algorithms/_mp_opd_full_meta.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
full_meta_step = module.full_meta_step


def run(sharded, nested=False, mixed=False, gemma=False):
    torch.manual_seed(42)
    model = (torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
             if nested else torch.nn.Linear(4, 2, bias=False)).cuda()
    if gemma:
        from transformers import Gemma2Config, Gemma2ForCausalLM
        config = Gemma2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, max_position_embeddings=64, attn_logit_softcapping=50.,
            final_logit_softcapping=30., use_cache=False)
        config._attn_implementation = 'eager'
        model = Gemma2ForCausalLM(config).cuda().train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if sharded:
        kwargs = {'mp_policy': MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)} if mixed else {}
        if gemma:
            for layer in model.model.layers: fully_shard(layer, **kwargs)
        elif nested:
            fully_shard(model[0], **kwargs)
            fully_shard(model[2], **kwargs)
        fully_shard(model, **kwargs)
    energy = torch.nn.Linear(4, 1, bias=False).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(.9,.98))
    eo = torch.optim.AdamW(energy.parameters(), lr=1e-3)
    x = torch.randn(8, 4, device='cuda')
    y = torch.randn(8, 2, device='cuda')
    ids = torch.randint(0,32,(2,8),device='cuda')
    def gemma_forward(tokens):
        if mixed and not sharded:
            # Match FSDP's parameter casting, including embeddings and norms;
            # autocast alone leaves those FP32 and is a different computation.
            cast = {n:p.to(torch.bfloat16) for n,p in model.named_parameters()}
            return torch.func.functional_call(model, cast, (tokens,)).logits.float()
        return model(tokens).logits.float()
    def forward(x):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            return model(x).float()
    def inner():
        if gemma:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
                logits = gemma_forward(ids)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1,32), ids.roll(-1,1).flatten(), reduction='none')
            return (loss.reshape(2,8)*energy(x).sigmoid().flatten()).mean()
        return (energy(x).sigmoid() * (forward(x)-y).square()).mean()
    def outer():
        if gemma:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
                logits=gemma_forward((ids+1)%32)
            return logits.square().mean()
        return (forward(x + .1) - y).square().mean()
    params = tuple(model.parameters())
    # Materialize Adam moments before the meta step so the diagnostic exercises
    # the actual offload/restore path instead of an empty optimizer state.
    warm_loss = inner()
    warm_loss.backward()
    torch.nn.utils.clip_grad_norm_(params, .1)
    opt.step()
    opt.zero_grad(set_to_none=True)
    eo.zero_grad(set_to_none=True)
    bridge = module.ForwardParameterBridge(model)
    loss = inner()
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    report = {'sharded': sharded, 'nested':nested, 'mixed':mixed, 'gemma':gemma, 'unused': [g is None for g in grads]}
    def refresh():
        for m in model.modules():
            if isinstance(m, FSDPModule): m.reshard()
    try:
        report['steps'] = []
        for step in range(3):
            refresh()
            before = [p.detach().clone() for p in params]
            metrics = full_meta_step(params, opt, energy, eo,
                [lambda: inner()/2, lambda: inner()/2], [outer], max_norm=.1,
                parameter_grad=bridge.grad, refresh_parameters=refresh,
                offload_adam_moments=True)
            report['steps'].append(metrics)
            assert all(torch.equal(a,b) for a,b in zip(params,before)), 'Virtual state did not roll back'
            inner().backward()
            torch.nn.utils.clip_grad_norm_(params, .1)
            opt.step(); opt.zero_grad(set_to_none=True)
            refresh()
    except Exception as exc:
        report['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        bridge.close()
    opt.zero_grad(set_to_none=True)
    inner().backward()
    report['backward_has_gradient'] = [p.grad is not None for p in model.parameters()]
    print('META_DEBUG ' + json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:29671', rank=0, world_size=1)
    torch.cuda.set_device(0)
    print('ENV ' + json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0)}), flush=True)
    try:
        for nested, mixed, gemma in ((False, False, False), (True, False, False), (True, True, False), (False, False, True), (False, True, True)):
            plain = run(False, nested, mixed, gemma)
            sharded = run(True, nested, mixed, gemma)
            assert 'error' not in plain and 'error' not in sharded, (plain, sharded)
            for a,b in zip(plain['steps'], sharded['steps']):
                for key in a:
                    assert abs(a[key]-b[key]) <= (1e-7 + (0.02 if mixed else 1e-5)*abs(a[key])), (key,a,b)
        print('BRIDGE_PARITY_PASS', flush=True)
    finally:
        dist.destroy_process_group()
