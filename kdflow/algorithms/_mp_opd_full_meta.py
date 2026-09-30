"""Streamed full-parameter one-step AdamW hypergradient.

The optimizer state is held constant (truncated one-step objective). A virtual
student is installed temporarily, always rolled back, and never checkpointed.
Only one rollout microbatch's higher-order graph is retained at a time.
"""
from contextlib import contextmanager
import torch
import json
from kdflow.training_checkpoint import capture_rng, restore_rng


def memory_event(phase, device, **details):
    if torch.device(device).type != 'cuda':
        return
    print('FULL_META_MEMORY '+json.dumps(dict(phase=phase,
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        **details)),flush=True)


def streamed_inner_losses(batches, loader, training_step, device):
    """Callbacks retain source handles only; autograd owns current inputs until backward."""
    def loss_at(index):
        batch = loader(batches[index])
        memory_event('inner_forward',device,index=index,
            shape=list(batch['stu_input_ids'].shape),
            teacher_hidden_shape=list(batch['teacher_hiddens'].shape)
                if torch.is_tensor(batch.get('teacher_hiddens')) else None)
        return training_step(batch)['loss']/len(batches)
    return [lambda index=index:loss_at(index) for index in range(len(batches))]


class ForwardParameterBridge:
    """Capture the actual forward tensors for a single-rank FSDP2 model.

    Hooks run after FSDP unshards each module. Optimizer DTensors stay the
    authoritative state; gradients are returned in their original layout.
    """
    def __init__(self, model):
        from torch.distributed.tensor import DTensor
        from torch.distributed.fsdp import FSDPModule
        self.entries = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if any(isinstance(p, DTensor) and p.device_mesh.size() != 1 for _, p in self.entries):
            raise ValueError('Full meta bridge supports exactly one rank')
        self.captured = {}
        self.handles = []
        self.policies = []
        for mod in model.modules():
            if isinstance(mod, FSDPModule):
                state = mod._get_fsdp_state()
                group = state._fsdp_param_group
                if group is not None:
                    # Torch 2.11 has public setters but no getters. Preserve the
                    # exact policy objects instead of guessing the root default.
                    self.policies.append((mod, state, group,
                        state._auto_reshard_after_forward,
                        group.post_forward_mesh_info, group.reshard_after_backward))
                    mod.set_reshard_after_forward(False, recurse=False)
                    mod.set_reshard_after_backward(False, recurse=False)
        for prefix, mod in model.named_modules():
            def capture(module, inputs, prefix=prefix):
                for name, p in module.named_parameters(recurse=False):
                    if p.requires_grad:
                        self.captured[prefix + '.' + name if prefix else name] = p
            self.handles.append(mod.register_forward_pre_hook(capture))

    def close(self):
        for handle in self.handles: handle.remove()
        self.captured.clear()
        for mod, state, group, auto, mesh, backward in self.policies:
            mod.reshard()
            state._auto_reshard_after_forward = auto
            group.post_forward_mesh_info = mesh
            group.reshard_after_backward = backward

    def grad(self, loss, params, **kwargs):
        from torch.distributed.tensor import DTensor
        if [id(p) for p in params] != [id(p) for _, p in self.entries]:
            raise ValueError('Meta bridge optimizer ordering changed')
        missing = [n for n, _ in self.entries if n not in self.captured]
        if missing: raise RuntimeError('Missing forward parameters: ' + ','.join(missing[:5]))
        targets = tuple(self.captured[n] for n, _ in self.entries)
        values = torch.autograd.grad(loss, targets, **kwargs)
        result = []
        for (_, p), value in zip(self.entries, values):
            if value is not None:
                value = value.to(p.dtype)
            if value is not None and isinstance(p, DTensor) and not isinstance(value, DTensor):
                value = DTensor.from_local(value, p.device_mesh, p.placements, run_check=False)
            result.append(value)
        return tuple(result)


def adam_value(p, g, state, group):
    if group.get("amsgrad") or group.get("maximize"):
        raise ValueError("Full meta supports standard, non-AMSGrad AdamW only")
    beta1, beta2 = group["betas"]
    step = float(state.get("step", 0)) + 1
    m = state.get("exp_avg", torch.zeros_like(p)).detach()
    v = state.get("exp_avg_sq", torch.zeros_like(p)).detach()
    m1 = beta1*m + (1-beta1)*g
    v1 = beta2*v + (1-beta2)*g.square()
    denom = (v1 / (1-beta2**step)).sqrt() + group["eps"]
    return p.detach()*(1-group["lr"]*group["weight_decay"]) - group["lr"]*m1/(1-beta1**step)/denom


@contextmanager
def offload_adam_moments_scope(optimizer, parameters, *, enabled=False, device=None):
    """Temporarily move CUDA Adam moments out of the higher-order phase.

    The one-step objective holds optimizer state fixed, and the mixed VJP below
    the optimizer map does not read the moments. Only ``exp_avg`` and
    ``exp_avg_sq`` are moved; scalar ``step`` and every other state entry stay
    untouched. No old tensor is retained in the restore metadata, so the move
    can actually release its CUDA storage. Restoration is fail-closed through
    the context manager's ``finally`` block.
    """
    if not enabled:
        yield {"enabled": False, "offloaded_bytes": 0}
        return
    entries = []
    offloaded_bytes = 0
    try:
        for parameter in parameters:
            state = optimizer.state.get(parameter, {})
            for name in ("exp_avg", "exp_avg_sq"):
                value = state.get(name)
                if value is None:
                    continue
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Adam state {name} must be a tensor")
                if value.device.type != "cuda":
                    continue
                original_device = value.device
                offloaded_bytes += value.numel() * value.element_size()
                state[name] = value.to("cpu", non_blocking=False)
                entries.append((state, name, original_device))
        if device is not None and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        yield {"enabled": True, "offloaded_bytes": offloaded_bytes}
    finally:
        for state, name, original_device in reversed(entries):
            value = state.get(name)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"Missing Adam state during restore: {name}")
            state[name] = value.to(original_device, non_blocking=False)
        if device is not None and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)


def clipped(grads, max_norm):
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach(), 2) for g in grads]), 2)
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite virtual gradient")
    scale = (max_norm/(norm+1e-6)).clamp(max=1) if max_norm > 0 else norm.new_tensor(1.)
    return [g*scale.to(g.dtype) for g in grads], norm, scale


def full_meta_step(parameters, optimizer, energy, energy_optimizer, inner_losses, meta_losses, *, max_norm, refresh_parameters=lambda:None, parameter_grad=torch.autograd.grad, offload_adam_moments=False):
    """Each callback returns its already normalized contribution to ONE batch.

    Inner callbacks keep credits/features detached, but expose energy marginals
    to autograd. Meta callbacks use independent reference data. No real student
    update occurs here: caller recomputes detached rates and uses its optimizer.
    """
    params = tuple(parameters)
    groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    if set(groups) != {id(p) for p in params}:
        raise ValueError("Virtual and real optimizer parameter sets differ")
    if any(p.grad is not None for p in params):
        raise ValueError("Meta update requires an optimizer boundary")
    phi = tuple(energy.parameters())
    initial_rng = capture_rng()
    g = [torch.zeros_like(p) for p in params]
    # Detached accumulation avoids retaining B=64 activation graphs.
    for index,loss_fn in enumerate(inner_losses):
        loss = loss_fn()
        memory_event('inner_first_backward',params[0].device,index=index)
        part = parameter_grad(loss, params, allow_unused=True)
        if all(value is None for value in part):
            raise RuntimeError('Meta inner loss is disconnected from optimizer parameters; FSDP parameter views require an explicit autograd bridge')
        for target, value in zip(g, part):
            if value is not None: target.add_(value.detach())
        # Do not retain a full gradient tuple across the next model forward.
        del part, loss, value, target
    cg, norm, scale = clipped(g, max_norm)
    refresh_parameters()
    originals = [p.detach().cpu().clone() for p in params]
    outer = [torch.zeros_like(p) for p in params]
    meta_value = 0.
    try:
        with torch.no_grad():
            for p, grad in zip(params, cg):
                p.copy_(adam_value(p, grad, optimizer.state.get(p, {}), groups[id(p)]))
        for loss_fn in meta_losses:
            loss = loss_fn()
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite meta NLL")
            meta_value += float(loss.detach())
            part = parameter_grad(loss, params, allow_unused=True)
            if all(value is None for value in part):
                raise RuntimeError('Meta outer loss is disconnected from optimizer parameters')
            for target, value in zip(outer, part):
                if value is not None: target.add_(value.detach())
            del part, loss, value, target
    finally:
        refresh_parameters()
        with torch.no_grad():
            for p, old in zip(params, originals): p.copy_(old.to(p.device))
    del originals
    # VJP of the optimizer, holding pre-step moments fixed. Work parameter by
    # parameter so optimizer-expression graphs do not scale with the full model.
    u = []
    for p, grad, vector in zip(params, cg, outer):
        group=groups[id(p)]; state=optimizer.state.get(p,{})
        b1,b2=group['betas']; step=float(state.get('step',0))+1
        m1=b1*(state['exp_avg'] if 'exp_avg' in state else torch.zeros_like(p))+(1-b1)*grad
        v1=b2*(state['exp_avg_sq'] if 'exp_avg_sq' in state else torch.zeros_like(p))+(1-b2)*grad.square()
        root=(v1/(1-b2**step)).sqrt(); denom=root+group['eps']
        # At v=g=0, the composite Adam map g/(|g|+eps) has derivative
        # 1/eps; naïve autograd through sqrt produces 0*inf -> NaN.
        root_safe=torch.where(root>0,root,torch.ones_like(root))
        correction=torch.where(root>0,m1*(1-b2)*grad/((1-b2**step)*root_safe*denom.square()),0.)
        tangent=(-group['lr']/(1-b1**step)*((1-b1)/denom-correction)*vector).detach()
        if not torch.isfinite(tangent).all():
            raise FloatingPointError("Nonfinite AdamW VJP; inspect zero-variance/numerical edge")
        u.append(tangent)
        del m1, v1, root, denom, root_safe, correction, tangent
    del cg, outer
    # VJP through GLOBAL gradient clipping, including its cross-parameter term.
    if max_norm > 0 and float(scale) < 1:
        dot = sum((a*b).sum() for a,b in zip(u,g))
        correction = scale*dot/(norm*(norm+1e-6))
        u = [scale.to(a.dtype)*a - correction.to(b.dtype)*b for a,b in zip(u,g)]
    del g
    restore_rng(initial_rng)
    hyper = [torch.zeros_like(p) for p in phi]
    with offload_adam_moments_scope(optimizer, params,
                              enabled=offload_adam_moments,
                              device=params[0].device) as moment_info:
        memory_event('adam_moments_offloaded', params[0].device,
                     offloaded_bytes=moment_info['offloaded_bytes'])
        for index,loss_fn in enumerate(inner_losses):
            loss = loss_fn()
            memory_event('inner_second_backward',params[0].device,index=index)
            part = parameter_grad(loss, params, create_graph=True, allow_unused=True)
            active = [(grad, vector) for grad, vector in zip(part,u) if grad is not None and grad.requires_grad]
            if active:
                hg = torch.autograd.grad(tuple(x for x,_ in active), phi,
                                         grad_outputs=tuple(v for _,v in active), allow_unused=True)
                for target, value in zip(hyper,hg):
                    if value is not None: target.add_(value.detach())
                del hg, target, value
            del active, part, loss
            memory_event('inner_second_released',params[0].device,index=index)
    if not all(torch.isfinite(x).all() for x in hyper):
        raise FloatingPointError("Nonfinite full energy hypergradient")
    energy_optimizer.zero_grad(set_to_none=True)
    old_energy=[p.detach().clone() for p in phi]
    for p, value in zip(phi,hyper): p.grad = value
    energy_optimizer.step(); energy_optimizer.zero_grad(set_to_none=True)
    # Real student pass sees exactly the original inner-forward RNG sequence.
    restore_rng(initial_rng)
    return {"mp_opd_meta_nll": meta_value,
            "mp_opd_energy_parameter_delta": float(sum((p.detach()-old).square().sum() for p,old in zip(phi,old_energy)).sqrt()),
            "mp_opd_energy_gradient_norm": float(sum(x.square().sum() for x in hyper).sqrt()),
            "mp_opd_virtual_gradient_norm": float(norm)}
