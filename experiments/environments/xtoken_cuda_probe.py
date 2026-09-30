"""Offline, setup-only CUDA localization. No optimizer, training or checkpoints."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def emit(stage, **fields):
    print(json.dumps(dict(stage=stage, rank=int(os.environ.get('RANK', '0')),
                          launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING'),
                          **fields)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['hf', 'nemo', 'ray'])
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(rank)
    models = json.loads((root/'models-evidence/models.json').read_text())
    assert models['complete'] and all(m['weights_verified'] for m in models['models'])
    teacher = next(m for m in models['models'] if m['role'] == 'teacher')
    assert teacher['revision'] == '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
    emit('start', mode=args.mode, torch=torch.__version__, cuda=torch.version.cuda)
    if args.mode == 'hf':
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            teacher['snapshot'], dtype=torch.bfloat16,
            attn_implementation='sdpa', local_files_only=True).to('cuda')
    else:
        from nemo_rl.algorithms.xtoken_off_policy_distillation import MasterConfig
        from nemo_rl.models.automodel.setup import (
            get_tokenizer, setup_distributed, setup_model_and_optimizer,
            validate_and_prepare_config,
        )
        from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers
        from omegaconf import OmegaConf
        register_omegaconf_resolvers()
        gate = json.loads((root/'smoke-2gpu-r1/config-valid.json').read_text())
        command = json.loads((Path(gate['evidence'])/'command.json').read_text())['command']
        entry = next(i for i, arg in enumerate(command) if arg.endswith('/validate_config.py'))
        config = load_config(str(root/'NeMo-RL/examples/configs/xtoken_off_policy_distillation.yaml'))
        config = parse_hydra_overrides(config, command[entry+1:])
        master = MasterConfig(**OmegaConf.to_container(config, resolve=True))
        config = master.teachers[0].policy_config()
        assert config['model_name'] == teacher['snapshot']
        tokenizer = get_tokenizer(config['tokenizer'])
        if args.mode == 'ray':
            from nemo_rl.distributed.virtual_cluster import init_ray, RayVirtualCluster
            from nemo_rl.models.policy.lm_policy import Policy
            from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
            import ray
            ACTOR_ENVIRONMENT_REGISTRY['cuda_probe_worker.BufferProbeWorker'] = ACTOR_ENVIRONMENT_REGISTRY[
                'nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2']
            init_ray()
            cluster = RayVirtualCluster(name='cuda_teacher_probe', bundle_ct_per_node_list=[2],
                                        use_gpus=True, num_gpus_per_node=2, max_colocated_worker_groups=2)
            policy = Policy(name_prefix='teacher_probe', cluster=cluster, config=config,
                            tokenizer=tokenizer, weights_path=None, optimizer_path=None,
                            init_optimizer=False, init_reference_model=False,
                            worker_extension_cls_fqn='cuda_probe_worker.BufferProbeWorker')
            policy.run_all_workers_single_data('inspect_buffers', stage='method_before_offload')
            policy.offload_after_refit()
            emit('ray_offload_ok')
            ray.shutdown()
            return
        runtime = validate_and_prepare_config(config=config, processor=None, rank=rank)
        context = setup_distributed(config=config, runtime_config=runtime)
        emit('distributed_ready')
        state = setup_model_and_optimizer(
            config=config, tokenizer=tokenizer, runtime_config=runtime,
            distributed_context=context, checkpoint_manager=None, is_vlm=False,
            init_optimizer=False, weights_path=None, optimizer_path=None)
        model = state[0]
    emit('model_constructed')
    torch.cuda.synchronize()
    emit('model_synchronized')
    for name, buffer in model.named_buffers():
        emit('buffer_before_copy', name=name, shape=list(buffer.shape),
             dtype=str(buffer.dtype), device=str(buffer.device),
             storage_bytes=buffer.untyped_storage().nbytes(), storage_offset=buffer.storage_offset())
        copied = buffer.to('cpu')
        emit('buffer_copy_ok', name=name, finite=bool(torch.isfinite(copied).all()),
             first_values=copied.flatten()[:4].float().tolist())
    emit('offload_start')
    model.to('cpu')
    torch.cuda.synchronize()
    emit('offload_ok')
    model.to('cuda')
    torch.cuda.synchronize()
    emit('onload_ok')
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(input_ids=torch.tensor([[1, 2, 3, 4]], device='cuda'), use_cache=False).logits
        assert torch.isfinite(logits).all()
    emit('forward_ok', logits_shape=list(logits.shape),
         peak_allocated_bytes=torch.cuda.max_memory_allocated())
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
