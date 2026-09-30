"""Single-attempt 4B / 2048 / batch64 workload with upstream full projection."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'xtoken/scripts'))
from run_logged import run_logged, workload_environment

SOURCE = '13a10647ebbf0f940d2b06ea41800b3f2fb46099'
UPSTREAM_LOCK = '95f63521d28a2a4104ff372c5985fe63826ab27d6901b78bada1ab1a89a81bf7'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overrides(target, student, teacher, steps):
    assert steps in (3, 10)
    return [f'policy.model_name={student}', f'policy.tokenizer.name={student}',
        f'teachers.0.model_name={teacher}', f'teachers.0.tokenizer.name={teacher}',
        'cluster.gpus_per_node=2', 'policy.train_global_batch_size=64',
        'policy.max_total_sequence_length=2048', 'teachers.0.train_global_batch_size=64',
        'teachers.0.max_total_sequence_length=2048', 'distillation.num_prompts_per_step=64',
        f'distillation.max_num_steps={steps}',
        f'teachers.0.projection_matrix_path={target}/projection-full/runtime_top4.pt',
        'data.train.characters_per_sample=16384', 'data.max_input_seq_length=2048',
        f'data.train.data_files={target}/data/formal-logic-prefix.parquet',
        'logger.tensorboard_enabled=true', f'logger.log_dir={target}/logs',
        '+policy.dtensor_cfg.automodel_kwargs.force_hf=true',
        '+policy.dtensor_cfg.automodel_kwargs.attn_implementation=sdpa',
        '+teachers.0.dtensor_cfg.automodel_kwargs.force_hf=true',
        '+teachers.0.dtensor_cfg.automodel_kwargs.attn_implementation=sdpa']


def projection_commands(base, target, student, teacher):
    out = target/'projection-full'
    prep = out/'prep32_special.pt'
    remapped = out/'prep32_special_exact_map_remapped.pt'
    return [
        base + ['-m', 'tools.x_token.minimal_projection_via_multitoken',
                '--student-model', student, '--teacher-model', teacher, '--top-k', '32',
                '--enable-reverse-pass', '--enable-scale-trick', '--enable-special-token-mapping',
                '--output-dir', str(out), '--output-filename', 'prep32'],
        base + ['-m', 'tools.x_token.reapply_exact_map', '--student-model', student,
                '--teacher-model', teacher, '--initial-projection-path', str(prep)],
        base + ['-m', 'tools.x_token.sort_and_cut_projection_matrix',
                '--initial-projection-path', str(remapped), '--top_k', '4',
                '--output_path', str(out/'runtime_top4.pt')],
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['models', 'data', 'projection', 'config', 'train'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--steps', type=int, choices=(3, 10), default=3,
                        help='Explicitly authorized optimizer-update count')
    parser.add_argument('--run-name')
    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = ('target-4b-2048-b64-full-r1' if args.steps == 3 else
                         'target-4b-2048-b64-full-10steps-r1')
    root = args.root.resolve()
    assert args.root.is_absolute() and root not in (Path('/'), Path.home())
    assert args.run_name.replace('-', '').isalnum()
    assert shutil.disk_usage(root).free > 3 * 1024**3
    source, target = root/'NeMo-RL', root/args.run_name
    assert digest(source/'uv.lock') == UPSTREAM_LOCK
    assert json.loads((root/'NeMo-RL.provenance.json').read_text())['source_modified'] is False
    target.mkdir(exist_ok=True)
    protocol = dict(optimizer_updates=args.steps, global_batch=64, sequence_length=2048)
    protocol_path = target/'requested-protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, 'Run identity/protocol mismatch'
    else:
        with protocol_path.open('x') as stream:
            json.dump(protocol, stream, indent=2)
    env = workload_environment()
    ephemeral_uv_root = env.pop('VDT_EPHEMERAL_UV_ROOT', None)
    uv_root = Path(ephemeral_uv_root).resolve() if ephemeral_uv_root else root
    assert uv_root.is_absolute() and uv_root not in (Path('/'), Path.home())
    for key in list(env):
        if key.startswith(('UV_', 'NCCL_')) or key in (
                'VIRTUAL_ENV', 'CONDA_PREFIX', 'PYTHONPATH', 'PYTHONHOME',
                'CUDA_LAUNCH_BLOCKING', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF'):
            env.pop(key)
    env.update(UV_CACHE_DIR=str(uv_root/'uv-cache'), UV_NO_CACHE='false',
        UV_LINK_MODE=('copy' if ephemeral_uv_root else 'hardlink'),
        UV_PYTHON_INSTALL_DIR=str(uv_root/'python'), UV_PROJECT_ENVIRONMENT=str(root/'xtoken/venv'),
        HF_HOME=str(root/'hf'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        WANDB_MODE='disabled', PYTHONPATH=str(source), OMP_NUM_THREADS='4',
        CUDA_VISIBLE_DEVICES='0,1', NEMO_RL_VENV_DIR=str(root/'worker-venvs'),
        NEMO_RL_PY_EXECUTABLES_SYSTEM='1', RAY_USAGE_STATS_ENABLED='0',
        NCCL_CUMEM_HOST_ENABLE='0', TORCH_CUDA_ARCH_LIST='8.6')
    alias = root/'worker-venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2'
    alias.parent.mkdir(exist_ok=True)
    if alias.is_symlink():
        assert alias.resolve() == (root/'xtoken/venv').resolve()
    elif alias.exists():
        raise RuntimeError('Unexpected worker environment')
    else:
        alias.symlink_to(root/'xtoken/venv', target_is_directory=True)
    base = ['uv', 'run', '--project', str(HERE/'xtoken'), '--locked', '--no-sync', 'python']
    commands = []
    config_digest = None
    if args.phase in ('models', 'data'):
        commands = [base + [str(HERE/'prepare_xtoken_target.py'), args.phase,
                           '--root', str(root), '--target', str(target), '--steps', str(args.steps)]]
    else:
        models = json.loads((target/'models.json').read_text())
        assert models['complete'] and all(m['weights_verified'] for m in models['models'])
        student, teacher = [m['snapshot'] for m in models['models']]
        if args.phase == 'projection':
            commands = projection_commands(base, target, student, teacher)
            commands.append(base + [str(HERE/'prepare_xtoken_target.py'), 'projection-audit',
                                   '--root', str(root), '--target', str(target)])
        else:
            assert json.loads((target/'projection-full/manifest.json').read_text())['valid']
            settings = overrides(target, student, teacher, args.steps)
            config_digest = hashlib.sha256(json.dumps(dict(overrides=settings, source=SOURCE,
                lock=digest(HERE/'xtoken/uv.lock'), models=digest(target/'models.json'),
                projection=digest(target/'projection-full/runtime_top4.pt'),
                data=digest(target/'data/manifest.json'), nccl_cumem_host_enable='0'),
                sort_keys=True).encode()).hexdigest()
            if args.phase == 'train':
                assert json.loads((target/'config-valid.json').read_text())['config_digest'] == config_digest
            script = (HERE/'validate_xtoken_target.py' if args.phase == 'config'
                      else source/'examples/run_xtoken_off_policy_distillation.py')
            validator_args = ['--expected-steps', str(args.steps)] if args.phase == 'config' else []
            commands = [base + [str(script), *validator_args, *settings]]
    # Each phase claims its marker before execution. Never replace previous evidence.
    with (target/f'{args.phase}-attempt-1.json').open('x') as stream:
        json.dump(dict(source=SOURCE, commands=commands, config_digest=config_digest,
            operational_environment={'NCCL_CUMEM_HOST_ENABLE': '0'},
            optimizer_updates=args.steps,
            scope=f'{args.steps}-update off-policy workload test, full projection; no OPD/quality claim'), stream, indent=2)
    for i, command in enumerate(commands):
        rc, out = run_logged(command, cwd=source, root=root/'evidence',
            name=f'xtoken-target-{args.phase}-{i+1}', timeout=1800, env=env)
        if rc:
            print(json.dumps(dict(phase=args.phase, failed_step=i+1, evidence=str(out), exit_code=rc)))
            return rc
    if args.phase == 'config':
        (target/'config-valid.json').write_text(json.dumps(dict(valid=True, config_digest=config_digest,
                                                               evidence=str(out))) + '\n')
    print(json.dumps(dict(phase=args.phase, success=True, evidence=str(out))), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
