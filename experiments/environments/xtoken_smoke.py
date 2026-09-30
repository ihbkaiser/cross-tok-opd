"""Explicit single phases of the pinned NVIDIA functional smoke on two GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'xtoken/scripts'))
from run_logged import run_logged, workload_environment
from download_models import MODELS

parser = argparse.ArgumentParser()
parser.add_argument('phase', choices=['data','projection','config','train'])
parser.add_argument('--root', required=True, type=Path)
parser.add_argument('--train-attempt', type=int, default=1)
parser.add_argument('--retry-reason')
parser.add_argument('--nccl-cumem-host-fallback', action='store_true',
                    help='Use legacy NCCL host memory on a diagnosed cuMem IPC-incompatible host')
parser.add_argument('--cuda-launch-blocking', action='store_true',
                    help='Synchronous CUDA diagnostics only; not a performance measurement')
args = parser.parse_args()
root = args.root.resolve()
if not args.root.is_absolute() or root in (Path('/'), Path.home()):
    parser.error('dedicated absolute runtime root required')
if shutil.disk_usage(root).free < 2 * 1024**3:
    parser.error('less than 2 GiB free; no workload started')
source = root/'NeMo-RL'
assert hashlib.sha256((source/'uv.lock').read_bytes()).hexdigest() == '95f63521d28a2a4104ff372c5985fe63826ab27d6901b78bada1ab1a89a81bf7'
models = json.loads((root/'models-evidence/models.json').read_text())
assert models['complete']
assert [(m['role'], m['repo'], m['revision']) for m in models['models']] == MODELS
assert all(m['weights_verified'] for m in models['models'])
student, teacher = [m['snapshot'] for m in models['models']]
smoke = root/'smoke-2gpu-r1'
smoke.mkdir(exist_ok=True)
env = workload_environment()
for key in list(env):
    if key.startswith('UV_') or key in ('VIRTUAL_ENV','CONDA_PREFIX','PYTHONPATH','PYTHONHOME'):
        env.pop(key)
env.update(UV_CACHE_DIR=str(root/'uv-cache'), UV_NO_CACHE='false', UV_LINK_MODE='hardlink',
           UV_PYTHON_INSTALL_DIR=str(root/'python'), UV_PROJECT_ENVIRONMENT=str(root/'xtoken/venv'),
           HF_HOME=str(root/'hf'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           WANDB_MODE='disabled', PYTHONPATH=str(source), OMP_NUM_THREADS='4',
           CUDA_VISIBLE_DEVICES='0,1', NEMO_RL_VENV_DIR=str(root/'worker-venvs'),
           NEMO_RL_PY_EXECUTABLES_SYSTEM='1', RAY_USAGE_STATS_ENABLED='0',
           TORCH_CUDA_ARCH_LIST='8.6', MAX_JOBS='4', CMAKE_BUILD_PARALLEL_LEVEL='4')
if args.nccl_cumem_host_fallback:
    env['NCCL_CUMEM_HOST_ENABLE'] = '0'
if args.cuda_launch_blocking:
    env['CUDA_LAUNCH_BLOCKING'] = '1'
# Operational settings are explicit evidence, not scientific hyperparameters.
cuda_environment = {key: value for key, value in env.items()
                    if key.startswith('NCCL_') or key in
                    ('CUDA_LAUNCH_BLOCKING', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF')}
venv_alias = root/'worker-venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2'
venv_alias.parent.mkdir(exist_ok=True)
if venv_alias.is_symlink():
    assert venv_alias.resolve() == root/'xtoken/venv'
elif venv_alias.exists():
    raise SystemExit('Unexpected worker environment; refusing replacement')
else:
    venv_alias.symlink_to(root/'xtoken/venv', target_is_directory=True)
base = ['uv','run','--project',str(HERE/'xtoken'),'--locked','--no-sync','python']
projection = smoke/'projection/xtoken_l1_smoke_special.pt'
if args.phase == 'data':
    cmd = base + [str(HERE/'freeze_smoke_data.py'), str(smoke/'data')]
    timeout = 120
elif args.phase == 'projection':
    if (smoke/'projection').exists():
        raise SystemExit('Projection attempt already exists; no implicit retry')
    cmd = base + ['-m','tools.x_token.minimal_projection_via_multitoken',
                  '--student-model',student,'--teacher-model',teacher,'--top-k','4',
                  '--enable-special-token-mapping','--enable-exact-match',
                  '--disable-reverse-pass','--disable-scale-trick',
                  '--output-filename','xtoken_l1_smoke','--output-dir',str(smoke/'projection')]
    timeout = 900
else:
    assert projection.is_file()
    assert (smoke/'data/manifest.json').is_file()
    if args.phase == 'train':
        assert (smoke/'config-valid.json').is_file(), 'run config validation first'
        if args.train_attempt < 1 or (args.train_attempt > 1 and not args.retry_reason):
            parser.error('numbered retries require an explicit diagnosed reason')
    cmd = base + [str(HERE/'validate_config.py') if args.phase == 'config' else str(source/'examples/run_xtoken_off_policy_distillation.py'),
        f'policy.model_name={student}', f'policy.tokenizer.name={student}',
        f'teachers.0.model_name={teacher}', f'teachers.0.tokenizer.name={teacher}',
        'cluster.gpus_per_node=2', 'policy.train_global_batch_size=8',
        'policy.max_total_sequence_length=256', 'teachers.0.train_global_batch_size=8',
        'teachers.0.max_total_sequence_length=256', 'distillation.num_prompts_per_step=8',
        'distillation.max_num_steps=3', f'teachers.0.projection_matrix_path={projection}',
        'data.train.characters_per_sample=256', 'data.max_input_seq_length=256',
        f'data.train.data_files={smoke}/data/formal-logic-32.parquet',
        'logger.tensorboard_enabled=true', f'logger.log_dir={smoke}/logs',
        '+policy.dtensor_cfg.automodel_kwargs.force_hf=true',
        '+policy.dtensor_cfg.automodel_kwargs.attn_implementation=sdpa',
        '+teachers.0.dtensor_cfg.automodel_kwargs.force_hf=true',
        '+teachers.0.dtensor_cfg.automodel_kwargs.attn_implementation=sdpa']
    timeout = 1200
    config_digest = hashlib.sha256(json.dumps(dict(
        overrides=cmd[len(base)+1:], cwd=str(source),
        lock_sha256=hashlib.sha256((HERE/'xtoken/uv.lock').read_bytes()).hexdigest(),
        projection_sha256=hashlib.sha256(projection.read_bytes()).hexdigest(),
        data_manifest_sha256=hashlib.sha256((smoke/'data/manifest.json').read_bytes()).hexdigest(),
        cuda_environment=cuda_environment,
    ), sort_keys=True).encode()).hexdigest()
    if args.phase == 'train':
        validated = json.loads((smoke/'config-valid.json').read_text())
        assert validated.get('config_digest') == config_digest, 'stale config validation; validate again'
        marker = smoke/('train-attempt.json' if args.train_attempt == 1 else f'train-attempt-{args.train_attempt}.json')
        with marker.open('x') as f:
            json.dump(dict(attempt=args.train_attempt, retry_reason=args.retry_reason,
                           config_digest=config_digest,
                           cuda_environment=cuda_environment,
                           source='13a10647ebbf0f940d2b06ea41800b3f2fb46099',
                           scope='off-policy engineering smoke, not OPD or paper reproduction',
                           source_modified=False), f, indent=2)
rc, out = run_logged(cmd, cwd=source, root=root/'evidence',
                     name='xtoken-'+args.phase, timeout=timeout, env=env)
print(json.dumps(dict(phase=args.phase, exit_code=rc, evidence=str(out))), flush=True)
if args.phase == 'config' and rc == 0:
    (smoke/'config-valid.json').write_text(json.dumps(dict(evidence=str(out), valid=True, config_digest=config_digest))+'\n')
sys.exit(rc)
