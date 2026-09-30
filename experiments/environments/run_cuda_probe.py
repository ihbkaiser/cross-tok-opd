"""One explicit bounded probe per invocation; never retries or syncs dependencies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'xtoken/scripts'))
from run_logged import run_logged, workload_environment

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['hf', 'nemo', 'ray'])
parser.add_argument('--probe-id')
parser.add_argument('--root', required=True, type=Path)
args = parser.parse_args()
root = args.root.resolve()
assert args.root.is_absolute() and root not in (Path('/'), Path.home())
assert shutil.disk_usage(root).free >= 2 * 1024**3, 'less than 2 GiB free'
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
probe_id = args.probe_id or args.mode
assert probe_id.replace('-', '').isalnum()
marker = root/'smoke-2gpu-r1'/f'cuda-probe-{probe_id}.json'
with marker.open('x') as f:
    json.dump(dict(mode=args.mode, scope='setup-only; no training',
                   launch_blocking=1,
                   nccl_environment={k: v for k, v in os.environ.items() if k.startswith('NCCL_')},
                   script_sha256=hashlib.sha256((HERE/'xtoken_cuda_probe.py').read_bytes()).hexdigest()), f, indent=2)
env = workload_environment()
for key in list(env):
    if key.startswith('UV_') or key in ('VIRTUAL_ENV', 'CONDA_PREFIX', 'PYTHONPATH', 'PYTHONHOME'):
        env.pop(key)
env.update(UV_CACHE_DIR=str(root/'uv-cache'), UV_PYTHON_INSTALL_DIR=str(root/'python'),
           UV_PROJECT_ENVIRONMENT=str(root/'xtoken/venv'), HF_HOME=str(root/'hf'),
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
           PYTHONPATH=str(HERE)+':'+str(root/'NeMo-RL'), OMP_NUM_THREADS='4',
           CUDA_VISIBLE_DEVICES='0,1', CUDA_LAUNCH_BLOCKING='1',
           PYTHONUNBUFFERED='1', RAY_USAGE_STATS_ENABLED='0',
           NEMO_RL_VENV_DIR=str(root/'worker-venvs'), NEMO_RL_PY_EXECUTABLES_SYSTEM='1')
if args.mode == 'ray':
    alias = root/'worker-venvs/cuda_probe_worker.BufferProbeWorker'
    if not alias.exists():
        alias.symlink_to(root/'xtoken/venv', target_is_directory=True)
    assert alias.resolve() == root/'xtoken/venv'
cmd = ['uv', 'run', '--project', str(HERE/'xtoken'), '--locked', '--no-sync', 'python']
if args.mode == 'nemo':
    cmd += ['-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2']
cmd += [str(HERE/'xtoken_cuda_probe.py'), args.mode, '--root', str(root)]
rc, out = run_logged(cmd, cwd=root/'NeMo-RL', root=root/'evidence',
                     name='cuda-probe-'+probe_id, timeout=240, env=env)
print(json.dumps(dict(exit_code=rc, evidence=str(out))))
sys.exit(rc)
