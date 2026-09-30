"""Bounded setup phases for independent locked CUDA12.8 native environments."""
import argparse
from pathlib import Path
import sys
import subprocess

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'xtoken/scripts'))
from run_logged import run_logged, workload_environment

parser = argparse.ArgumentParser()
parser.add_argument('target', choices=['xtoken','simct'])
parser.add_argument('phase', choices=['lock','sync','check','canary'])
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--sync-timeout-seconds', type=int, default=1800,
                    help='Explicitly authorized install deadline; sync only (default: 1800)')
parser.add_argument('--sync-http-timeout-seconds', type=int, default=30,
                    help='uv HTTP timeout for an authorized sync attempt (default: 30)')
parser.add_argument('--pypi-mirror', action='store_true',
                    help='Authorized transport-only TUNA fallback for X-Token cuBLAS/cuDNN')
args = parser.parse_args()
if args.pypi_mirror and (args.phase != 'sync' or args.target != 'xtoken'):
    parser.error('--pypi-mirror applies only to xtoken sync')
if args.sync_timeout_seconds <= 0:
    parser.error('sync timeout must be positive')
if args.phase != 'sync' and args.sync_timeout_seconds != 1800:
    parser.error('--sync-timeout-seconds applies only to sync')
if args.sync_http_timeout_seconds <= 0:
    parser.error('sync HTTP timeout must be positive')
if args.phase != 'sync' and args.sync_http_timeout_seconds != 30:
    parser.error('--sync-http-timeout-seconds applies only to sync')
root = args.root.resolve()
if not args.root.is_absolute() or root in (Path('/'), Path.home()):
    parser.error('Choose a dedicated absolute runtime root')
root.mkdir(parents=True, exist_ok=True)
version = subprocess.run(['uv', '--version'], check=True, capture_output=True, text=True).stdout.split()
if version[1] != '0.12.7':
    parser.error('workload uv must be pinned to 0.12.7')
env=workload_environment()
for key in list(env):
    if key.startswith('UV_') or key in ('VIRTUAL_ENV','CONDA_PREFIX','PYTHONPATH','PYTHONHOME'):
        env.pop(key)
env.update(UV_CACHE_DIR=str(root/'uv-cache'), UV_NO_CACHE='false', UV_LINK_MODE='hardlink',
           UV_PYTHON_INSTALL_DIR=str(root/'python'), UV_PROJECT_ENVIRONMENT=str(root/args.target/'venv'),
           HF_HOME=str(root/'hf'), WANDB_MODE='disabled', TORCH_CUDA_ARCH_LIST='8.6',
           MAX_JOBS='4', CMAKE_BUILD_PARALLEL_LEVEL='4')
if args.phase=='lock':
    cmd=['uv','lock']
elif args.phase=='sync':
    # Explicit non-secret env assignment is retained in command/result evidence.
    cmd=['env', f'UV_HTTP_TIMEOUT={args.sync_http_timeout_seconds}', 'uv','sync','--locked']
    if args.pypi_mirror:
        cmd=['env', f'UV_HTTP_TIMEOUT={args.sync_http_timeout_seconds}', sys.executable,
             str(HERE/'mirror_sync.py'), '--root', str(root)]
elif args.phase=='check':
    cmd=['uv','pip','check','--python',str(root/args.target/'venv/bin/python')]
else:
    if args.target == 'xtoken':
        source = root/'NeMo-RL'
        if not (source/'nemo_rl/algorithms/xtoken_off_policy_distillation.py').is_file():
            parser.error('pinned NeMo-RL source must be prepared first')
        env['PYTHONPATH'] = str(source)
    cmd=['uv','run','--locked','--no-sync','python','-m','torch.distributed.run',
         '--standalone','--nproc-per-node=2',str(HERE/'cuda_canary.py'),args.target]
rc,_=run_logged(cmd,cwd=HERE/args.target,root=root/'evidence',
                name=args.target+'-'+args.phase,
                timeout=args.sync_timeout_seconds if args.phase=='sync' else 600,env=env)
sys.exit(rc)
