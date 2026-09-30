"""Bounded isolated native setup. No source/driver/default-env modifications."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, '/workspace/simct-xtoken-harness/experiments/xtoken/scripts')
from run_logged import run_logged, workload_environment

phase = argparse.ArgumentParser()
phase.add_argument('target', choices=['automodel', 'simct', 'simct-core'])
target = phase.parse_args().target
root = Path('/workspace/xtoken-native')
repo = root/'NeMo-RL'
env = workload_environment()
for key in list(env):
    if key.startswith('UV_') or key in ('VIRTUAL_ENV', 'CONDA_PREFIX', 'PYTHONPATH', 'PYTHONHOME'):
        env.pop(key)
env.update(UV_CACHE_DIR=str(root/'uv-cache'), UV_NO_CACHE='false', UV_LINK_MODE='hardlink',
           TORCH_CUDA_ARCH_LIST='8.6', MAX_JOBS='4', CMAKE_BUILD_PARALLEL_LEVEL='4',
           HF_HOME=str(root/'hf-cache'), WANDB_MODE='disabled')
out = root/'smoke-20260903-r1'

def run(name, cmd, timeout=600, cwd=repo):
    rc, evidence = run_logged(cmd, cwd=cwd, root=out, name=name, timeout=timeout, env=env)
    if rc:
        print(json.dumps({'failed_phase': name, 'evidence': str(evidence), 'no_auto_retry': True}))
        sys.exit(rc)

if target == 'automodel':
    worker = root/'worker-venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2'
    env['UV_PROJECT_ENVIRONMENT'] = str(worker)
    run('automodel-base', ['uv', 'sync', '--locked'])
    # Dense HF + PyTorch SDPA only: omit optional vendor kernels. Exact upstream
    # package versions remain locked. Source and algorithms are unmodified.
    omitted = ['transformer-engine', 'flash-attn', 'deep-ep', 'mamba-ssm',
               'causal-conv1d', 'nv-grouped-gemm', 'tilelang']
    cmd = ['uv', 'sync', '--locked', '--extra', 'automodel']
    for package in omitted:
        cmd += ['--no-install-package', package]
    (out/'automodel-native-variant.json').write_text(json.dumps(dict(
        mode='dense-HF-PyTorch-SDPA-native', skipped_optional_packages=omitted,
        reason='Single Ampere dense models; host nvcc12.8 differs from locked torch cu130; no MoE/SSM/TE/Flash kernel build',
        worker_path=str(worker), upstream_source_unmodified=True), indent=2)+'\n')
    run('automodel-dense-install', cmd, timeout=900)
    run('automodel-worker-import', [str(worker/'bin/python'), '-c',
        'from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import DTensorPolicyWorkerV2; import torch; print("DTensorPolicyWorkerV2_IMPORT_PASS", torch.__version__, torch.cuda.is_available())'], timeout=180)
else:
    simct = Path('/workspace/simct-native')
    simct.mkdir(exist_ok=True)
    if target == 'simct':
        run('simct-venv', ['uv', 'venv', '--python', '3.12', str(simct/'.venv')], timeout=180)
    py = str(simct/'.venv/bin/python')
    package = 'sglang==0.5.9' if target == 'simct-core' else 'sglang[all]==0.5.9'
    # Versions from the SimCT CUDA12.8/torch2.9.1/sglang0.5.9 Docker recipe.
    run(target+'-resolve', ['uv', 'pip', 'install', '--python', py, '--dry-run',
                         package, 'torch==2.9.1',
                         '-e', '/workspace/simct-xtoken-harness'], timeout=300)
    run(target+'-install', ['uv', 'pip', 'install', '--python', py,
                         package, 'torch==2.9.1',
                         '-e', '/workspace/simct-xtoken-harness'], timeout=900)
