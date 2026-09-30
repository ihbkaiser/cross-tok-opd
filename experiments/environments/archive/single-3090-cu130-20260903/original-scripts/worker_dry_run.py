"""Record the exact locked upstream Automodel install plan without installing it."""
from pathlib import Path
import sys

sys.path.insert(0, '/workspace/simct-xtoken-harness/experiments/xtoken/scripts')
from run_logged import run_logged, workload_environment

root = Path('/workspace/xtoken-native')
env = workload_environment()
env.update(UV_CACHE_DIR=str(root/'uv-cache'), UV_NO_CACHE='false', UV_LINK_MODE='hardlink',
           UV_PROJECT_ENVIRONMENT=str(root/'worker-env'), TORCH_CUDA_ARCH_LIST='8.6',
           HF_HOME=str(root/'hf-cache'))
rc, _ = run_logged(['uv', 'sync', '--locked', '--extra', 'automodel', '--dry-run'],
                   cwd=root/'NeMo-RL', root=root/'smoke-20260903-r1',
                   name='automodel-locked-dry-run', timeout=300, env=env)
sys.exit(rc)
