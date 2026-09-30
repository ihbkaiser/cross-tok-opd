"""Bounded B200 autograd diagnostic; no model downloads or company data."""
import json
import os
import subprocess
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/opt/overlay')
IMAGE = 'docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f'
app = modal.App('simct-full-meta-autograd-debug')
image = (modal.Image.from_registry(IMAGE).entrypoint([])
    .add_local_dir(str(ROOT/'kdflow'), '/opt/overlay/kdflow', ignore=['**/__pycache__/**','**/*.pyc'])
    .add_local_dir(str(ROOT/'tests'), '/opt/overlay/tests', ignore=['**/__pycache__/**','**/*.pyc'])
    .add_local_dir(str(ROOT/'experiments/runai'), '/opt/overlay/experiments/runai', ignore=['**/__pycache__/**','**/*.pyc'])
    .add_local_dir(str(ROOT/'experiments/mp_opd'), '/opt/overlay/experiments/mp_opd', ignore=['**/__pycache__/**','**/*.pyc'])
    .add_local_dir(str(ROOT/'scripts'), '/opt/overlay/scripts', ignore=['**/__pycache__/**','**/*.pyc'])
    .add_local_file(str(ROOT/'experiments/modal/vendor/xtoken_upstream_token_aligner.py'), '/opt/overlay/xtoken_upstream_token_aligner.py')
    .add_local_file(str(ROOT/'experiments/modal/mp_preprocessing_benchmark.py'), '/opt/overlay/preprocessing.py')
    .add_local_file(str(ROOT/'experiments/modal/full_meta_debug_worker.py'), '/opt/overlay/probe.py'))
if os.environ.get('META_DEBUG_SOURCE_BUNDLE'):
    image = image.add_local_file(os.environ['META_DEBUG_SOURCE_BUNDLE'], '/tmp/meta-source.bundle')

@app.function(image=image, cpu=2, memory=8192, timeout=180, retries=0, max_containers=1)
def regression():
    work = '/opt/overlay'
    if Path('/tmp/meta-source.bundle').exists():
        subprocess.run(['git','clone','--branch','vdt/ops/b200-portable','/tmp/meta-source.bundle','/tmp/meta-source'],check=True)
        work = '/tmp/meta-source'
    env = dict(os.environ, PYTHONPATH=work+'/experiments/modal/vendor:'+work, PYTHONDONTWRITEBYTECODE='1')
    p = subprocess.run(['/opt/venvs/simct-b200/bin/python', '-m', 'pytest',
        'tests/mp_opd', 'tests/test_training_checkpoint.py', 'tests/test_meta_data.py',
        'tests/test_trajectory.py', 'tests/test_exact_trajectory_integration.py', '-q', '-p', 'no:cacheprovider'],
        cwd=work, env=env, capture_output=True, text=True, timeout=150)
    print(p.stdout, flush=True); print(p.stderr, flush=True)
    return {'returncode':p.returncode, 'stdout':p.stdout, 'stderr':p.stderr}

@app.function(image=image, gpu=os.environ.get('META_DEBUG_GPU', 'B200'), cpu=2, memory=8192, timeout=180, retries=0, max_containers=1)
def probe(preprocessing: bool = False):
    import os
    env = dict(os.environ, PYTHONPATH='/opt/overlay', PYTHONUNBUFFERED='1')
    p = subprocess.run(['/opt/venvs/simct-b200/bin/python',
        '/opt/overlay/preprocessing.py' if preprocessing else '/opt/overlay/probe.py'],
        env=env, capture_output=True, text=True, timeout=150)
    print(p.stdout, flush=True)
    print(p.stderr, flush=True)
    return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}

@app.local_entrypoint()
def main(run_id: str, regression_only: bool = False, preprocessing: bool = False):
    result = regression.remote() if regression_only else probe.remote(preprocessing)
    dest = ROOT/'remote_artifacts'/run_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'result.json').write_text(json.dumps(result, indent=2))
    if result['returncode']:
        raise RuntimeError('Diagnostic failed; inspect result.json')
