"""Bounded, sequential capacity probe on existing lhtu05 base-weight cache."""
import json
import os
from pathlib import Path
import subprocess
import time
import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/opt/overlay')
IMAGE = 'docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f'
app = modal.App('simct-full-meta-capacity')
image = (modal.Image.from_registry(IMAGE).entrypoint([])
    .add_local_dir(str(ROOT/'kdflow'), '/opt/overlay/kdflow', ignore=['**/__pycache__/**'])
    .add_local_file(str(ROOT/'experiments/modal/full_meta_capacity.py'), '/opt/overlay/capacity.py'))
assets = modal.Volume.from_name('simct-phi-gemma-assets', create_if_missing=False)

@app.function(image=image, gpu='B200', cpu=8, memory=98304, timeout=1500,
              retries=0, max_containers=1, volumes={'/assets': assets})
def sweep(micros: list[int], steps: int, batch: int, length: int, meta_micro: int,
          attention: str, offload_adam_moments: bool):
    start = time.monotonic(); results = []
    # Separate process releases all allocations after an OOM. Failures other
    # than OOM stop the sweep rather than masquerading as capacity findings.
    for micro in micros:
        remaining = 1400 - (time.monotonic()-start)
        if remaining < 60: break
        try:
            command = ['/opt/venvs/simct-b200/bin/python', '/opt/overlay/capacity.py',
                '--micro', str(micro), '--steps', str(steps), '--batch', str(batch),
                '--length', str(length), '--meta-micro', str(meta_micro),
                '--attention', attention]
            if offload_adam_moments:
                command.append('--offload-adam-moments')
            p = subprocess.run(command, capture_output=True, text=True,
                env=dict(os.environ, PYTHONUNBUFFERED='1'),
                timeout=min(remaining, 500) if steps == 1 else remaining)
        except subprocess.TimeoutExpired:
            results.append(dict(micro=micro,status='timeout')); break
        print(p.stdout, flush=True); print(p.stderr[-4000:], flush=True)
        markers = [line for line in p.stdout.splitlines() if line.startswith('CAPACITY_RESULT ')]
        record = json.loads(markers[-1].split(' ',1)[1]) if markers else dict(micro=micro,status='error')
        results.append(record)
        if p.returncode or record['status'] != 'oom': break
    return results

@app.local_entrypoint()
def main(micros: str = '16,8,4', run_id: str = 'full-meta-capacity-r2',
         steps: int = 1, batch: int = 0, length: int = 1024, meta_micro: int = 4,
         attention: str = 'eager', offload_adam_moments: bool = False):
    values = [int(x) for x in micros.split(',')]
    if any(x not in (1,2,4,8,16,32,64) for x in values):
        raise ValueError('microbatch must divide B64')
    if steps not in (1,2,10,30) or batch not in (0,64) or meta_micro not in (1,2,4,8,16):
        raise ValueError('Invalid bounded probe configuration')
    if attention not in ('eager', 'sdpa'):
        raise ValueError('Invalid attention backend')
    dest = ROOT/'remote_artifacts'/run_id
    dest.mkdir(parents=True, exist_ok=False)
    results = sweep.remote(values, steps, batch, length, meta_micro, attention,
                           offload_adam_moments)
    (dest/'results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
