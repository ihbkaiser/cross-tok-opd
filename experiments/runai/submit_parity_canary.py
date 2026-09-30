"""Submit one capture-enabled soft canary to the existing owner queue."""
import json
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/workspace/storage-shared/nlp/tungks/borrow8-8MgodXcM')


def main():
    assert socket.gethostname() == 'tungks-0-1', 'Owner host only'
    owner = BASE / 'owner-tungks-0-1'
    config = json.loads((owner / 'campaign.json').read_text())
    sys.path.insert(0, config['manager'])
    from job_manager.store import connect, rows, submit
    from job_manager.__main__ import snapshot
    db = connect(Path(config['state']))
    try:
        status = snapshot(db)
        assert status['manager']['running'] and status['manager']['tick_age_seconds'] < 30
        assert not status['paused'] and not status['quarantined']
        numeric = next(j for j in rows(db) if j['id'] == 'numeric')
        assert numeric['status'] == 'completed' and numeric['returncode'] == 0, 'Owner numeric must pass'
        original = json.loads((BASE / 'work/jobs.json').read_text())
        spec = next(j for j in original if j['id'] == 'soft-canary')
        spec.update(id='soft-parity-capture', project='simct-owner-parity',
                    cwd=str(ROOT), gpus=numeric['spec']['gpus'], dependencies=['numeric'],
                    timeout_seconds=86400)
        spec.pop('deadline', None)
        spec['argv'] = ['bash', str(ROOT / 'experiments/runai/run_single_gpu.sh'), '1', 'soft', '5']
        spec['env'].pop('MP_TRAIN_STOP_AT', None)
        spec['env'].pop('MP_CHECKPOINT_RESERVE_SECONDS', None)
        spec['env'].update(MP_RUN_ROOT=str(owner/'soft-parity-canary'),
                           MP_PARITY_CAPTURE_DIR=str(owner/'parity-captures'),
                           MP_ENERGY_CHECKPOINT=str(owner/'energy-select-4.pt'))
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
        spec['metadata'] = dict(source_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), scope='diagnostic only; parity gate unchanged')
        submit(db, [spec])
        (owner/'parity-job.json').write_text(json.dumps(spec, indent=2))
        print(json.dumps(snapshot(db), indent=2))
    finally:
        db.close()


if __name__ == '__main__':
    main()
