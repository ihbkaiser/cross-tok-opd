"""Submit SimCT43 or a bounded soft diagnostic without changing running jobs."""
import argparse
import copy
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/workspace/storage-shared/nlp/tungks/borrow8-8MgodXcM')
BORROWED = 'tungdd11-sparse-vllm-core-0-0'
OWNER = 'tungks-0-1'


def specification(template, kind, gpu, output):
    spec = copy.deepcopy(template)
    spec.update(id='simct43-full312' if kind == 'simct' else 'soft-parity-50',
                project='simct-followup', cwd=str(ROOT), gpus=[gpu],
                timeout_seconds=86400, dependencies=[], dependency_policy='success')
    spec.pop('deadline', None)
    spec.pop('require_full_budget', None)
    env = spec['env']
    for key in ('MP_TRAIN_STOP_AT', 'MP_CHECKPOINT_RESERVE_SECONDS', 'MP_ENERGY_CHECKPOINT',
                'MP_PARITY_CAPTURE_DIR', 'MP_PREFLIGHT_ONLY'):
        env.pop(key, None)
    env.update(MP_RUN_ROOT=str(output), MP_SEED='43' if kind == 'simct' else '42',
               MP_ALGORITHM='span_ctkd' if kind == 'simct' else 'mp_opd',
               MP_ATTN_IMPLEMENTATION='sdpa' if kind == 'simct' else 'eager')
    if kind == 'soft':
        env.update(MP_ENERGY_CHECKPOINT=str(BASE/'owner-tungks-0-1/energy-select-4.pt'),
                   MP_PARITY_CAPTURE_DIR=str(output/'captures'))
    spec['argv'] = ['bash', str(ROOT/'experiments/runai/run_single_gpu.sh'), '1',
                    'atomic' if kind == 'simct' else 'soft', '0' if kind == 'simct' else '50']
    spec['metadata'] = dict(scope='full312 SimCT seed43; historical sdpa backend' if kind == 'simct'
                            else '50-update diagnostic; unchanged parity gate; no efficacy claim',
                            allocation_deadline=None, watchdog_seconds=86400)
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind', choices=['simct', 'soft'])
    args = p.parse_args()
    expected = BORROWED if args.kind == 'simct' else OWNER
    if socket.gethostname() != expected:
        raise ValueError('Run on '+expected)
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'], cwd=ROOT, text=True).strip():
        raise ValueError('Dirty source')
    manager = BASE/'job-manager'
    sys.path.insert(0, str(manager))
    from job_manager.store import connect, rows, submit
    from job_manager.__main__ import snapshot
    work = BASE/('simct43-followup' if args.kind == 'simct' else 'owner-tungks-0-1/soft50-followup')
    receipt = work/'submission.json'
    if receipt.exists():
        print(receipt.read_text()); return
    template = next(j for j in json.loads((BASE/'work/jobs.json').read_text()) if j['id']=='soft-canary')
    uuid = subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'], text=True).strip()
    required = 'GPU-dd4574a2-6869-ca46-3b2d-0fc496260ad7' if args.kind=='simct' else 'GPU-c8e6cc11-aa9d-392f-df6b-1f7d550b88d2'
    if uuid != required:
        raise ValueError('GPU1 UUID changed; inspect mapping')
    spec = specification(template, args.kind, uuid, work/'runs')
    spec['metadata']['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip()
    if args.kind == 'soft':
        cfg = json.loads((BASE/'owner-tungks-0-1/campaign.json').read_text())
        state = Path(cfg['state'])
        if not Path(spec['env']['MP_ENERGY_CHECKPOINT']).is_file():
            raise ValueError('Missing qualified energy checkpoint')
    else:
        # New manager shares the same per-GPU leases with the old manager.
        # It does not inherit or change the old immutable eight-hour deadline.
        state = Path(tempfile.mkdtemp(prefix='simct43-followup-', dir='/var/tmp'))
        subprocess.run([sys.executable,'-m','job_manager','--state',str(state),'init',
                        '--gpus',uuid,'--cpu-slots','8'],cwd=manager,check=True)
    db = connect(state)
    try:
        if args.kind == 'soft':
            status = snapshot(db)
            if not status['manager']['running'] or status['paused'] or status['quarantined']:
                raise ValueError('Owner manager not ready')
            prior = next(j for j in rows(db) if j['id']=='soft-parity-capture')
            if prior['status']!='completed' or prior['returncode']!=0:
                raise ValueError('Inspect previous parity canary before extending')
            spec['dependencies'] = ['soft-parity-capture']
        work.mkdir(parents=True, exist_ok=True)
        submit(db,[spec])
        receipt.write_text(json.dumps(dict(state=str(state),source=str(ROOT),spec=spec),indent=2))
        if args.kind == 'simct':
            with (work/'manager.log').open('ab') as out:
                proc = subprocess.Popen([sys.executable,'-m','job_manager','--state',str(state),'run'],
                    cwd=manager,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            time.sleep(2)
            if proc.poll() is not None:
                raise RuntimeError('Manager exited; inspect '+str(work/'manager.log'))
        print(json.dumps(snapshot(db),indent=2))
        print('SUBMISSION_RECEIPT:',receipt)
    finally:
        db.close()


if __name__ == '__main__':
    main()
