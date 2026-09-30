"""Managed continuation after verified setup and diagnosed Viewer HTTP502.

No installer, no model transfer, no scientific changes, no training replay.
Every phase has a durable log and a hard deadline. User authorized operational
retries; only transient data HTTP failures are automatically retried here.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_target_after_setup import execute, resource_gate, expected_metadata_only
from resumable_target_data import json_atomic
sys.path.insert(0, str(HERE.parent/'xtoken/scripts'))
from run_logged import run_logged, workload_environment

RUN = 'target-4b-2048-b64-full-10steps-r4'
OLD = 'target-4b-2048-b64-full-10steps-r3'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_previous(record):
    assert record['status'] == 'stopped' and record['exit_code'] == 1
    assert record['optimizer_updates'] == 10 and record['training_phase_invoked'] is False
    assert [r['phase'] for r in record['phases']] == ['check', 'canary', 'models', 'data']
    assert record['phases'][0]['expected_metadata_exclusions_only'] is True
    assert [r['exit_code'] for r in record['phases']] == [1, 0, 0, 1]


def preflight(root):
    previous = root/('sequence-' + OLD)
    record = json.loads((previous/'result.json').read_text())
    validate_previous(record)
    assert json.loads(Path(record['inputs'][0]).read_text())['exit_code'] == 0
    data_evidence = Path(record['phases'][-1]['evidence'][0])
    assert data_evidence.resolve().is_relative_to(root/'evidence')
    assert 'HTTP Error 502: Bad Gateway' in (data_evidence/'stdout.log').read_text()
    # All prior NeMo/config/native lock/target sources remain exact, not just HEAD.
    fingerprints = json.loads((previous/'source-fingerprints.json').read_text())
    for name, checksum in fingerprints.items():
        path = Path(name)
        assert path.is_relative_to(root/'NeMo-RL') or path.is_relative_to(HERE.parent)
        assert sha(path) == checksum, f'Prior source drift: {path.name}'
    models = json.loads((root/OLD/'models.json').read_text())
    assert models['complete'] and all(m['weights_verified'] for m in models['models'])
    for model in models['models']:
        for item in model['files']:
            path = Path(model['snapshot'])/item['name']
            assert path.resolve().is_relative_to(root/'hf')
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    resource_gate()
    return fingerprints


def run(root):
    with (root/'.xtoken-managed.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = root/('sequence-' + RUN)
        out.mkdir(mode=0o700, exist_ok=False)
        state = dict(status='preflight', optimizer_updates=10, training_phase_invoked=False,
            phases=[], pid=os.getpid(), parent_pid=os.getppid(),
            started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            continuation_of=str(root/('sequence-' + OLD)),
            reason='User authorized automatic operational recovery after Viewer HTTP502',
            retry_policy='data: 8 attempts per page, 1500s deadline; no silent replay')

        def save():
            state['updated_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
            json_atomic(out/'result.json', state)

        save()
        try:
            fingerprints = preflight(root)
            fingerprints.update({str(p): sha(p) for p in HERE.glob('*.py')})
            json_atomic(out/'source-fingerprints.json', fingerprints)
            target = root/RUN
            target.mkdir(exist_ok=False)
            shutil.copyfile(root/OLD/'models.json', target/'models.json')
            json_atomic(target/'reused-gates.json', dict(predecessor=str(root/('sequence-' + OLD)),
                install_repeated=False, weights_rehashed=True, model_transfer_repeated=False,
                legacy_pages_reused=False, reason='Legacy pages have no revision sidecars'))
            for phase in ('check', 'canary', 'data', 'projection', 'config', 'train'):
                assert all(sha(Path(p)) == h for p, h in fingerprints.items()), 'Source drift during run'
                if phase in ('canary', 'projection', 'train'):
                    resource_gate(projection=phase == 'projection')
                state.update(status='running_phase', current_phase=phase)
                if phase == 'train':
                    state['training_phase_invoked'] = True
                save()
                log = out/(phase + '.log')
                if phase == 'data':
                    env = workload_environment()
                    env.update(HF_HUB_DISABLE_IMPLICIT_TOKEN='1', PYTHONUNBUFFERED='1')
                    rc, evidence = run_logged([str(root/'xtoken/venv/bin/python'),
                        str(HERE/'resumable_target_data.py'), '--target', str(target)],
                        cwd=HERE, root=root/'evidence', name='xtoken-data-recovery-r4', timeout=1800, env=env)
                    evidence = [str(evidence)]
                else:
                    script = HERE/('run_phase.py' if phase in ('check', 'canary') else 'xtoken_target.py')
                    args = (['xtoken', phase, '--root', str(root)] if phase in ('check', 'canary') else
                            [phase, '--root', str(root), '--steps', '10', '--run-name', RUN])
                    rc = execute(script, args, log)
                    evidence = re.findall(r'^EVIDENCE_DIR=(.+)$', log.read_text(), re.MULTILINE)
                assert evidence and all(Path(p).resolve().is_relative_to(root/'evidence') for p in evidence)
                results = [json.loads((Path(p)/'result.json').read_text()) for p in evidence]
                accepted = phase == 'check' and rc == 1 and len(results) == 1 and expected_metadata_only(
                    (Path(evidence[0])/'stdout.log').read_text())
                state['phases'].append(dict(phase=phase, exit_code=rc, evidence=evidence,
                                           expected_metadata_exclusions_only=accepted))
                save()
                assert accepted or (rc == 0 and all(r['exit_code'] == 0 for r in results)), \
                    f'{phase} failed; diagnostics required before another distinct attempt'
            state.update(status='training_process_complete_pending_metric_review', exit_code=0)
        except Exception as error:
            state.update(status='stopped', exit_code=1, error_type=type(error).__name__, error=str(error))
        except KeyboardInterrupt:
            state.update(status='interrupted', exit_code=130)
        finally:
            state['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
            save()
        return state['exit_code']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    assert args.root.is_absolute() and root not in (Path('/'), Path.home())
    os.umask(0o077)
    os.environ.clear()
    os.environ.update(PATH='/usr/local/bin:/usr/bin:/bin', HOME='/root', LANG='C.UTF-8', PYTHONUNBUFFERED='1')
    if args.preflight_only:
        preflight(root)
        print(json.dumps(dict(preflight='passed', workload_started=False)))
    else:
        sys.exit(run(root))
