"""One explicitly authorized recovery, run under Supervisor, never SSH lifetime.

Install is bounded to 5400 seconds with the unchanged lock and HTTP timeout 300.
Only successful setup can invoke the existing guarded ten-update sequence once.
The exclusive run directory prevents restarts from replaying expensive work.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_target_after_setup import execute, resource_gate, validate_setup_only_predecessor

NAME = 'managed-target-10steps-r3'
MIRROR_NAME = 'managed-target-10steps-mirror-r4'
SCIENTIFIC = ('xtoken_target.py', 'prepare_xtoken_target.py',
              'validate_xtoken_target.py', 'xtoken/pyproject.toml', 'xtoken/uv.lock')


def read_json(path):
    return json.loads(path.read_text())


def validate_recovery(root):
    predecessor = root/'sequence-target-4b-2048-b64-full-10steps-r2'
    old = read_json(predecessor/'result.json')
    validate_setup_only_predecessor(old)
    assert old['dependencies'] == ['failed', 'ready', 'ready']
    paths = [Path(p) for p in old['inputs']]
    assert len(paths) == 3 and all(p.resolve().is_relative_to(root) for p in paths)
    install = read_json(paths[0])
    assert install['error_type'] == 'BrokenPipeError' and install['exit_code'] == 127
    assert install['timed_out'] is False
    assert paths[1] == root/'model-transfer90-teacher-r2/result.json'
    assert paths[2] == root/'student-http-resume-r4/result.json'
    assert read_json(paths[1])['exit_code'] == 0
    student = read_json(paths[2])
    assert student['status'] == 'verified' and student['transfer_verified'] is True
    old_hashes = read_json(predecessor/'source-fingerprints.json')
    for relative in SCIENTIFIC:
        path = HERE/relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == old_hashes[str(path)]
    return predecessor, paths


def validate_mirror_recovery(root):
    predecessor, paths = validate_recovery(root)
    previous = root/NAME
    record = read_json(previous/'result.json')
    assert record['status'] == 'setup_failed' and record['exit_code'] == 124
    assert record['training_phase_invoked'] is False
    assert record['requested_optimizer_updates'] == 10
    setup = Path(record['setup_result'])
    assert setup.resolve().is_relative_to(root/'evidence')
    terminal = read_json(setup)
    assert terminal['exit_code'] == 124 and terminal['timed_out'] is True
    assert terminal['console_disconnected'] is False
    assert not (root/'sequence-target-4b-2048-b64-full-10steps-r3').exists()
    hashes = read_json(previous/'source-fingerprints.json')
    for relative in SCIENTIFIC:
        path = HERE/relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path)]
    return predecessor, paths


def run(root, mirror=False):
    assert root.is_absolute() and root not in (Path('/'), Path.home())
    root = root.resolve(strict=True)
    # Also serialize future managed invocations against this active workload.
    with (root/'.xtoken-managed.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = root/(MIRROR_NAME if mirror else NAME)
        out.mkdir(exist_ok=False, mode=0o700)
        state = dict(status='preflight', requested_optimizer_updates=10,
                     training_phase_invoked=False, automatic_retry=False,
                     started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                     pid=os.getpid(), parent_pid=os.getppid(),
                     reason=('User authorized transport-only TUNA fallback after bounded PyPI download timeout'
                             if mirror else 'User authorized launcher BrokenPipe repair and one cache-based continuation'))

        def save():
            state['updated_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
            pending = out/'result.pending.json'
            pending.write_text(json.dumps(state, indent=2)+'\n')
            pending.replace(out/'result.json')

        save()
        try:
            predecessor, paths = (validate_mirror_recovery(root) if mirror else validate_recovery(root))
            resource_gate()
            files = list(HERE.glob('*.py')) + [HERE/p for p in SCIENTIFIC]
            files.append(HERE.parent/'xtoken/scripts/run_logged.py')
            hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
            (out/'source-fingerprints.json').write_text(json.dumps(hashes, indent=2)+'\n')
            state['status'] = 'installing'
            save()
            rc = execute(HERE/'run_phase.py', ['xtoken', 'sync', '--root', str(root),
                         '--sync-timeout-seconds', '5400', '--sync-http-timeout-seconds', '300'] +
                         (['--pypi-mirror'] if mirror else []),
                         out/'install.log')
            evidence = re.findall(r'^EVIDENCE_DIR=(.+)$', (out/'install.log').read_text(), re.MULTILINE)
            assert len(evidence) == 1
            setup = Path(evidence[0])/'result.json'
            assert setup.resolve().is_relative_to(root/'evidence')
            state['setup_result'] = str(setup)
            state['setup_exit_code'] = rc
            save()
            if rc != 0 or read_json(setup)['exit_code'] != 0:
                state.update(status='setup_failed', exit_code=rc or 1)
                return state['exit_code']
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
                       for p, sha in hashes.items()), 'Source changed during install'
            state['status'] = 'running_sequence'
            sequence = root/'sequence-target-4b-2048-b64-full-10steps-r3'
            state['sequence_result'] = str(sequence/'result.json')
            save()
            # Producers are terminal now; this bounds readiness only, not phases.
            deadline = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=2)
            rc = execute(HERE/'run_target_after_setup.py', [
                '--root', str(root), '--setup-result', str(setup),
                '--teacher-result', str(paths[1]), '--student-result', str(paths[2]),
                '--ready-deadline-utc', deadline.isoformat(), '--steps', '10',
                '--sequence-attempt', '3', '--continuation-of', str(predecessor)], out/'sequence.log')
            terminal = read_json(sequence/'result.json')
            state.update(status=terminal['status'], exit_code=rc,
                         training_phase_invoked=terminal['training_phase_invoked'])
            return rc
        except Exception as error:
            state.update(status='stopped', exit_code=1, error_type=type(error).__name__)
            return 1
        except KeyboardInterrupt:
            state.update(status='interrupted', exit_code=130)
            return 130
        finally:
            state['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
            save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--pypi-mirror', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    # The worker does not need any credentials or arbitrary inherited debug flags.
    home_path = os.environ.get('HOME', '/root')
    os.environ.clear()
    os.environ.update(PATH='/usr/local/bin:/usr/bin:/bin', HOME=home_path,
                      LANG='C.UTF-8', PYTHONUNBUFFERED='1')
    if args.preflight_only:
        assert args.root.is_absolute() and args.root not in (Path('/'), Path.home())
        (validate_mirror_recovery if args.pypi_mirror else validate_recovery)(args.root.resolve(strict=True))
        resource_gate()
        print(json.dumps(dict(preflight='passed', workload_started=False)))
        return 0
    return run(args.root, mirror=args.pypi_mirror)


if __name__ == '__main__':
    sys.exit(main())
