"""One authorized, fail-closed target sequence; no install/download/training retries.

Waits only for the three explicitly supplied existing setup results. Each actual
phase retains its own hard deadline and process-group cleanup in run_logged.
This foreground process is not a durable service or a recurring automation.
"""
import argparse
from contextlib import redirect_stdout
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'xtoken/scripts'))
from run_logged import BestEffortOutput, workload_environment

EXPECTED_MISSING = {'nvidia-nvshmem-cu13', 'nvidia-cudnn-cu13', 'tilelang',
                    'mooncake-transfer-engine-cuda13', 'nixl-cu13'}


def dependency_result(path, student=False):
    if not path.is_file():
        return 'waiting'
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError:
        return 'waiting'  # An existing producer may be finalizing its small JSON.
    if student:
        if result.get('status') == 'verified' and result.get('transfer_verified') is True:
            return 'ready'
        return 'waiting' if result.get('status') == 'running' else 'failed'
    if result.get('exit_code') is None:
        return 'waiting'
    return 'ready' if result['exit_code'] == 0 else 'failed'


def expected_metadata_only(text):
    lines = [line for line in text.splitlines() if line.startswith('The package ')]
    missing = set()
    for line in lines:
        match = re.fullmatch(r"The package `[^`]+` requires `([a-z0-9-]+)[^`]*`, but it's not installed", line)
        if match is None:
            return False
        missing.add(match[1])
    return len(lines) == 5 and missing == EXPECTED_MISSING and 'Found 5 incompatibilities' in text


def validate_setup_only_predecessor(record):
    assert record['status'] == 'stopped' and record['exit_code'] == 1
    assert record['optimizer_updates'] == 10
    assert record['training_phase_invoked'] is False and record['phases'] == []


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = BestEffortOutput(stream), log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        self.stream.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def execute(script, arguments, log):
    previous = sys.argv
    try:
        sys.argv = [str(script), *arguments]
        with log.open('x') as stream, redirect_stdout(Tee(sys.stdout, stream)):
            try:
                runpy.run_path(str(script), run_name='__main__')
            except SystemExit as stopped:
                return 0 if stopped.code is None else int(stopped.code)
        return 0
    finally:
        sys.argv = previous


def resource_gate(projection=False):
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=15,
                            env=workload_environment())
    used = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    assert len(used) == 2 and max(used) <= 256, 'Two idle GPUs required; do not cancel other work'
    if projection:
        base = Path('/sys/fs/cgroup')
        limit = int((base/'memory.max').read_text())
        current = int((base/'memory.current').read_text())
        stats = dict(line.split() for line in (base/'memory.stat').read_text().splitlines())
        headroom = limit - current + int(stats['inactive_file'])
        assert headroom >= 80 * 1024**3, 'Insufficient reclaimable cgroup headroom for full projection'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--setup-result', required=True, type=Path)
    parser.add_argument('--teacher-result', required=True, type=Path)
    parser.add_argument('--student-result', required=True, type=Path)
    parser.add_argument('--ready-deadline-utc', required=True,
                        help='End of already-authorized setup window, not a new transfer timeout')
    parser.add_argument('--steps', required=True, type=int, choices=(10,))
    parser.add_argument('--sequence-attempt', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--continuation-of', type=Path,
                        help='Required for the separately authorized setup-only continuation')
    args = parser.parse_args()
    root = args.root.resolve()
    assert args.root.is_absolute() and root not in (Path('/'), Path.home())
    inputs = [args.setup_result, args.teacher_result, args.student_result]
    assert all(path.is_absolute() and path.resolve().is_relative_to(root) for path in inputs)
    deadline = dt.datetime.fromisoformat(args.ready_deadline_utc.replace('Z', '+00:00'))
    assert deadline.tzinfo is not None
    run_name = f'target-4b-2048-b64-full-10steps-r{args.sequence_attempt}'
    if args.sequence_attempt > 1:
        predecessor = root/f'sequence-target-4b-2048-b64-full-10steps-r{args.sequence_attempt - 1}'
        assert args.continuation_of == predecessor
        validate_setup_only_predecessor(json.loads((predecessor/'result.json').read_text()))
        old_hashes = json.loads((predecessor/'source-fingerprints.json').read_text())
        for relative in ('xtoken_target.py', 'prepare_xtoken_target.py',
                         'validate_xtoken_target.py', 'xtoken/pyproject.toml', 'xtoken/uv.lock'):
            path = HERE/relative
            assert hashlib.sha256(path.read_bytes()).hexdigest() == old_hashes[str(path)]
        assert args.student_result == root/'student-http-resume-r4/result.json'
    else:
        assert args.continuation_of is None
    out = root/('sequence-'+run_name)
    out.mkdir(exist_ok=False)
    state = dict(status='waiting_for_existing_setup', optimizer_updates=args.steps,
                 training_phase_invoked=False, inputs=[str(p) for p in inputs],
                 started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                 readiness_deadline=deadline.isoformat(), phases=[], automatic_retry=False,
                 sequence_attempt=args.sequence_attempt,
                 continuation_of=str(args.continuation_of) if args.continuation_of else None)

    def save():
        state['updated_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        temporary = out/'result.pending.json'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(out/'result.json')

    # Freeze task launchers and original NeMo Python/config sources before waiting.
    files = list(HERE.glob('*.py')) + [HERE/'xtoken/pyproject.toml', HERE/'xtoken/uv.lock',
                                      HERE.parent/'xtoken/scripts/run_logged.py']
    source = root/'NeMo-RL'
    files += [p for p in source.rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.yml')]
    fingerprints = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out/'source-fingerprints.json').write_text(json.dumps(fingerprints, indent=2)+'\n')

    def unchanged():
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == value
                   for p, value in fingerprints.items()), 'Source changed during pending sequence'

    save()
    console = BestEffortOutput(sys.stdout)
    print(json.dumps(dict(sequence=str(out), status=state['status'])), file=console, flush=True)
    try:
        while True:
            statuses = [dependency_result(p, student=i == 2) for i, p in enumerate(inputs)]
            state['dependencies'] = statuses
            save()
            if 'failed' in statuses:
                raise RuntimeError('An existing prerequisite failed; no retry or training launched')
            if statuses == ['ready'] * 3:
                break
            if dt.datetime.now(dt.timezone.utc) >= deadline:
                raise TimeoutError('Existing setup window ended without verified readiness')
            print(json.dumps(dict(waiting=True, dependencies=statuses)), file=console, flush=True)
            time.sleep(30)
        for phase in ('check', 'canary', 'models', 'data', 'projection', 'config', 'train'):
            unchanged()
            if phase in ('canary', 'projection', 'train'):
                resource_gate(projection=phase == 'projection')
            state['status'], state['current_phase'] = 'running_phase', phase
            if phase == 'train':
                state['training_phase_invoked'] = True
            save()
            if phase in ('check', 'canary'):
                script = HERE/'run_phase.py'
                arguments = ['xtoken', phase, '--root', str(root)]
            else:
                script = HERE/'xtoken_target.py'
                arguments = [phase, '--root', str(root), '--steps', str(args.steps), '--run-name', run_name]
            log = out/(phase+'.log')
            rc = execute(script, arguments, log)
            evidence = re.findall(r'^EVIDENCE_DIR=(.+)$', log.read_text(), re.MULTILINE)
            assert evidence and all(Path(p).resolve().is_relative_to(root/'evidence') for p in evidence)
            results = [json.loads((Path(p)/'result.json').read_text()) for p in evidence]
            accepted_metadata = (phase == 'check' and rc == 1 and len(results) == 1 and
                                 results[0]['exit_code'] == 1 and expected_metadata_only(
                                     (Path(evidence[0])/'stdout.log').read_text()))
            state['phases'].append(dict(phase=phase, exit_code=rc, evidence=evidence,
                                       expected_metadata_exclusions_only=accepted_metadata))
            save()
            assert accepted_metadata or (rc == 0 and all(r['exit_code'] == 0 for r in results)), \
                f'{phase} failed; no automatic retry'
        state['status'] = 'training_process_complete_pending_metric_review'
        state['exit_code'] = 0
    except Exception as error:
        state.update(status='stopped', exit_code=1, error_type=type(error).__name__, error=str(error))
    except KeyboardInterrupt:
        state.update(status='interrupted', exit_code=130)
    finally:
        state['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()
    print(json.dumps(state), file=console, flush=True)
    return state['exit_code']


if __name__ == '__main__':
    sys.exit(main())
