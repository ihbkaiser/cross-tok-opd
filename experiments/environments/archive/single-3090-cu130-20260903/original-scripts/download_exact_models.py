"""Protected, single-attempt download of the user-authorized exact model pair."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path('/workspace/xtoken-native')
OUT = ROOT / 'smoke-20260903-r1'
MODELS = [
    ('student', 'meta-llama/Llama-3.2-1B', '4e20de362430cd3b72f300e6b0f18e50e7166e08'),
    ('teacher', 'Qwen/Qwen3-1.7B', '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'),
]

def main():
    token = sys.stdin.read().strip()
    if not token.startswith('hf_'):
        raise SystemExit('Missing designated HF token; no request made')
    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / 'model_weights_verified.json'
    if report_path.exists():
        raise SystemExit('Existing attempt evidence: refusing an implicit rerun')
    report = dict(started_at=dt.datetime.now(dt.timezone.utc).isoformat(), attempt=3,
                  retry_reason='Authenticated access passed; CLI interpreted extra include patterns as filenames, so prior pass only proves metadata download. Now download and verify actual safetensors.',
                  token_persisted=False, models=[], complete=False)
    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    def safe(text):
        text = text.replace(token, '[REDACTED]')
        return re.sub(r'(https?://[^\s?]+)\?[^\s]+', r'\1?[REDACTED]', text)
    save()
    env = {k: v for k, v in os.environ.items() if not any(
        key in k.upper() for key in ('TOKEN', 'SECRET', 'PASSWORD', 'API_KEY'))}
    env.update(HF_TOKEN=token, HF_HOME=str(ROOT/'hf-cache'), HF_HUB_DISABLE_TELEMETRY='1',
               HF_HUB_DISABLE_IMPLICIT_TOKEN='0', HF_HUB_DOWNLOAD_TIMEOUT='90')
    hf = ROOT/'NeMo-RL/.venv/bin/hf'
    try:
        for role, repo, revision in MODELS:
            row = dict(role=role, repo=repo, revision=revision)
            report['models'].append(row)
            for phase, files in [('weights_verified', [])]:
                cmd = [str(hf), 'download', repo, *files, '--revision', revision,
                       '--cache-dir', str(ROOT/'hf-cache/hub'), '--quiet']
                cmd += ['--include', '*.safetensors', '--max-workers', '4']
                start = time.monotonic()
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                      timeout=180 if phase == 'access' else 1500)
                log = safe(proc.stdout+'\n'+proc.stderr)
                (OUT/f'{role}_{phase}_authenticated.log').write_text(log)
                row[phase] = dict(exit_code=proc.returncode, seconds=time.monotonic()-start)
                if proc.returncode:
                    save()
                    print(json.dumps(dict(role=role, phase=phase, status='FAILED',
                                          exit_code=proc.returncode)), flush=True)
                    return proc.returncode
                print(json.dumps(dict(role=role, phase=phase, status='PASS')), flush=True)
                save()
            snapshot = ROOT/'hf-cache/hub'/('models--'+repo.replace('/', '--'))/'snapshots'/revision
            row['snapshot_path'] = str(snapshot)
            row['files'] = [dict(name=p.name, bytes=p.stat().st_size) for p in sorted(snapshot.iterdir()) if p.is_file()]
            weight_files = list(snapshot.glob('*.safetensors'))
            assert weight_files and sum(p.stat().st_size for p in weight_files) > 1_000_000_000, 'Missing actual weights'
            index = snapshot/'model.safetensors.index.json'
            if index.exists():
                expected = set(json.loads(index.read_text())['weight_map'].values())
                assert expected.issubset({p.name for p in weight_files}), 'Incomplete shards'
            row['weights_verified'] = True
            row['config_sha256'] = hashlib.sha256((snapshot/'config.json').read_bytes()).hexdigest()
            save()
        report['complete'] = True
        return 0
    except subprocess.TimeoutExpired:
        report['error'] = 'explicit_download_timeout_no_retry'
        return 124
    finally:
        report['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()
        env.pop('HF_TOKEN', None)

if __name__ == '__main__':
    sys.exit(main())
