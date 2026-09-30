"""Receive HF token on stdin; fetch the pinned smoke pair without persisting auth."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

MODELS = [
    ('student', 'meta-llama/Llama-3.2-1B', '4e20de362430cd3b72f300e6b0f18e50e7166e08'),
    ('teacher', 'Qwen/Qwen3-1.7B', '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.root.is_absolute() or root in (Path('/'), Path.home()):
        parser.error('dedicated absolute runtime root required')
    token = sys.stdin.read().strip()
    if not token.startswith('hf_'):
        raise SystemExit('Missing designated HF token; no network request made')
    out = root/'models-evidence'
    out.mkdir(exist_ok=False)
    report = dict(started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  attempt=1, complete=False, token_persisted=False, models=[])
    env = {k: v for k, v in os.environ.items() if not any(
        s in k.upper() for s in ('TOKEN', 'SECRET', 'PASSWORD', 'API_KEY'))}
    env.update(HF_HOME=str(root/'hf'), HF_HUB_DISABLE_TELEMETRY='1',
               HF_HUB_DISABLE_IMPLICIT_TOKEN='0', HF_HUB_DOWNLOAD_TIMEOUT='90')
    def save():
        (out/'models.json').write_text(json.dumps(report, indent=2)+'\n')
    def safe(s):
        return re.sub(r'https?://[^\s]+', '[REMOTE_URL]', s.replace(token, '[REDACTED]'))
    try:
        for role, repo, rev in MODELS:
            model_env = dict(env)
            if role == 'student':
                model_env['HF_TOKEN'] = token
            row = dict(role=role, repo=repo, revision=rev, weights_verified=False)
            report['models'].append(row)
            save()
            for phase, pattern in [('metadata', '*.json'), ('weights', '*.safetensors')]:
                command = [str(root/'xtoken/venv/bin/hf'), 'download', repo,
                           '--revision', rev, '--cache-dir', str(root/'hf/hub'),
                           '--include', pattern, '--max-workers', '4', '--quiet']
                proc = subprocess.run(command, env=model_env, capture_output=True,
                                      text=True, timeout=1500)
                (out/f'{role}-{phase}.log').write_text(safe(proc.stdout+'\n'+proc.stderr))
                row[phase+'_exit_code'] = proc.returncode
                save()
                if proc.returncode:
                    print(json.dumps(dict(role=role, phase=phase, status='FAILED')))
                    return proc.returncode
            snapshot = root/'hf/hub'/('models--'+repo.replace('/', '--'))/'snapshots'/rev
            weights = list(snapshot.glob('*.safetensors'))
            assert weights and sum(p.stat().st_size for p in weights) > 1_000_000_000
            index = snapshot/'model.safetensors.index.json'
            if index.exists():
                assert set(json.loads(index.read_text())['weight_map'].values()) <= {p.name for p in weights}
            for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
                assert (snapshot/name).is_file(), name
            row.update(snapshot=str(snapshot), weights_verified=True,
                       files=[dict(name=p.name, bytes=p.stat().st_size,
                                   sha256=hashlib.file_digest(p.open('rb'), 'sha256').hexdigest())
                              for p in sorted(snapshot.iterdir()) if p.is_file()])
            print(json.dumps(dict(role=role, status='PASS', files=len(row['files']))), flush=True)
        report['complete'] = True
        return 0
    except subprocess.TimeoutExpired:
        report['error'] = 'download_timeout_no_retry'
        return 124
    finally:
        report['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    sys.exit(main())
