"""Publish HF export readiness only after the training transaction commits."""
import json
import os
from pathlib import Path


def marker(root, step):
    return Path(root) / 'export-ready' / f'step{step}.json'


def publish(root, step):
    root = Path(root)
    folder = root / f'step{step}'
    files = {str(p.relative_to(folder)): p.stat().st_size
             for p in folder.rglob('*') if p.is_file()}
    if not files or 'config.json' not in files:
        raise ValueError('Incomplete HF export')
    dest = marker(root, step)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(step=step, path=str(folder.resolve()), files=files)
    temp = dest.with_suffix('.pending')
    with temp.open('w') as f:
        json.dump(payload, f); f.flush(); os.fsync(f.fileno())
    os.replace(temp, dest)
    fd = os.open(dest.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def validate(root, step):
    root = Path(root)
    value = json.loads(marker(root, step).read_text())
    folder = root / f'step{step}'
    actual = {str(p.relative_to(folder)): p.stat().st_size
              for p in folder.rglob('*') if p.is_file()}
    if value != dict(step=step, path=str(folder.resolve()), files=actual):
        raise ValueError('Export readiness mismatch')
    return folder
