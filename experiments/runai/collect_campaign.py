#!/usr/bin/env python3
"""Read shared campaign evidence; never launch jobs or modify source artifacts.

Default inventories metadata. --export writes a private evidence archive;
--responses includes model responses. Model weights are inventoried, not copied.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import tarfile
import time

BASE = Path('/workspace/storage-shared/nlp/tungks/borrow8-8MgodXcM')
OLD = Path('/workspace/storage-shared/nlp/tungks/simct-eval-queue-qjG2gMc2/trial256-20260910-083920-466282')
PRUNE = {'.git', '.venv', 'venv', 'env', 'cache', '.cache', '__pycache__',
         'source', 'node_modules', 'wandb', 'job-manager', 'exports', 'evidence-exports'}
WEIGHTS = {'.safetensors', '.bin', '.pt', '.pth', '.ckpt'}

def selected(p, responses):
    name = p.name.lower()
    if name in {'tokenizer.json', 'vocab.json'}:
        return False
    if any(x in name for x in ('secret', 'credential', '.env', 'netrc')):
        return False
    if name == 'responses.jsonl':
        return responses
    return (p.suffix in {'.json', '.log', '.exitcode'} or
            name in {'exitcode', 'scores.jsonl', 'metrics.jsonl', 'history.jsonl'})

def inventory(roots, responses=False):
    result = {'schema': 1, 'host': socket.gethostname(), 'time_unix': time.time(),
              'roots': [], 'files': [], 'checkpoints': [], 'errors': [],
              'warning': 'Private evidence. Review/redact before external publication. '
                         'Directory existence does not validate a checkpoint.'}
    for index, root in enumerate(roots):
        root = root.resolve()
        result['roots'].append({'alias': f'root{index}', 'path': str(root), 'exists': root.is_dir()})
        if not root.is_dir():
            continue
        def walk_error(exc):
            result['errors'].append(str(exc))
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
            d = Path(directory)
            dirs[:] = sorted(x for x in dirs if x not in PRUNE and not (d/x).is_symlink()
                             and not x.startswith(('publish-', 'source-', 'lcbfix-source-', 'six-source-')))
            if re.fullmatch(r'step\d+', d.name) or any(Path(f).suffix in WEIGHTS for f in files):
                weights = []
                for f in files:
                    p = d/f
                    if p.suffix in WEIGHTS and not p.is_symlink():
                        try:
                            weights.append({'name': f, 'bytes': p.stat().st_size})
                        except OSError as exc:
                            result['errors'].append(str(exc))
                result['checkpoints'].append({'path': str(d), 'weights': weights,
                                              'validated': False})
            for f in sorted(files):
                p = d/f
                if p.is_symlink() or not selected(p, responses):
                    continue
                try:
                    s = p.stat()
                    result['files'].append({'source': str(p), 'member': f'root{index}/'+p.relative_to(root).as_posix(),
                                            'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns})
                except OSError as exc:
                    result['errors'].append(str(exc))
    result['total_bytes'] = sum(x['bytes'] for x in result['files'])
    return result

def export(report, output):
    # Exclusive creation prevents replacing earlier evidence.
    with output.open('xb') as raw, tarfile.open(fileobj=raw, mode='w:gz') as tar:
        for entry in report['files']:
            p = Path(entry['source'])
            try:
                with p.open('rb') as f:
                    before = os.fstat(f.fileno())
                    data = f.read()
                    after = os.fstat(f.fileno())
                entry['sha256'] = hashlib.sha256(data).hexdigest()
                entry['exported_bytes'] = len(data)
                entry['changed_during_collection'] = (before.st_mtime_ns != after.st_mtime_ns or
                    before.st_size != after.st_size or before.st_mtime_ns != entry['mtime_ns'])
                info = tarfile.TarInfo(entry['member'])
                info.size, info.mode, info.mtime = len(data), 0o600, int(before.st_mtime)
                tar.addfile(info, io.BytesIO(data))
            except OSError as exc:
                entry['export_error'] = str(exc)
                report['errors'].append(str(exc))
        data = json.dumps(report, indent=2).encode()
        info = tarfile.TarInfo('manifest.json')
        info.size, info.mode = len(data), 0o600
        tar.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256()
    with output.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    output.with_suffix(output.suffix+'.sha256').write_text(digest.hexdigest()+'  '+output.name+'\n')
    return digest.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, action='append', help='Repeat to include additional historical run roots')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--export', action='store_true')
    p.add_argument('--responses', action='store_true')
    args = p.parse_args()
    roots = args.root or [BASE, OLD, Path('/workspace/storage-shared/nlp/tungks/SimCT/runs')]
    report = inventory(roots, args.responses)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = f'{time.time_ns()}-{socket.gethostname()}'
    dest = args.out/f'campaign-{stamp}.inventory.json'
    dest.write_text(json.dumps(report, indent=2))
    print('INVENTORY', dest)
    print(json.dumps({k: report[k] for k in ('roots', 'total_bytes', 'errors')}))
    print('FILES', len(report['files']), 'CHECKPOINT_DIRECTORIES', len(report['checkpoints']))
    if args.export:
        archive = args.out/f'campaign-{stamp}.tar.gz'
        print('ARCHIVE', archive, 'SHA256', export(report, archive))
        print('COLLECTION_ERRORS', len(report['errors']))

if __name__ == '__main__':
    main()
