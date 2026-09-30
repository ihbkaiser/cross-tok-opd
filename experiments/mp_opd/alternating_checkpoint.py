"""Crash-consistent single-process pilot checkpoints and derived journals."""
import contextlib
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch

SCHEMA = 'mp-alternating-resume-v2'


def atomic_write(path, writer):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextlib.contextmanager
def output_lock(directory, resume):
    import fcntl
    directory = Path(directory)
    if resume:
        if not (directory / 'latest.pt').is_file():
            raise ValueError('resume requires existing latest.pt in output directory')
    else:
        directory.mkdir(parents=True, exist_ok=False)
    with (directory / '.writer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another pilot owns this output directory') from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def contract(manifest):
    value = dict(manifest)
    value['args'] = {k:v for k,v in manifest['args'].items()
                     if k not in {'output','resume','stop_after_groups'}}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def rng_state():
    return {'torch':torch.get_rng_state(), 'python':random.getstate(),
            'numpy':np.random.get_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    if len(state['cuda']) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError('CUDA device count changed since checkpoint')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def materialize(directory, payload):
    """Checkpoint is the commit record; journals are recoverable projections."""
    for name,key in (('results.jsonl','results'),('trajectories.jsonl','traces')):
        content=''.join(json.dumps(row,allow_nan=False)+'\n' for row in payload[key]).encode()
        atomic_write(Path(directory)/name, lambda f, content=content:f.write(content))
    status=('completed' if payload['step'] else 'no_valid_updates') if payload['cursor']==payload['total_groups'] else 'in_progress'
    budget=payload['manifest']['args'].get('max_student_updates')
    if budget is not None:
        status='completed' if payload['step']>=budget else ('budget_unmet' if payload['cursor']==payload['total_groups'] else 'in_progress')
    summary={'schema':SCHEMA,'status':status,'next_group':payload['cursor'],
             'target_student_updates':budget,'total_groups':payload['total_groups'],
             'student_updates':payload['step'],'energy_updates':payload['energy_updates'],
             'invalid_groups':payload['invalid'], 'scope':payload['manifest']['scope'],
             'evidence':'adapter training pilot; no benchmark efficacy claim',
             'checkpoint':str(Path(directory)/'latest.pt')}
    content=json.dumps(summary,indent=2).encode()
    atomic_write(Path(directory)/'summary.json',lambda f:f.write(content))


def save(directory, adapter, energy, optimizer, manifest, results, traces, total_groups):
    valid=[r for r in results if not r['invalid']]
    payload={'schema':SCHEMA,'contract':contract(manifest),'manifest':manifest,
             'cursor':len(results),'total_groups':total_groups,'step':len(valid),
             'energy_updates':sum(r['energy_updates'] for r in valid),
             'invalid':len(results)-len(valid),'results':results,'traces':traces,
             'a':adapter.a.detach().cpu(),'b':adapter.b.detach().cpu(),
             'energy':energy.state_dict(),'energy_optimizer':optimizer.state_dict(),
             'rng':rng_state()}
    atomic_write(Path(directory)/'latest.pt',lambda f:torch.save(payload,f))
    materialize(directory,payload)
    return payload


def load(directory, adapter, energy, optimizer, manifest, total_groups):
    # Only load trusted locally-created pilot checkpoints, never arbitrary uploads.
    payload=torch.load(Path(directory)/'latest.pt',map_location='cpu',weights_only=False)
    if payload.get('schema')!=SCHEMA:
        raise ValueError('unsupported resume checkpoint; v1 has no reliable cursor/journal state')
    if payload['contract']!=contract(manifest):
        raise ValueError('resume provenance/configuration mismatch')
    results,traces=payload['results'],payload['traces']
    cursor=payload['cursor']
    if not (cursor==len(results)==len(traces) and 0<=cursor<=total_groups==payload['total_groups']):
        raise ValueError('invalid resume cursor')
    if [r['index'] for r in results]!=list(range(cursor)) or [t['index'] for t in traces]!=list(range(cursor)):
        raise ValueError('noncontiguous journal indices')
    valid=[r for r in results if not r['invalid']]
    if payload['step']!=len(valid) or payload['invalid']!=cursor-len(valid) or payload['energy_updates']!=sum(r['energy_updates'] for r in valid):
        raise ValueError('resume counters disagree with journals')
    if adapter.a.shape!=payload['a'].shape or adapter.b.shape!=payload['b'].shape:
        raise ValueError('adapter shape mismatch')
    with torch.no_grad():
        adapter.a.copy_(payload['a']);adapter.b.copy_(payload['b'])
    energy.load_state_dict(payload['energy'])
    optimizer.load_state_dict(payload['energy_optimizer'])
    # Restore after model/optimizer construction and loading consumed RNG.
    restore_rng(payload['rng'])
    materialize(directory,payload)
    return payload