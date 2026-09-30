"""Prepare an independent guide-only recovery; never mutate a live campaign."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def prepare(original, out):
    root = Path(__file__).resolve().parents[2]
    previous = json.loads((original / 'plan.json').read_text())
    state = json.loads((original / 'campaign-state.json').read_text())
    deadline = state['started'] + previous['hours'] * 3600
    if deadline <= time.time():
        raise ValueError('Original campaign deadline has passed')
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((original / 'config.json').read_text())
    cfg['reference_input'] = str(out / 'guide.jsonl')
    cfg['reference_key'] = 'label'
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    jobs = []
    def add(key, action, hours, gpu=None, after=(), lr='.1'):
        jobs.append(dict(id=key, gpu=gpu, after=list(after), budget_seconds=int(hours * 3600),
            argv=['/usr/bin/python3.12', str(root / 'experiments/runai/campaign_job.py'), action,
                  '--work', str(out), '--gpu', str(gpu or 0), '--name', key, '--lr', lr]))
    add('acquire-guide', 'guide-acquire', 1)
    add('generate-guide', 'guide-generate', 1.5, 1, ('acquire-guide',))
    add('filter-guide', 'guide-finalize', .1, after=('generate-guide',))
    add('prepare-guide', 'prepare-probe', .25, after=('filter-guide',))
    for i, lr in enumerate(('.01', '.1', '1.0')):
        add('probe-' + str(i), 'probe', .75, 1, ('prepare-guide',), lr)
    # A failed acquisition/probe is technical missing evidence, not a weak-signal pilot.
    add('probe-decision', 'decision', .05, after=('probe-0', 'probe-1', 'probe-2'))
    plan = dict(schema='campaign-v1', hours=previous['hours'], source=str(root),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        config_sha256=hashlib.sha256((out / 'config.json').read_bytes()).hexdigest(), jobs=jobs,
        recovery_of=str(original), original_deadline=deadline)
    path = out / 'plan.json'
    path.write_text(json.dumps(plan, indent=2))
    (out / 'campaign-state.json').write_text(json.dumps(dict(
        plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), started=state['started'], jobs={})))
    print('PLAN=' + str(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.original.resolve(), args.out.resolve())
