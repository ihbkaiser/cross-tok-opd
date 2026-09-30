"""Prepare bounded oracle retries using intact guide data and local tokenizers."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.mp_opd.real_oracle import preflight_lengths, validate_groups


def budgets(deadline, now):
    remaining = int(deadline - now) - 300
    if remaining < 1200:
        raise ValueError("Less than 25 minutes remain; do not reset campaign clock")
    return remaining, (remaining - 120) // 2


def main(original, guide, out):
    from transformers import AutoTokenizer, AutoConfig
    prior = json.loads((original / 'campaign-state.json').read_text())
    plan = json.loads((original / 'plan.json').read_text())
    deadline = prior['started'] + plan['hours'] * 3600
    budgets(deadline, time.time())
    cfg = json.loads((guide / 'config.json').read_text())
    data_path = guide / 'probe-data.json'
    raw = data_path.read_bytes()
    data = json.loads(raw)
    audit = validate_groups(data['groups'])
    tokenizers = [AutoTokenizer.from_pretrained(cfg[k], local_files_only=True, trust_remote_code=False)
                  for k in ('student', 'teacher')]
    lengths = preflight_lengths(data['groups'], *tokenizers, 4096, 4096)
    # Conservative upper bound includes EOS in reference length. No truncation/filtering.
    limits = [AutoConfig.from_pretrained(cfg[k], local_files_only=True, trust_remote_code=False).max_position_embeddings
              for k in ('student', 'teacher')]
    if lengths['student_prompt'] + max(lengths['reference'], 128) > limits[0]:
        raise ValueError('Student context capacity exceeded')
    if lengths['teacher_prompt'] + 512 > limits[1]:
        raise ValueError('Teacher context margin insufficient')
    primary, sensitivity = budgets(deadline, time.time())
    out.mkdir(parents=True, exist_ok=False)
    (out / 'probe-data.json').write_bytes(raw)
    report = dict(lengths=lengths, audit=audit, data_sha256=hashlib.sha256(raw).hexdigest(),
                  context_limits=limits, prompt_cap=4096, reference_cap=4096, deadline=deadline,
                  scope='Same guide groups; adapter-only diagnostic; no automatic training promotion')
    (out / 'preflight.json').write_text(json.dumps(report, indent=2))
    jobs = []
    for name, lr, gpu, budget in [('probe-1', '.1', 0, primary), ('probe-0', '.01', 1, sensitivity), ('probe-2', '1.0', 1, sensitivity)]:
        argv = ['bash', str(ROOT / 'experiments/runai/python-b200-host.sh'),
                str(ROOT / 'experiments/mp_opd/real_oracle.py'), 'run',
                '--student', cfg['student'], '--teacher', cfg['teacher'],
                '--data', str(out / 'probe-data.json'), '--output', str(out / name),
                '--adapter-module', 'model.layers.25.self_attn.q_proj', '--device', 'cuda:0',
                '--rank', '4', '--virtual-lr', lr, '--model-dtype', 'float32', '--max-span', '2',
                '--weighting-steps', '1', '--select-counts', '1', '4', '--max-new-tokens', '128',
                '--max-prompt-tokens', '4096', '--max-reference-tokens', '4096']
        jobs.append(dict(id=name, gpu=gpu, budget_seconds=budget, argv=argv))
    campaign = dict(hours=plan['hours'], source=str(ROOT),
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), jobs=jobs)
    target = out / 'plan.json'
    target.write_text(json.dumps(campaign, indent=2))
    state = dict(plan_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), started=prior['started'], jobs={})
    (out / 'campaign-state.json').write_text(json.dumps(state, indent=2))
    print('PREFLIGHT_PASS', json.dumps(report), flush=True)
    print('READY', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('original', 'guide', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    a = parser.parse_args()
    main(a.original.resolve(), a.guide.resolve(), a.out.resolve())
