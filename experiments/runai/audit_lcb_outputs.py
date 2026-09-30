"""Read-only LCB journal diagnostics; never repair journals or change scores."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/evaluation'))
from contract_eval import author_helpers


def read_rows(path):
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row['id'] in rows:
            raise ValueError('Duplicate ID: '+str(path))
        rows[row['id']] = row
    return rows


def audit(cell, helpers):
    metrics = json.loads((cell/'metrics.json').read_text())
    for name in ('responses','scores'):
        path = cell/(name+'.jsonl')
        if hashlib.sha256(path.read_bytes()).hexdigest() != metrics[name+'_sha256']:
            raise ValueError('Journal hash mismatch: '+str(path))
    responses = read_rows(cell/'responses.jsonl')
    scores = read_rows(cell/'scores.jsonl')
    if set(responses) != set(scores) or len(scores) != metrics['count']:
        raise ValueError('Incomplete/mismatched cell: '+str(cell))
    counts = Counter()
    for key, row in responses.items():
        choice = row['response']['choices'][0]
        text = choice['message']['content'] or ''
        cleaned, _, truncated_thinking = helpers['strip_thinking_content'](text)
        code = helpers['_extract_code_block'](cleaned)
        flags = {'finish_length':choice.get('finish_reason')=='length',
                 'empty_extracted':not code.strip(), 'no_code_fence':'```' not in cleaned,
                 'truncated_thinking':truncated_thinking,
                 'outer_timeout':scores[key].get('timeout',False)}
        try:
            ast.parse(code)
            flags['syntax_error'] = False
        except (SyntaxError, ValueError):
            flags['syntax_error'] = True
        passed = scores[key]['passed']
        counts['total'] += 1
        counts['passed'] += int(passed)
        for name, value in flags.items():
            if value:
                counts[name] += 1
                counts[name+'_failed'] += int(not passed)
    return dict(cell=str(cell), counts=dict(counts),
                note='Overlapping diagnostic flags; valid syntax does not imply correct code. Inner test timeouts were not retained by the old scorer.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    args = parser.parse_args()
    if args.root is None:
        cfg = json.loads(Path('/workspace/storage-shared/nlp/tungks/campaign20-A7hEqD4H/work/config.json').read_text())
        args.root = Path(cfg['eval_template']).parent
    print('AUDIT_ROOT:',args.root)
    helpers = author_helpers()
    cells = sorted(args.root.glob('cells/*/live-code-bench-v6/*/metrics.json'))
    if not cells:
        print('NO_QUEUE_CELLS: provide --root for the historical response/score journal directory')
    for path in cells:
        print(json.dumps(audit(path.parent,helpers)))


if __name__ == '__main__':
    main()
