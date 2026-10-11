"""Independent math-verify CPU score; isolated dependencies and raw outputs."""
import importlib.metadata
import json
from pathlib import Path
import signal
import sys
sys.path.insert(0, '/assets/mathverify080-v1')
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

OUT = Path('/results')


def score(gold, prediction):
    target = parse('$'+gold+'$', extraction_config=[LatexExtractionConfig()])
    if not target: raise ValueError('Gold math answer unparseable')
    result = parse(prediction, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
    return bool(result and verify(target, result))


assert score(r'\frac{1}{2}', r'\boxed{0.5}')
assert not score('2', r'\boxed{3}')
assert score(r'145^\circ', r'\boxed{145}')
summaries = {}
for arm in ('qwen-nonthinking', 'qwen-thinking'):
    source = OUT/(arm+'-responses.json')
    if not source.exists(): continue
    records = json.loads(source.read_text())
    rows = []
    for row in records:
        gold = row['gold'].split('####')[-1].strip() if row['benchmark']=='gsm8k' else row['gold']
        error = None
        try: passed = score(gold, row['final'])
        except Exception as exc: passed = None; error = type(exc).__name__+': '+str(exc)
        rows.append({'benchmark': row['benchmark'], 'index': row['index'],
                     'passed': passed, 'error': error, 'previous': row['passed']})
    (OUT/(arm+'-mathverify.json')).write_text(json.dumps(rows, indent=2))
    summaries[arm] = {bench: {'n': sum(r['benchmark']==bench for r in rows),
        'correct': sum(r['benchmark']==bench and r['passed'] is True for r in rows),
        'errors': sum(r['benchmark']==bench and r['passed'] is None for r in rows),
        'changed': sum(r['benchmark']==bench and r['passed'] != r['previous'] for r in rows)}
        for bench in ('math500','gsm8k')}
result = {'scorer': 'math-verify isolated CPU, contract extraction settings',
          'versions': {k: importlib.metadata.version(k) for k in
                       ('math-verify','latex2sympy2_extended','antlr4-python3-runtime','sympy')},
          'arms': summaries}
(OUT/'mathverify-summary.json').write_text(json.dumps(result, indent=2))
print('MATHVERIFY_SUMMARY='+json.dumps(result), flush=True)
