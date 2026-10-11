"""CPU-only regrading with the missing pinned LaTeX parser, no model rerun."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

deps = Path('/assets/author-scorer-antlr411-v1')
if not (deps/'antlr4').is_dir():
    env = dict(os.environ)
    env.pop('UV_OFFLINE', None)
    if shutil.which('uv'):
        command = ['uv', 'pip', 'install', '--python', sys.executable, '--target', str(deps)]
    else:
        command = [sys.executable, '-m', 'pip', 'install', '--target', str(deps)]
    subprocess.run(command+['--no-deps', 'antlr4-python3-runtime==4.11.1'], env=env, check=True, timeout=180)
sys.path.insert(0, str(deps))
sys.path.insert(0, '/opt/screen')
from worker import audited_helpers, OUT
h = audited_helpers()
assert h['is_math_equivalent'](r'\frac{1}{2}', '0.5')
assert not h['is_math_equivalent'](r'\frac{1}{2}', '0.6')


def alarm(signum, frame): raise TimeoutError('Symbolic grading exceeded 5 seconds')


signal.signal(signal.SIGALRM, alarm)
summaries = {}
for arm in ('qwen-nonthinking', 'qwen-thinking'):
    source = OUT/(arm+'-responses.json')
    if not source.exists(): continue
    rows = json.loads(source.read_text())
    graded = []
    for row in rows:
        value = row['passed']; error = None
        if row['benchmark'] == 'math500':
            try:
                signal.alarm(5)
                value = bool(h['is_math_equivalent'](h['extract_boxed_answer'](row['final']), row['gold']))
            except Exception as exc:
                value = None; error = type(exc).__name__+': '+str(exc)
            finally: signal.alarm(0)
        graded.append({'benchmark': row['benchmark'], 'index': row['index'],
                       'passed': value, 'previous': row['passed'], 'error': error})
    (OUT/(arm+'-regraded.json')).write_text(json.dumps(graded, indent=2))
    summaries[arm] = {bench: {'n': sum(r['benchmark']==bench for r in graded),
        'correct': sum(r['benchmark']==bench and r['passed'] is True for r in graded),
        'errors': sum(r['benchmark']==bench and r['passed'] is None for r in graded),
        'changed': sum(r['benchmark']==bench and r['passed'] != r['previous'] for r in graded)}
        for bench in ('math500', 'gsm8k')}
result = {'scorer': 'canonical author-code with bounded symbolic parsing',
          'antlr_version': importlib.metadata.version('antlr4-python3-runtime'),
          'sympy_version': importlib.metadata.version('sympy'), 'arms': summaries}
(OUT/'regraded-summary.json').write_text(json.dumps(result, indent=2))
print('REGRADED_SUMMARY='+json.dumps(result), flush=True)

modern = Path('/assets/mathverify080-v1')
if not (modern/'math_verify').is_dir():
    env = dict(os.environ); env.pop('UV_OFFLINE', None)
    if shutil.which('uv'):
        command = ['uv', 'pip', 'install', '--python', sys.executable, '--target', str(modern)]
    else: command = [sys.executable, '-m', 'pip', 'install', '--target', str(modern)]
    subprocess.run(command+['math-verify==0.8.0'], env=env, check=True, timeout=180)
subprocess.run([sys.executable, '-u', '/opt/screen/mathverify_worker.py'], check=True, timeout=240)
