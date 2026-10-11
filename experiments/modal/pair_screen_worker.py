"""Small matched math screen; raw outputs are mandatory, not a paper score."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path('/assets')
OUT = Path('/results')
N = 128
SEED = 42
CAP = 4096


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''): h.update(chunk)
    return h.hexdigest()


def audited_helpers():
    sys.path.insert(0, '/opt/screen/scripts/evaluation')
    import contract_eval as ce
    source = Path('/opt/screen/scripts/evaluation/evaluation.py').read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(source).hexdigest() == ce.AUTHOR_SHA
    ce.HERE = OUT/'scorer'
    ce.HERE.mkdir(exist_ok=True)
    canonical = ce.HERE/'evaluation.py'
    if canonical.exists(): assert canonical.read_bytes() == source
    else: canonical.write_bytes(source)
    h = ce.author_helpers()
    assert h['is_math_equivalent']('2', '2') and not h['is_math_equivalent']('3', '2')
    return h


def prepare():
    from huggingface_hub import HfApi, snapshot_download
    import pyarrow.parquet as pq
    api = HfApi()
    assets = {}
    for key, repo, typ, patterns in [
        ('qwen', 'Qwen/Qwen3-1.7B', 'model', ['*.json', '*.safetensors', '*.jinja']),
        ('math500', 'HuggingFaceH4/MATH-500', 'dataset', ['test.jsonl']),
        ('gsm8k', 'openai/gsm8k', 'dataset', ['main/test-*.parquet']),
    ]:
        rev = api.repo_info(repo, repo_type=typ).sha
        path = Path(snapshot_download(repo, repo_type=typ, revision=rev,
            allow_patterns=patterns, local_dir=ROOT / key))
        assets[key] = {'repo': repo, 'revision': rev,
            'files': {str(p.relative_to(path)): file_hash(p)
                      for p in sorted(path.rglob('*')) if p.is_file() and '.cache' not in p.parts}}
        print('ASSET_READY=' + json.dumps({'key': key, 'revision': rev}), flush=True)
    items = []
    for bench in ('math500', 'gsm8k'):
        if bench == 'math500':
            rows = [json.loads(line) for line in (ROOT/bench/'test.jsonl').read_text().splitlines() if line]
        else:
            files = sorted((ROOT / bench).rglob('*test*.parquet'))
            if not files: raise ValueError('No test parquet: ' + bench)
            rows = [r for p in files for r in pq.read_table(p).to_pylist()]
        expected = 500 if bench == 'math500' else 1319
        assert len(rows) == expected, (bench, len(rows))
        idx = sorted(random.Random(20261011).sample(range(len(rows)), N))
        for i in idx:
            row = rows[i]
            items.append({'benchmark': bench, 'index': i,
                'id': row.get('unique_id', str(i)),
                'prompt': row['problem'] if bench == 'math500' else row['question'],
                'gold': row['answer'], 'level': row.get('level'), 'subject': row.get('subject')})
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT / 'qwen', local_files_only=True)
    a = tok.apply_chat_template([{'role': 'user', 'content': '2+2?'}], tokenize=False,
                               add_generation_prompt=True, enable_thinking=False)
    b = tok.apply_chat_template([{'role': 'user', 'content': '2+2?'}], tokenize=False,
                               add_generation_prompt=True, enable_thinking=True)
    assert a != b and '</think>' in a
    manifest = {'assets': assets, 'sample_seed': 20261011, 'items': items,
                'sampling': {'temperature': .6, 'top_p': .95, 'top_k': 20,
                             'max_new_tokens': CAP, 'seed': SEED},
                'scorer': 'pinned repository author-code; thinking removed strictly'}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    write('manifest.json', manifest)
    print('PREPARE_PASS=256 paired math items; Qwen thinking template switch verified', flush=True)


def final_text(text, thinking):
    if '</think>' in text: return text.rsplit('</think>', 1)[1], False
    if thinking or '<think>' in text: return '', True
    return text, False


def evaluate(arm):
    import torch
    from transformers import AutoTokenizer
    import sglang
    h = audited_helpers()
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    path = ROOT / 'qwen'
    thinking = arm == 'qwen-thinking'
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    env = {'gpu': torch.cuda.get_device_name(), 'image': os.environ['SCREEN_IMAGE'],
           'versions': {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'sglang')},
           'arm': arm, 'thinking': thinking, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(arm + '-env.json', env)
    engine = None
    start = time.monotonic()
    records = []
    try:
        engine = sglang.Engine(model_path=str(path), dtype='bfloat16', tp_size=1,
            mem_fraction_static=.70, context_length=8192, max_running_requests=64,
            random_seed=SEED, attention_backend='trtllm_mha', disable_cuda_graph=True,
            trust_remote_code=False)
        for offset in range(0, len(manifest['items']), 64):
            batch = manifest['items'][offset:offset+64]
            prompts = []
            for item in batch:
                system = h['QWEN_MATH_SYSTEM_PROMPT'] if item['benchmark'] == 'math500' else h['GSM8K_SYSTEM_PROMPT']
                prompts.append(tok.apply_chat_template([{'role': 'system', 'content': system},
                    {'role': 'user', 'content': item['prompt']}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=thinking))
            outputs = engine.generate(prompts, sampling_params={
                'temperature': .6, 'top_p': .95, 'top_k': 20, 'max_new_tokens': CAP})
            assert len(outputs) == len(batch)
            for item, prompt, output in zip(batch, prompts, outputs):
                raw = output['text']; text, unfinished = final_text(raw, thinking)
                if item['benchmark'] == 'math500':
                    pred = h['extract_boxed_answer'](text)
                    passed = h['is_math_equivalent'](pred, item['gold'])
                else:
                    pred = h['extract_number_from_answer'](text)
                    gold = h['extract_number_from_answer'](item['gold'])
                    passed = pred is not None and gold is not None and abs(pred-gold) < 1e-6
                records.append(dict(item, rendered_prompt=prompt, output=raw, final=text,
                    prediction=pred, passed=bool(passed), unfinished_thinking=unfinished,
                    meta=output['meta_info']))
            write(arm + '-responses.json', records)
            print('SCREEN_PROGRESS=' + json.dumps({'arm': arm, 'completed': len(records),
                'correct': sum(r['passed'] for r in records)}), flush=True)
    finally:
        if engine: engine.shutdown()
    summary = {'arm': arm, 'elapsed_seconds': time.monotonic()-start, 'benchmarks': {}}
    for bench in ('math500', 'gsm8k'):
        rows = [r for r in records if r['benchmark'] == bench]
        summary['benchmarks'][bench] = {'n': len(rows), 'correct': sum(r['passed'] for r in rows),
            'accuracy': sum(r['passed'] for r in rows)/len(rows),
            'unfinished_thinking': sum(r['unfinished_thinking'] for r in rows),
            'token_cap': sum(r['meta'].get('completion_tokens', 0) >= CAP for r in rows),
            'mean_tokens': sum(r['meta'].get('completion_tokens', 0) for r in rows)/len(rows)}
    write(arm + '-summary.json', summary)
    print('SCREEN_SUMMARY=' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'prepare': prepare()
    else: evaluate(sys.argv[1])
