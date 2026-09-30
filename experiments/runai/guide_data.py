"""Bounded source acquisition on the company host; no dataset download at import."""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def normalize(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def prompt(row):
    if row.get('messages') is not None:
        return '\n'.join(m['content'] for m in row['messages'] if m['role'] == 'user')
    for key in ('question', 'problem', 'instruction', 'input', 'description', 'question_content', 'prompt'):
        if isinstance(row.get(key), str):
            return row[key]
    raise ValueError('Unknown prompt schema')


class Dedup:
    def __init__(self):
        self.texts = []
        self.index = {}
        self.exact = set()

    def grams(self, text):
        words = text.split()
        return set(zip(*(words[i:] for i in range(5)))) if len(words) >= 5 else {(text,)}

    def add(self, text):
        text = normalize(text)
        grams = self.grams(text)
        index = len(self.texts)
        self.texts.append((text, grams))
        self.exact.add(text)
        for gram in grams:
            self.index.setdefault(gram, set()).add(index)

    def matches(self, text):
        text = normalize(text)
        if text in self.exact:
            return True
        grams = self.grams(text)
        candidates = set()
        for gram in grams:
            candidates.update(self.index.get(gram, ()))
        for index in candidates:
            other, theirs = self.texts[index]
            if min(len(text), len(other)) >= 80 and (text in other or other in text):
                return True
            if len(grams & theirs) / max(1, len(grams | theirs)) >= .8:
                return True
        return False


def rows(path):
    if Path(path).suffix == '.parquet':
        import pyarrow.parquet as pq
        yield from pq.read_table(path).to_pylist()
    else:
        with open(path) as stream:
            if Path(path).suffix == '.json':
                data = json.load(stream)
                # queue_data writes benchmark envelopes, not a bare row list.
                if isinstance(data, dict) and isinstance(data.get('items'), list) and 'benchmark' in data:
                    data = data['items']
                if not isinstance(data, list):
                    raise ValueError('Expected JSON row list or benchmark/items envelope: ' + str(path))
                yield from data
            else:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)


def acquire(work):
    from datasets import load_dataset
    cfg = json.loads((work / 'config.json').read_text())
    dedup = Dedup()
    audit = {}
    files = [cfg['dataset'], cfg['selected']] + cfg['exclude']
    template = json.loads(Path(cfg['eval_template']).read_text())
    files += [d['path'] for d in template['data'].values()]
    for path in dict.fromkeys(files):
        count = 0
        for row in rows(path):
            dedup.add(prompt(row))
            count += 1
        audit[str(path)] = {'rows': count, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        print('EXCLUSION_LOADED', path, count, flush=True)
    sources = json.loads(Path(__file__).with_name('guide-sources.json').read_text())
    selected, stats = [], {}
    for name, source in sources.items():
        target = source['target'] // 10
        accepted = scanned = 0
        urls = [f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{f}" for f in source['files']]
        kind = 'json' if source['files'][0].endswith('.jsonl') else 'parquet'
        stream = load_dataset(kind, data_files={'train': urls}, split='train', streaming=True)
        for row in stream.shuffle(seed=20260910, buffer_size=1024):
            scanned += 1
            if scanned > 100000:
                break
            text = prompt(row)
            if not 40 <= len(text) <= 3000 or dedup.matches(text):
                continue
            dedup.add(text)
            selected.append({'id': hashlib.sha256(normalize(text).encode()).hexdigest(), 'source': name,
                             'source_revision': source['revision'], 'messages': [{'role': 'user', 'content': text}]})
            accepted += 1
            if accepted == target:
                break
        stats[name] = {'target': target, 'accepted': accepted, 'scanned': scanned}
        print('GUIDE_SOURCE', name, stats[name], flush=True)
        if accepted != target:
            raise ValueError('Insufficient unseen prompts: ' + name + ' ' + str(stats[name]))
    (work / 'guide-prompts.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    (work / 'guide-acquisition.json').write_text(json.dumps({'sources': sources, 'counts': stats, 'exclusions': audit,
        'dedup': 'NFKC words, exact, containment >=80 chars, 5-gram Jaccard >=0.8; not semantic proof'}, indent=2))


def generate(work):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cfg = json.loads((work / 'config.json').read_text())
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()
    provenance = {p.name: digest(p) for p in Path(cfg['teacher']).iterdir()
                  if p.is_file() and (p.suffix in ('.safetensors', '.json'))}
    (work / 'guide-teacher.json').write_text(json.dumps(provenance, indent=2))
    tok = AutoTokenizer.from_pretrained(cfg['teacher'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(cfg['teacher'], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval()
    out = work / 'guide-responses.jsonl'
    done = {r['id'] for r in rows(out)} if out.exists() else set()
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pending = [r for r in rows(work / 'guide-prompts.jsonl') if r['id'] not in done]
    with out.open('a') as stream:
        for start in range(0, len(pending), 8):
            batch = pending[start:start + 8]
            texts = [tok.apply_chat_template(r['messages'], tokenize=False, add_generation_prompt=True) for r in batch]
            inputs = tok(texts, return_tensors='pt', padding=True).to('cuda')
            if inputs.input_ids.shape[1] > 2048:
                raise ValueError('Guide prompt exceeds 2048 tokens')
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=768, do_sample=False, pad_token_id=tok.pad_token_id)
            for row, tokens in zip(batch, outputs[:, inputs.input_ids.shape[1]:]):
                row.update(label=tok.decode(tokens, skip_special_tokens=True), terminated=tok.eos_token_id in tokens.tolist(),
                           teacher=cfg['teacher'], reference_kind='teacher pseudo-label, correctness unverified')
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
            print('GUIDE_GENERATED', min(start + 8, len(pending)), len(pending), flush=True)


def finalize(work):
    data = list(rows(work / 'guide-responses.jsonl'))
    good = [r for r in data if r['terminated'] and r['label'].strip()]
    if len(good) < 288:
        raise ValueError('Fewer than 288 complete teacher references')
    (work / 'guide.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in good))
    (work / 'guide-quality.json').write_text(json.dumps({'generated': len(data), 'accepted': len(good),
        'check': 'nonempty and EOS only; correctness unverified; source proportions after filtering may differ'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['acquire', 'generate', 'finalize'])
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.work)
