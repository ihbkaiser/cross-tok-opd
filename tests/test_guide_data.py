import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('guide_data', Path(__file__).resolve().parents[1] / 'experiments/runai/guide_data.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def test_dedup_normalizes_and_excludes_wrapped_test():
    d = g.Dedup()
    text = ' '.join('word' + str(i) for i in range(30))
    d.add(text)
    assert d.matches(text.upper())
    assert d.matches('Solve the following question ' + text)
    assert not d.matches('A wholly unrelated problem about trees')


def test_prompt_removes_sft_answer():
    assert g.prompt({'messages': [{'role': 'user', 'content': 'Question'}, {'role': 'assistant', 'content': 'Answer'}]}) == 'Question'


def test_source_allocation():
    import json
    sources = json.loads((Path(g.__file__).parent / 'guide-sources.json').read_text())
    assert sum(s['target'] // 10 for s in sources.values()) == 1000
    assert all(len(s['revision']) == 40 for s in sources.values())


def test_canonical_eval_envelope_excludes_benchmark_prompt(tmp_path):
    import json
    path = tmp_path / 'gsm8k.json'
    item = {'id': '1', 'messages': [{'role': 'user', 'content': 'A benchmark question'}]}
    path.write_text(json.dumps({'benchmark': 'gsm8k', 'items': [item], 'source': {}, 'profile': 'company-internal-v1'}))
    d = g.Dedup()
    for row in g.rows(path):
        d.add(g.prompt(row))
    assert d.matches('A benchmark question')


def test_malformed_envelope_is_not_silently_skipped(tmp_path):
    import json
    import pytest
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps({'benchmark': 'gsm8k', 'items': {}}))
    with pytest.raises(ValueError):
        list(g.rows(path))


def test_filter_does_not_accept_truncated_references(tmp_path):
    import json
    import pytest
    (tmp_path / 'guide-responses.jsonl').write_text(json.dumps({'terminated': False, 'label': 'unfinished'}))
    with pytest.raises(ValueError, match='288'):
        g.finalize(tmp_path)
    assert not (tmp_path / 'guide.jsonl').exists()
