import importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location('telemetry',Path(__file__).resolve().parents[1]/'experiments/runai/eval_telemetry.py')
T=importlib.util.module_from_spec(s);s.loader.exec_module(T)

def test_incremental_counts_partial_and_replacement(tmp_path):
    p=tmp_path/'journal';p.write_bytes(b'first\npartial');c=T.Counter()
    assert c.count(p)==1
    with p.open('ab') as f: f.write(b'-end\nthird\n')
    assert c.count(p)==3
    assert c.count(p)==3
    p.write_bytes(b'new\n');assert c.count(p)==1
    p.unlink();p.write_bytes(b'a\nb\n');assert c.count(p)==2

def test_sample_reports_backlog_without_answers(tmp_path,monkeypatch):
    cell=tmp_path/'cells/atomic-312/lcb/42';cell.mkdir(parents=True)
    (cell/'responses.jsonl').write_text('PRIVATE_ANSWER\nSECOND\n')
    (cell/'scores.jsonl').write_text('score\n')
    monkeypatch.setattr(T,'gpu',lambda:[]);monkeypatch.setattr(T,'workers',lambda plan:{})
    result=T.sample(tmp_path,tmp_path/'plan.json',T.Counter())
    assert result['cells'][0]['waiting_for_score']==1
    assert 'PRIVATE_ANSWER' not in str(result)
