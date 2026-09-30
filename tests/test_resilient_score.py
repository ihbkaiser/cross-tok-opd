import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/runai'))
spec=importlib.util.spec_from_file_location('resilient',ROOT/'experiments/runai/resilient_score.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

def error(code):return 'item_id=x '+json.dumps(dict(error='scorer runtime error',returncode=code))

def test_persistent_budget_survives_restart(tmp_path):
    for _ in range(3):assert R.reserve(tmp_path,error(-15),write)
    assert not R.reserve(tmp_path,error(-15),write)
    assert json.loads((tmp_path/'score-recovery.json').read_text())['exhausted']

def test_crash_retries_once_and_never_retries_contract_errors(tmp_path):
    assert R.reserve(tmp_path,error(-11),write)
    assert not R.reserve(tmp_path,error(-11),write)
    assert not R.reserve(tmp_path,'Spool contract drift',write)
    assert R.signal_code(error(-9))==-9
    assert R.signal_code(error(1)) is None

def test_failed_call_retries_serial_without_changing_original_args(tmp_path,monkeypatch):
    calls=[]
    class Deadline(Exception):pass
    def original(*args):
        calls.append(args[-2].score_workers)
        if len(calls)==1:raise RuntimeError(error(-15))
        return 'done'
    Q=NS(run_cell=original,Deadline=Deadline);M=NS(B=NS(write=write))
    monkeypatch.setattr(R.time,'sleep',lambda _:None)
    R.install(M,Q);args=NS(score_workers=4)
    assert Q.run_cell(tmp_path,{},'hash',{'id':'job'},'math500',44,None,{},args,R.time.time()+1000)=='done'
    assert calls==[4,1] and args.score_workers==4

def test_deadline_propagates_without_retry(tmp_path):
    class Deadline(Exception):pass
    def original(*args):raise Deadline()
    Q=NS(run_cell=original,Deadline=Deadline);R.install(NS(B=NS(write=write)),Q)
    import pytest
    with pytest.raises(Deadline):
        Q.run_cell(tmp_path,{},'hash',{'id':'job'},'math500',44,None,{},NS(score_workers=4),R.time.time()+1000)
    assert not list(tmp_path.rglob('score-recovery.json'))
