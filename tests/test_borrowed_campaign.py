import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('borrowed',ROOT/'experiments/runai/borrowed_campaign.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)


def test_refuses_original_host_before_side_effects(tmp_path):
    with pytest.raises(ValueError,match='Wrong host'):
        b.prepare(tmp_path,tmp_path,0,'tungks-0-0')
    assert list(tmp_path.iterdir())==[]


def test_deadline_partial_is_usable_but_collapse_is_not(tmp_path):
    run=tmp_path/'soft-train/run';cp=run/'checkpoint';cp.mkdir(parents=True)
    summary=cp/'run-summary.json'
    summary.write_text(json.dumps(dict(status='stopped',optimizer_updates=120,stop_reason='deadline_checkpoint_reserve')))
    Path(str(run)+'.exitcode').write_text('0')
    assert b.trained_run(tmp_path,'soft-train')==run
    summary.write_text(json.dumps(dict(status='stopped',optimizer_updates=120,stop_reason='collapse_gate_before_update')))
    with pytest.raises(ValueError,match='Non-deadline'):
        b.trained_run(tmp_path,'soft-train')


def test_checkpoint_modified_during_hash_is_rejected(tmp_path,monkeypatch):
    path=tmp_path/'config.json';path.write_text('{}')
    def identify(p):
        path.write_text('{"changed":true}')
        return dict(path=str(p))
    monkeypatch.setattr(b,'eval_modules',lambda:(NS(checkpoint_identity=identify),None,None))
    with pytest.raises(ValueError,match='changed during'):
        b.checkpoint_stable(tmp_path)


def test_prepared_queue_preserves_budget_and_independent_branches(tmp_path,monkeypatch):
    base=tmp_path/'shared';old=base/'old';old.mkdir(parents=True)
    random=base/'random';Path(str(random)+'.log').write_text('completed_optimizer_updates: 222.000000')
    energy=base/'partition-qualify-fix-IcuIufCf/work/energy/energy-select-4.pt'
    energy.parent.mkdir(parents=True);energy.write_bytes(b'fixture')
    (old/'config.json').write_text(json.dumps(dict(student=str(base),teacher=str(base),dataset=str(base),eval_template='/unavailable')))
    monkeypatch.setattr(b,'BASE',base);monkeypatch.setattr(b,'OLD_WORK',old);monkeypatch.setattr(b,'OLD_RANDOM',random)
    monkeypatch.setattr(b.socket,'gethostname',lambda:'new-host')
    monkeypatch.setattr(b,'execute',lambda *a,**k:None)
    monkeypatch.setattr(b,'sha',lambda p:b.ENERGY_SHA)
    def command(argv,**kw):
        if argv[0]=='nvidia-smi':return '0, GPU-a, NVIDIA B200, 0\n1, GPU-b, NVIDIA B200, 0\n'
        return '' if 'status' in argv else 'a'*40
    monkeypatch.setattr(b.subprocess,'check_output',command)
    monkeypatch.setattr(b,'build_eval',lambda *a,**k:None)
    def missing(p):raise OSError('optional missing')
    monkeypatch.setattr(b,'eval_modules',lambda:(NS(read_json=missing),None,None))
    captured={}
    store=types.ModuleType('job_manager.store')
    store.initialize=lambda state,config:captured.update(config=config)
    store.connect=lambda state:NS(close=lambda:None)
    store.submit=lambda db,jobs:captured.update(jobs=jobs)
    monkeypatch.setitem(sys.modules,'job_manager.store',store)
    import tempfile
    state=tmp_path/'state';state.mkdir()
    monkeypatch.setattr(tempfile,'mkdtemp',lambda **kw:str(state))
    started=time.time()-10;work=tmp_path/'work'
    b.prepare(work,tmp_path,started,'new-host')
    jobs={j['id']:j for j in captured['jobs']}
    assert captured['config']['deadline']==started+8*3600
    assert jobs['soft-train']['dependencies']==['validate-canary']
    assert jobs['fixed-train']['dependencies']==['old-eval']
    assert jobs['fixed-train']['dependency_policy']=='terminal'
    assert jobs['soft-train']['env']['MP_TRAIN_STOP_AT']==str(started+5.5*3600)
    assert jobs['soft-train']['gpus']==['GPU-b']
    assert jobs['fixed-train']['gpus']==['GPU-a']
    assert jobs['new-eval-plan']['dependencies']==['soft-train','fixed-train']
    assert jobs['old-score']['gpus']==0
    assert jobs['old-score']['dependencies']==[]
    assert jobs['old-score']['cpu_slots']==4
    assert jobs['new-score']['dependencies']==['new-eval-plan']
    assert 'old-score' in jobs['collect']['dependencies']
    assert 'new-score' in jobs['collect']['dependencies']
    assert jobs['collect']['deadline']==started+8*3600
    assert json.loads((work/'campaign.json').read_text())['started']==started
