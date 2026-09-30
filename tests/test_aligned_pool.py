import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('aligned',ROOT/'experiments/runai/aligned_eval_pool.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
Q,E=M.Q,M.E


def fixture(tmp_path):
    data=tmp_path/'data.json'
    item=dict(id='0',messages=[{'role':'user','content':'sum'}],gold='#### 2')
    E.write_new(data,dict(items=[item]))
    plan=dict(profile=M.D.PROFILE,seeds=[42,43,44],source=M.D.script_hashes(),
              protocol={'scope':'old','decoding':'fixed'},
              data={'gsm8k':dict(path=str(data),sha256=E.file_hash(data),count=1)})
    job=dict(id='random-seed43-200',checkpoint={'sha256':'cp'})
    args=argparse.Namespace(phase='generate',min_free_gib=0,concurrency=1)
    def gen(base,item,benchmark,seed,deadline):
        return dict(id=item['id'],seed=seed,request_sha256=E.digest(E.encoded(E.generation_payload('eval-gemma',item,benchmark,seed))),
            response={'choices':[{'finish_reason':'stop','message':{'content':'#### 2'}}]})
    old=tmp_path/'old'
    with patch.object(Q,'generate_one',side_effect=gen):
        Q.run_cell(old,plan,'oldhash',job,'gsm8k',42,'',{},args,time.time()+10)
    return plan,job,args,old/'cells'/job['id']/'gsm8k/42'


def test_migrate_response_spool_preserves_content_and_rebinds_contract(tmp_path):
    plan,job,args,src=fixture(tmp_path)
    before=(src/'responses.jsonl').read_bytes()
    dst=tmp_path/'new'
    result=M.migrate_cell(src,dst,plan,'oldhash',job,'newhash',job,'gsm8k',42)
    assert result==dict(responses=1,scores=0,metrics=0)
    assert (src/'responses.jsonl').read_bytes()==before==(dst/'responses.jsonl').read_bytes()
    assert E.read_json(src/'contract.json')['plan_sha256']=='oldhash'
    assert E.read_json(dst/'generation-complete.json')['contract']['plan_sha256']=='newhash'


def test_generation_does_not_read_live_score_journal(tmp_path):
    plan,job,args,cell=fixture(tmp_path)
    with patch.object(Q,'journal',side_effect=AssertionError('must not touch score journal')):
        Q.run_cell(tmp_path/'old',plan,'oldhash',job,'gsm8k',42,'',{},args,time.time()+10)


def test_eval_config_drift_is_rejected(tmp_path):
    plan,*_=fixture(tmp_path)
    other=copy.deepcopy(plan);other['protocol']['scope']='new scope'
    M.compatible(plan,other)
    other['seeds']=[42]
    with pytest.raises(ValueError,match='seeds'):M.compatible(plan,other)
    other=copy.deepcopy(plan);other['protocol']['decoding']='changed'
    with pytest.raises(ValueError,match='protocol'):M.compatible(plan,other)
    other=copy.deepcopy(plan);other['source']['contract_eval.py']='changed'
    with pytest.raises(ValueError,match='source'):M.compatible(plan,other)


def test_corrupt_spool_not_migrated(tmp_path):
    plan,job,_,src=fixture(tmp_path)
    with (src/'responses.jsonl').open('a') as f:f.write('bad\n')
    with pytest.raises(ValueError):
        M.migrate_cell(src,tmp_path/'new',plan,'oldhash',job,'newhash',job,'gsm8k',42)
