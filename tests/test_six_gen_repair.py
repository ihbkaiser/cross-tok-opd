import argparse
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('repair',ROOT/'experiments/runai/repair_six_gen.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
sys.path.insert(0,str(ROOT/'scripts/evaluation'))
import eval_queue as Q
import contract_eval as E


def test_missing_minfree_reproduces_and_shim_reaches_real_generation(tmp_path):
    data=tmp_path/'data.json'
    item=dict(id='0',messages=[{'role':'user','content':'sum'}],gold='#### 2')
    E.write_new(data,dict(items=[item]))
    plan=dict(data={'gsm8k':dict(path=str(data),sha256=E.file_hash(data),count=1)})
    job=dict(id='job',checkpoint={'sha256':'cp'})
    args=argparse.Namespace(phase='generate',concurrency=256)
    def run(root,plan,ph,job,args,end):
        return Q.run_cell(root,plan,ph,job,'gsm8k',42,'',{},args,end)
    def generate(base,item,benchmark,seed,deadline):
        return dict(id=item['id'],seed=seed,
            request_sha256=E.digest(E.encoded(E.generation_payload('eval-gemma',item,benchmark,seed))),
            response={'choices':[{'finish_reason':'stop','message':{'content':'#### 2'}}]})
    with patch.object(Q,'generate_one',side_effect=generate), \
         patch('shutil.disk_usage',return_value=SimpleNamespace(free=30*1024**3)), \
         patch.object(Q,'run_checkpoint',side_effect=run):
        with pytest.raises(AttributeError,match='min_free_gib'):
            Q.run_checkpoint(tmp_path/'old',plan,'hash',job,args,time.time()+30)
        R.install_adapter(Q)
        Q.run_checkpoint(tmp_path/'fixed',plan,'hash',job,args,time.time()+30)
        marker=E.read_json(tmp_path/'fixed/cells/job/gsm8k/42/generation-complete.json')
        assert marker['count']==1 and args.min_free_gib==20


def test_recovery_only_acknowledges_this_adapter_bug():
    assert R.is_adapter_error({'error':"'Namespace' object has no attribute 'min_free_gib'"})
    assert not R.is_adapter_error({'error':'SGLang exited'})
    assert not R.is_adapter_error({'error':'checkpoint changed after plan'})
