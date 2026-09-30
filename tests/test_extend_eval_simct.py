import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest
import shutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/evaluation'))
import eval_queue as Q
E,D=Q.E,Q.D
SCRIPT=ROOT/'experiments/runai/transfer/extend-eval-simct.py'


@pytest.mark.parametrize("update",[False,"d574482","5199cb0","52a8428","7ced1f1","45f6425"])
def test_migration_preserves_results_and_clock(tmp_path,update):
    old=tmp_path/'old';old.mkdir();out=tmp_path/'new';simct=tmp_path/'simct';simct.mkdir()
    E.write_new(simct/'run-summary.json',dict(kd_algorithm='span_ctkd',status='completed',optimizer_updates=312,student='/sft'))
    for step in (40,80,120,156,200,240,280,312):
        ck=simct/f'step{step}';ck.mkdir()
        for name in ('model.safetensors','config.json','tokenizer.json','tokenizer_config.json'): (ck/name).write_text('{}')
    jobs=[dict(id='sft-0',mode='sft',step=0,tier=0,checkpoint={'path':'/sft','sha256':'sft'})]
    for step in (312,156,80,240,40,200,120,280):
        for mode,name in D.RUNS.items(): jobs.append(dict(id=f'{mode}-{step}',mode=mode,step=step,tier=0,checkpoint={'path':f'/runs/{name}/checkpoint/step{step}','sha256':mode+str(step)}))
    data={};item={'id':'one','messages':[{'role':'user','content':'1+1?'}]}
    for b in E.CAPS:
        path=tmp_path/(b+'.json');E.write_new(path,{'items':[item]});data[b]={'path':str(path),'sha256':E.file_hash(path),'count':1}
    source=ROOT
    hashes=D.script_hashes()
    if update:
        source=tmp_path/'old-source'
        shutil.copytree(ROOT/'scripts',source/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
        (source/'scripts/evaluation/eval_queue.py').write_bytes(subprocess.check_output(['git','show',str(update)+':scripts/evaluation/eval_queue.py'],cwd=ROOT))
        hashes={name:E.file_hash(source/'scripts/evaluation'/name) for name in hashes}
        for step in (40,80,120,156,200,240,280,312): jobs.append(dict(id=f'simct-{step}',mode='simct',step=step,tier=0,checkpoint=E.checkpoint_identity(simct/f'step{step}')))
    plan=dict(profile=D.PROFILE,source=hashes,jobs=jobs,data=data,seeds=[42,43,44])
    E.write_new(old/'plan.json',plan);ph=E.file_hash(old/'plan.json')
    import time
    state=dict(plan_sha256=ph,started=1,deadline=time.time()+3600,admit_until=time.time()+1800,jobs={'sft-0':{'status':'running'}},durations=[])
    E.write_new(old/'state.json',state)
    cell=old/'cells/sft-0/gsm8k/42';cell.mkdir(parents=True)
    contract=Q.cell_contract(ph,jobs[0],data,'gsm8k',42,{})
    E.write_new(cell/'contract.json',contract)
    response=dict(id='one',seed=42,request_sha256=E.digest(E.encoded(E.generation_payload('eval-gemma',item,'gsm8k',42))),response={'choices':[{'finish_reason':'stop','message':{'content':'2'}}]})
    Q.append(cell/'responses.jsonl',response)
    Q.append(cell/'scores.jsonl',dict(id='one',passed=True,response_sha256=E.digest(E.encoded(response))))
    metrics=dict(contract=contract,status='completed',count=1,score=1.,responses_sha256=E.file_hash(cell/'responses.jsonl'),scores_sha256=E.file_hash(cell/'scores.jsonl'))
    E.write_new(cell/'metrics.json',metrics)
    if update in ('7ced1f1','45f6425'):
        E.write_new(cell/'generation-complete.json',dict(contract=contract,count=1,responses_sha256=E.file_hash(cell/'responses.jsonl')))
        E.write_new(old/'generation-state.json',dict(state))
    before={p.name:p.read_bytes() for p in cell.iterdir()}
    cmd=[sys.executable,str(SCRIPT),'--source',str(source),'--plan',str(old/'plan.json'),'--out',str(out),'--simct',str(simct),'--worker-pids','999999999']
    if update: cmd+=['--queue-update',str(ROOT/'scripts/evaluation/eval_queue.py')]
    subprocess.run(cmd,check=True)
    if update in ('7ced1f1','45f6425'):
        assert E.read_json(out/'cells'/cell.relative_to(old/'cells')/'generation-complete.json')['contract']['plan_sha256']==E.file_hash(out/'plan.json')
    if update:
        assert E.read_json(out/'plan.json')['source']==D.script_hashes()
        assert E.file_hash(source/'scripts/evaluation/eval_queue.py')==hashes['eval_queue.py']
        assert E.read_json(out/'migration.json')['score_workers']==16
        assert E.read_json(out/'migration.json')['concurrency_by_gpu']=={'0':256,'1':128}
    assert len(E.read_json(out/'plan.json')['jobs'])==25
    assert E.read_json(out/'state.json')['deadline']==state['deadline']
    assert E.read_json(out/'migration.json')['totals']=={'responses':1,'scores':1,'metrics':1}
    assert all((cell/n).read_bytes()==v for n,v in before.items())
    assert (out/'cells/sft-0/gsm8k/42/responses.jsonl').read_bytes()==before['responses.jsonl']
    assert E.read_json(out/'cells/sft-0/gsm8k/42/metrics.json')['contract']['plan_sha256']==E.file_hash(out/'plan.json')


def test_refuses_unrelated_pid():
    spec=importlib.util.spec_from_file_location('extend',SCRIPT);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    import os,pytest
    with pytest.raises(ValueError,match='PID identity mismatch'):
        m.stop_workers(Path('/wrong-plan'),Path('/wrong-queue'),[os.getpid()])
