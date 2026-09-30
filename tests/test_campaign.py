import importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location('campaign',Path(__file__).resolve().parents[1]/'experiments/runai/campaign.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
def test_failed_dependency_cannot_train():
    assert c.eligible({'after':['canary']},{'jobs':{'canary':{'status':'failed'}}})=='blocked'
def test_terminal_probe_failure_allows_explicit_fallback():
    assert c.eligible({'after':['probe'],'after_terminal':True},{'jobs':{'probe':{'status':'timeout'}}})=='ready'
def test_fallback_waits_for_all_probes():
    assert c.eligible({'after':['a','b'],'after_terminal':True},{'jobs':{'a':{'status':'completed'},'b':{'status':'running'}}})=='waiting'
def test_default_dependency_waits():
    assert c.eligible({'after':['a']},{'jobs':{}})=='waiting'

def test_fallback_never_bypasses_failed_canary():
    job={'after':['probe'],'after_terminal':True,'require_success':['canary']}
    assert c.eligible(job,{'jobs':{'probe':{'status':'failed'},'canary':{'status':'failed'}}})=='blocked'
def test_real_cpu_job_and_resume(tmp_path):
    import json, subprocess, sys
    root=Path(__file__).resolve().parents[1]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    plan={'hours':.01,'source':str(root),'source_commit':commit,'jobs':[{'id':'one','budget_seconds':10,'argv':[sys.executable,'-c',"print('done')"]}]}
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    c.run(path)
    first=json.loads((tmp_path/'campaign-state.json').read_text())
    assert first['jobs']['one']['status']=='completed'
    c.run(path)
    second=json.loads((tmp_path/'campaign-state.json').read_text())
    assert first['started']==second['started']
    assert (tmp_path/'one.log').read_text().count('done')==1
