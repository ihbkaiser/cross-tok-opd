import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('campaign',ROOT/'experiments/runai/campaign.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def test_external_gate_waits_then_releases(tmp_path):
    state=tmp_path/'recovery.json'
    (tmp_path/'admission.json').write_text(json.dumps({'atomic43':{'state':str(state),'job':'probe-decision'}}))
    assert not c.external_ready(tmp_path,'atomic43')
    state.write_text(json.dumps({'jobs':{'probe-decision':{'status':'running'}}}))
    assert not c.external_ready(tmp_path,'atomic43')
    assert c.external_ready(tmp_path,'fixed43')
    for status in ('completed','failed','blocked','timeout'):
        state.write_text(json.dumps({'jobs':{'probe-decision':{'status':status}}}))
        assert c.external_ready(tmp_path,'atomic43')


def test_linux_zombie_wait_status():
    import os
    import sys
    import time
    sys.path.insert(0,str(ROOT/'experiments/runai'))
    import drain_priority as d
    pid=os.fork()
    if pid==0:os._exit(7)
    try:
        for _ in range(100):
            info=d.inspect(pid)
            if info['state']=='Z':break
            time.sleep(.01)
        assert info['state']=='Z'
        assert os.waitstatus_to_exitcode(info['exit'])==7
    finally:os.waitpid(pid,0)


def test_drain_preserves_work_and_replacement_completes(tmp_path):
    import subprocess
    import sys
    import time
    work=tmp_path/'work';work.mkdir()
    recovery=tmp_path/'recovery';recovery.mkdir()
    (recovery/'plan.json').write_text('{}')
    (recovery/'campaign-state.json').write_text(json.dumps({'jobs':{'probe-decision':{'status':'completed'}}}))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    plan={'source':str(ROOT),'source_commit':commit,'hours':.02,'jobs':[
        {'id':'fixed43','budget_seconds':30,'argv':[sys.executable,'-c',"import time; time.sleep(3); print('preserved')"]},
        {'id':'atomic43','after':['fixed43'],'budget_seconds':10,'argv':[sys.executable,'-c',"print('after-guide')"]}]}
    (work/'plan.json').write_text(json.dumps(plan))
    manager=subprocess.Popen([sys.executable,str(ROOT/'experiments/runai/campaign.py'),str(work/'plan.json')])
    try:
        for _ in range(100):
            statepath=work/'campaign-state.json'
            if statepath.exists() and json.loads(statepath.read_text()).get('jobs',{}).get('fixed43',{}).get('status')=='running':break
            time.sleep(.05)
        subprocess.run([sys.executable,str(ROOT/'experiments/runai/drain_priority.py'),'--work',str(work),'--recovery',str(recovery)],check=True,timeout=30)
        manager.wait(timeout=5)
        for _ in range(200):
            state=json.loads(statepath.read_text())
            if 'ended' in state:break
            time.sleep(.05)
        assert state['jobs']['fixed43']['status']=='completed'
        assert state['jobs']['atomic43']['status']=='completed'
        assert (work/'fixed43.log').read_text().count('preserved')==1
    finally:
        if manager.poll() is None:manager.kill();manager.wait()
