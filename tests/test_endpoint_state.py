import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/evaluation'))
import eval_queue as Q


def test_initialized_state_supports_separate_generation(tmp_path,monkeypatch):
    monkeypatch.setattr(Q,'GENERATION_ONLY',False)
    plan={'hours':1,'admit_hours':1}
    state=Q.read_state(tmp_path,'hash',plan)
    state['qualification']=[{'passed':True}]
    Q.atomic_json(tmp_path/'state.json',state)
    monkeypatch.setattr(Q,'GENERATION_ONLY',True)
    generated=Q.read_state(tmp_path,'hash',plan)
    assert generated['deadline']==state['deadline']
    assert generated['qualification']==state['qualification']
    assert generated['jobs']=={}
    assert (tmp_path/'generation-state.json').exists()
