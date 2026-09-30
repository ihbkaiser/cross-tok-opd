import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('audit',Path(__file__).parents[1]/'scripts/evaluation/audit_campaign_export.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_training_and_eval_seed_variance_are_separate():
    runs={str(s):{'group':'soft','primary':True,'train_seed':s} for s in (42,43,44)}
    cells=[{'run':str(s),'step':40,'benchmark':'gsm8k','eval_seed':e,'score':v,'valid':True}
           for s,vals in [(42,[.1,.2,.3]),(43,[.3,.4,.5]),(44,[.5,.6,.7])]
           for e,v in zip((42,43,44),vals)]
    points=m.aggregate(cells)
    for n in (1,2,3):
        g=m.grouped(points[:n],runs)[0]
        assert g['train_n']==n
        if n==1: assert g['train_std'] is None
        else: assert g['train_std']>0
    assert m.grouped(points,runs)[0]['mean']==pytest.approx(.4)
    assert m.grouped(points,runs)[0]['train_std']==pytest.approx(.2)

def test_partial_eval_never_fabricates_mean_or_zero():
    cells=[{'run':'r','step':40,'benchmark':'b','eval_seed':42,'score':.6,'valid':True}]
    p=m.aggregate(cells)[0]
    assert p['mean'] is None and p['eval_n']==1
    assert m.grouped([p],{'r':{'primary':True,'train_seed':42,'group':'g'}})==[]

def test_duplicate_training_seed_rejected():
    points=[{'run':r,'step':40,'benchmark':'b','mean':.5} for r in ('a','b')]
    runs={r:{'group':'g','primary':True,'train_seed':42} for r in ('a','b')}
    with pytest.raises(ValueError):m.grouped(points,runs)

def test_training_parser_flags_nonfinite_and_conflicting_duplicate():
    prefix='[on_policy_kd_trainer.py:logging:757] step [2/312], '
    rows,issues=m.training_rows(prefix+'loss: 0.1, grad_norm: nan\n'+prefix+'loss: 0.2, grad_norm: 1.0')
    assert rows==[{'step':2,'loss':.2,'grad_norm':1.0}]
    assert issues==[{'step':2,'nonfinite':['grad_norm']},{'step':2,'conflicting_duplicate':True}]


def test_single_checkpoint_retains_all_eval_seeds():
    cells=[{'run':'r','step':50,'benchmark':'gsm8k','eval_seed':s,'score':v,'valid':True}
           for s,v in zip((42,43,44),(.1,.2,.3))]
    point=m.aggregate(cells)[0]
    assert point['eval_scores']=={42:.1,43:.2,44:.3}
    assert point['eval_n']==3 and point['eval_std']==pytest.approx(.1)
