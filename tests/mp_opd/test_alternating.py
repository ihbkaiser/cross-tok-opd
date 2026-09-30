import copy
import pytest
import torch
from kdflow.algorithms._mp_opd_alternating import alternating_step, meta_objective
from kdflow.algorithms._mp_opd_energy import MPAtomEnergy

def fixture():
    torch.manual_seed(7)
    energy=MPAtomEnergy(10,hidden_dim=3,layers=1).eval()
    theta=torch.nn.Parameter(torch.tensor([.3,-.2]))
    x=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    atom=lambda ps:(x@ps[0]-torch.tensor([1.,-.5,.7])).square()
    meta=lambda ps:(ps[0]-torch.tensor([.8,.4])).square().mean()
    args=dict(features=torch.randn(3,10),base=torch.tensor([2.,-.7,.1]),weight=torch.ones(3),lr=.1,max_span=2,temperature=1.)
    return energy,theta,atom,meta,args

def test_exact_hypergradient_matches_finite_difference_and_virtual_has_no_mutation():
    e,t,atom,meta,args=fixture();before=t.detach().clone()
    objective,_=meta_objective(e,(t,),atom,meta,**args)
    param=e.scorer[-1].weight
    grad=torch.autograd.grad(objective,param)[0]
    direction=torch.ones_like(param)
    with torch.no_grad():param.add_(direction,alpha=.001)
    plus=meta_objective(e,(t,),atom,meta,**args)[0].item()
    with torch.no_grad():param.add_(direction,alpha=-.002)
    minus=meta_objective(e,(t,),atom,meta,**args)[0].item()
    with torch.no_grad():param.add_(direction,alpha=.001)
    assert (grad*direction).sum().item()==pytest.approx((plus-minus)/.002,abs=2e-5)
    assert torch.equal(t,before) and t.grad is None

def test_alternates_energy_then_exactly_one_student_step_and_persists():
    e,t,atom,meta,args=fixture();opt=torch.optim.SGD(e.parameters(),lr=.5)
    e_ref=copy.deepcopy(e);t_ref=torch.nn.Parameter(t.detach().clone())
    opt_ref=torch.optim.SGD(e_ref.parameters(),lr=.5)
    objective,_=meta_objective(e_ref,(t_ref,),atom,meta,**args)
    for p,g in zip(e_ref.parameters(),torch.autograd.grad(objective,tuple(e_ref.parameters()))):p.grad=g
    opt_ref.step()
    expected=alternating_step(e_ref,opt_ref,(t_ref,),atom,meta,rollout_ids=['b'],meta_ids=['m'],update_energy=False,**args)
    r=alternating_step(e,opt,(t,),atom,meta,rollout_ids=['b'],meta_ids=['m'],**args)
    assert r['energy_parameter_delta']>0 and r['student_updates']==1
    assert torch.allclose(t,t_ref) and t.grad is None
    before=t.detach().clone()
    alternating_step(e,opt,(t,),atom,meta,rollout_ids=['b2'],meta_ids=['m2'],**args)
    assert not torch.equal(before,t)
    assert abs(r['credit_residual'])<1e-5

def test_frozen_control_and_id_overlap():
    e,t,atom,meta,args=fixture();opt=torch.optim.AdamW(e.parameters(),lr=.01)
    before=t.detach().clone()
    with pytest.raises(ValueError,match='disjoint'):
        alternating_step(e,opt,(t,),atom,meta,rollout_ids=['same'],meta_ids=['same'],**args)
    assert torch.equal(before,t)
    r=alternating_step(e,opt,(t,),atom,meta,rollout_ids=['b'],meta_ids=['m'],update_energy=False,**args)
    assert r['energy_parameter_delta']==0 and not torch.equal(before,t)
@pytest.mark.parametrize("frozen", [False, True])
def test_runner_persists_student_across_fresh_rollouts(tmp_path, monkeypatch, frozen):
    import json, sys
    from types import SimpleNamespace
    from experiments.mp_opd.real_oracle import run, digest
    from kdflow.algorithms._mp_opd_energy import save_energy_checkpoint
    class Tokenizer:
        eos_token_id=3
        def apply_chat_template(self,messages,**kw):return [int(messages[0]["content"][0])]
        def encode(self,text,**kw):return [1,2]
        def decode(self,ids,**kw):return ''.join({0:'p',1:'a',2:'b',3:'!'}[int(i)] for i in ids)
        def get_added_vocab(self):return {}
    seen=[]
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.block=torch.nn.Module();self.block.proj=torch.nn.Linear(4,4)
        def forward(self,input_ids,**kw):
            return SimpleNamespace(logits=self.block.proj(torch.nn.functional.one_hot(input_ids,4).float()))
        def generate(self,input_ids,**kw):
            seen.append(self.block.proj.b.detach().clone())
            import random, numpy as np
            torch.rand(2); random.random(); np.random.rand()
            response=[3] if int(input_ids[0,0])==1 else [1,2,3]
            return torch.cat([input_ids,torch.tensor([response])],dim=1)
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a,**k:Model()),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer())))
    groups=[]
    for i in range(3):
        groups.append({role:{'id':f'{i}-{role}','messages':[{'role':'user','content':f'{i}-{role}'}],
                            'reference':'ab'} for role in ('rollout','select','eval')})
    data=tmp_path/'data.json';data.write_text(json.dumps({'schema':'mp-oracle-data-v1','groups':groups}))
    model=tmp_path/'model';model.mkdir()
    energy=MPAtomEnergy(10,32,2);opt=torch.optim.AdamW(energy.parameters())
    ck=tmp_path/'energy.pt';save_energy_checkpoint(ck,energy,opt,step=1,extra_config={'max_span_length':2})
    args=SimpleNamespace(data=data,select_counts=[1],alternating_student=True,learn_partition=True,
        energy_checkpoint=ck,energy_sha256=digest(ck),seed=42,output=tmp_path/'out',device='cpu',model_dtype='float32',
        student=model,teacher=model,max_prompt_tokens=16,max_reference_tokens=16,adapter_module='block.proj',rank=2,
        weighting_lr=.01,energy_lr=.01,max_span=2,energy_temperature=1.,virtual_lr=.1,freeze_energy=frozen,
        temperature=.6,top_p=.95,max_new_tokens=5)
    run(args)
    summary=json.loads((args.output/'summary.json').read_text())
    assert summary['schema']=='mp-alternating-resume-v2'
    assert summary['status']=='completed' and summary['next_group']==3
    assert summary['student_updates']==2 and summary['energy_updates']==(0 if frozen else 2)
    assert len(seen)==3 and not torch.equal(seen[0],seen[1])
    saved=torch.load(args.output/'latest.pt',weights_only=False)
    assert saved['step']==2 and not torch.equal(saved['b'],seen[-1])
    assert len((args.output/'results.jsonl').read_text().splitlines())==3
    assert saved['cursor']==3 and saved['invalid']==1
    baseline_results=json.loads(json.dumps(saved['results']))
    baseline_traces=saved['traces']
    args.output=tmp_path/'paused';args.stop_after_groups=2
    run(args)
    paused=torch.load(args.output/'latest.pt',weights_only=False)
    assert paused['cursor']==2 and paused['step']==1 and paused['invalid']==1
    # Simulate a crash/torn projection after the checkpoint was committed.
    (args.output/'results.jsonl').write_text('corrupt tail')
    (args.output/'trajectories.jsonl').unlink()
    args.resume=True;args.stop_after_groups=None
    run(args)
    assert json.loads((args.output/'summary.json').read_text())['next_group']==3
    resumed_summary=json.loads((args.output/'summary.json').read_text())
    assert {k:v for k,v in resumed_summary.items() if k!='checkpoint'}=={k:v for k,v in summary.items() if k!='checkpoint'}
    resumed=torch.load(args.output/'latest.pt',weights_only=False)
    def same(a,b):
        import numpy as np
        if isinstance(a,torch.Tensor):assert torch.equal(a,b)
        elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:same(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):same(x,y)
        else:assert a==b
    for k in ('a','b','energy','energy_optimizer','rng','cursor','step','invalid','energy_updates'):
        same(saved[k],resumed[k])
    for rows in (baseline_results,resumed['results']):
        for row in rows:row.pop('seconds',None)
    same(baseline_results,resumed['results']);same(baseline_traces,resumed['traces'])
    assert len((args.output/'results.jsonl').read_text().splitlines())==3
    # Resume of a complete run must not take another update.
    run(args)
    complete=torch.load(args.output/'latest.pt',weights_only=False)
    same(complete['b'],saved['b'])
    from experiments.mp_opd import alternating_checkpoint as cio
    original_save=cio.save
    for phase in ('before_commit','after_commit'):
        args.output=tmp_path/phase;args.resume=False
        def interrupted(*a,**kw):
            if len(a[5])==1:
                if phase=='after_commit':original_save(*a,**kw)
                raise RuntimeError('simulated process death')
            return original_save(*a,**kw)
        monkeypatch.setattr(cio,'save',interrupted)
        with pytest.raises(RuntimeError,match='simulated process death'):run(args)
        monkeypatch.setattr(cio,'save',original_save)
        args.resume=True
        run(args)
        recovered=torch.load(args.output/'latest.pt',weights_only=False)
        for k in ('a','b','energy','energy_optimizer','rng','cursor','step','invalid','energy_updates'):
            same(saved[k],recovered[k])
        same(saved['traces'],recovered['traces'])
    args.output=tmp_path/'budget';args.resume=False;args.max_student_updates=1
    run(args)
    budgeted=torch.load(args.output/'latest.pt',weights_only=False)
    assert budgeted['cursor']==budgeted['step']==1
    assert json.loads((args.output/'summary.json').read_text())['status']=='completed'
    args.resume=True
    run(args)
    same(budgeted['b'],torch.load(args.output/'latest.pt',weights_only=False)['b'])
    args.output=tmp_path/'budget-unmet';args.resume=False;args.max_student_updates=3
    with pytest.raises(RuntimeError,match='data exhausted'):run(args)
    unmet=json.loads((args.output/'summary.json').read_text())
    assert unmet['status']=='budget_unmet' and unmet['student_updates']==2
    args.output=tmp_path/'budget-skip';args.max_student_updates=2
    run(args)
    skipped=torch.load(args.output/'latest.pt',weights_only=False)
    assert skipped['step']==2 and skipped['cursor']==3 and skipped['invalid']==1
    args.resume=True
    args.virtual_lr=.2
    with pytest.raises(ValueError,match='configuration mismatch'):run(args)


def test_exclusive_output_and_resume_requires_checkpoint(tmp_path):
    from experiments.mp_opd.alternating_checkpoint import output_lock
    output=tmp_path/'run'
    with output_lock(output,False):
        (output/'latest.pt').write_bytes(b'placeholder')
        with pytest.raises(RuntimeError,match='another pilot'):
            with output_lock(output,True):pass
    with output_lock(output,True):pass
    with pytest.raises(ValueError,match='existing latest'):
        with output_lock(tmp_path/'absent',True):pass
