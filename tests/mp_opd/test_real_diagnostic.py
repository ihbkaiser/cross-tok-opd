import copy
import json
import torch
import pytest
from kdflow.algorithms._mp_opd_diagnostic import (
    AtomWeighting, diagnostic, matched_partition, partition_rates, train_weighting_step)
from experiments.mp_opd.real_oracle import validate_groups, prepare


def fixture():
    theta = torch.tensor([.2, -.15], dtype=torch.float64, requires_grad=True)
    x = torch.tensor([[1., .5], [-.7, 1.], [.4, -.9]], dtype=torch.float64)
    base = torch.tensor([.6, -.8, .3], dtype=torch.float64)
    w = torch.tensor([1., 2., 1.], dtype=torch.float64)
    nll = lambda ps: torch.nn.functional.softplus(x @ ps[0])
    select = lambda ps: ((ps[0] - torch.tensor([-.1, .3], dtype=torch.float64)) ** 2).sum()/2
    evaluate = lambda ps: ((ps[0] - torch.tensor([.1, .2], dtype=torch.float64)) ** 2).sum()/2
    return (theta,), nll, base, w, select, evaluate


def test_oracle_actual_and_eval_independence():
    p, nll, b, w, select, evaluate = fixture()
    before = p[0].detach().clone()
    out = diagnostic(p,nll,b,w,select,evaluate,lr=.01,max_span=3)
    second = diagnostic(p,nll,b,w,select,lambda ps: -evaluate(ps),lr=.01,max_span=3)
    assert out['controls']['oracle']['partition'] == second['controls']['oracle']['partition']
    assert torch.equal(before,p[0]) and p[0].grad is None
    candidates=["atomic","atomic_oracle_mix_0.25","atomic_oracle_mix_0.5","atomic_oracle_mix_0.75","oracle"]
    chosen=out['controls']['select_chosen_atomic_oracle_mix']['chosen_control']
    assert out['controls'][chosen]['select_nll']==min(out['controls'][x]['select_nll'] for x in candidates)
    assert out['controls']['atomic']['gradient_delta_from_atomic_norm']==0
    assert out['controls']['skip']['eval_improvement'] == 0
    assert out['controls']['atomic_oracle_norm']['gradient_norm'] == pytest.approx(out['controls']['oracle']['gradient_norm'],abs=1e-7)
    gradient = torch.autograd.grad(((b/w)*nll(p)).sum()/w.sum(),p)[0]
    actual = float(evaluate((p[0]-.01*gradient,)).detach())
    assert out['controls']['atomic']['eval_nll'] == pytest.approx(actual)


def test_directional_prediction_and_oracle_enumeration():
    from kdflow.algorithms._mp_opd_oracle import enumerate_partitions
    p,nll,b,w,select,evaluate = fixture()
    out = diagnostic(p,nll,b,w,select,evaluate,lr=1e-5,max_span=3)
    predictions = []
    outergrad = torch.autograd.grad(select(p),p)[0]
    for part in enumerate_partitions(3,3):
        h = (partition_rates(b,w,part)*nll(p)).sum()/w.sum()
        g = torch.autograd.grad(h,p)[0]
        predictions.append(float(1e-5*(outergrad*g).sum()))
    assert out['controls']['oracle']['predicted_select_improvement'] == pytest.approx(max(predictions),abs=1e-10)
    for value in out['controls'].values():
        assert out['before_select']-value['select_nll'] == pytest.approx(value['predicted_select_improvement'],abs=1e-9)


def test_matched_lengths_and_degeneracy():
    reference = ((0,1),(1,4),(4,6),(6,7))
    result = matched_partition(reference,42)
    assert sorted(e-s for s,e in reference) == sorted(e-s for s,e in result)
    assert result[0][0] == 0 and result[-1][1] == 7
    assert matched_partition(((0,2),(2,4)),43) == ((0,2),(2,4))
    with pytest.raises(ValueError):
        matched_partition(((0,2),(3,4)),0)


def test_weighting_hypergradient_and_no_student_mutation():
    p,nll,b,w,select,evaluate = fixture()
    net = AtomWeighting().double()
    assert torch.allclose(net(b,w), b/w)
    before = p[0].detach().clone()
    old = copy.deepcopy(net.state_dict())
    optimizer = torch.optim.SGD(net.parameters(),lr=.1)
    train_weighting_step(net,optimizer,p,nll,b,w,select,.1)
    assert torch.equal(before,p[0]) and p[0].grad is None
    assert any(not torch.equal(v,old[k]) for k,v in net.state_dict().items())
    # Compare the exact one-step hypergradient against finite difference.
    from kdflow.algorithms._mp_opd_diagnostic import gradients
    def objective():
        g = gradients((net(b,w)*nll(p)).sum()/w.sum(),p,create_graph=True)
        return select(tuple(x-.1*y for x,y in zip(p,g)))
    target = net.net[-1].weight
    analytic = torch.autograd.grad(objective(),target)[0][0,0].item()
    saved = target.detach().clone()
    values=[]
    for delta in (1e-5,-1e-5):
        with torch.no_grad():
            target.copy_(saved); target[0,0] += delta
        values.append(float(objective().detach()))
    with torch.no_grad(): target.copy_(saved)
    assert analytic == pytest.approx((values[0]-values[1])/2e-5,abs=1e-8)


def test_split_overlap_content_not_only_id():
    row = lambda i: {'id':str(i),'messages':[{'role':'user','content':str(i)}],'reference':'answer'}
    group = {'rollout':row(1),'select':row(2),'eval':row(3)}
    assert validate_groups([group])['rows']==3
    group['eval']['messages']=group['select']['messages']
    with pytest.raises(ValueError,match='overlapping'):
        validate_groups([group])


def test_prepare_deterministic_and_no_overwrite(tmp_path):
    from argparse import Namespace
    path=tmp_path/'source.jsonl'
    path.write_text('\n'.join(json.dumps({'messages':[{'role':'user','content':str(i)}],'answer':'ref'}) for i in range(9)))
    args=Namespace(input=path,output=tmp_path/'one.json',messages_key='messages',reference_key='answer',groups=3,seed=42)
    prepare(args)
    first=json.loads(args.output.read_text())
    assert all('reference' not in g['rollout'] for g in first['groups'])
    args.output=tmp_path/'two.json'; prepare(args)
    assert json.loads(args.output.read_text())==first
    with pytest.raises(FileExistsError): prepare(args)


@pytest.mark.parametrize("counts",[None,[1,4,8]])
def test_complete_runner_with_tiny_hf_transport(tmp_path, monkeypatch,counts):
    import sys
    import types
    from argparse import Namespace
    from experiments.mp_opd.real_oracle import run
    class Tokenizer:
        eos_token_id=7
        all_special_ids=[7]
        def apply_chat_template(self, messages, **kwargs): return [0,3]
        def encode(self,text,**kwargs): return [1,2]
        def decode(self,ids,**kwargs): return ''.join({1:'a',2:'b',7:'<eos>'}[i] for i in ids)
        def get_added_vocab(self): return {'<eos>':7}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed=torch.nn.Embedding(8,4)
            self.block=torch.nn.Module()
            self.block.q_proj=torch.nn.Linear(4,4)
            self.head=torch.nn.Linear(4,8)
        def forward(self,input_ids,**kwargs):
            h=self.embed(input_ids).cumsum(1)
            return types.SimpleNamespace(logits=self.head(torch.tanh(self.block.q_proj(h))))
        def generate(self,input_ids,**kwargs):
            return torch.cat([input_ids,torch.tensor([[1,2,7]])],dim=1)
    models=[]
    def load(*args,**kwargs):
        m=Model(); models.append(m); return m
    fake=types.ModuleType('transformers')
    fake.AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=load)
    fake.AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer())
    monkeypatch.setitem(sys.modules,'transformers',fake)
    row=lambda i: {'id':str(i),'messages':[{'role':'user','content':str(i)}],'reference':'ab'}
    groups=[dict(zip(('rollout','select','eval'),[row(i+j) for j in range(3)])) for i in (0,3)]
    if counts:
        groups=[{'rollout':row(i),'select':[row(i+j) for j in range(1,9)],'eval':[row(i+j) for j in range(9,13)]} for i in (0,13)]
    data=tmp_path/'data.json';data.write_text(json.dumps({'schema':'mp-oracle-data-v2' if counts else 'mp-oracle-data-v1','groups':groups}))
    for name in ('student','teacher'):
        (tmp_path/name).mkdir();(tmp_path/name/'config.json').write_text('{}')
    args=Namespace(student=tmp_path/'student',teacher=tmp_path/'teacher',data=data,output=tmp_path/'out',
                   adapter_module='block.q_proj',device='cpu',rank=2,seed=42,max_span=3,virtual_lr=.1,
                   weighting_lr=.01,weighting_steps=1,temperature=.6,top_p=.95,max_new_tokens=8,
                   max_prompt_tokens=16,max_reference_tokens=16)
    args.select_counts=counts
    args.learn_partition=True;args.energy_steps=2;args.energy_lr=.001;args.energy_temperature=1.;args.norm_controls=True
    run(args)
    assert (args.output / "energy-select-1.pt").is_file()
    saved=torch.load(args.output / "energy-select-1.pt",weights_only=False)
    assert saved["step"]>0 and saved["config"]["feature_dim"]==10
    report=json.loads((args.output/'summary.json').read_text())
    assert report['valid_groups']==2
    results=[json.loads(x) for x in (args.output/'results.jsonl').read_text().splitlines()]
    assert [x['weighting_prior_groups'] for x in results]==[0,1]
    assert all(x['parameters_unchanged'] for x in results)
    if counts:
        for item in results:
            sweep=item['select_count_sweep']
            assert [len(sweep[str(n)]['select_ids']) for n in counts]==counts
            assert all(sweep[str(n)]['eval_ids']==sweep['1']['eval_ids'] for n in counts)
            assert all(sweep[str(n)]['controls']['atomic']['eval_nll']==sweep['1']['controls']['atomic']['eval_nll'] for n in counts)
        paired=json.loads((args.output/'paired-select-counts.json').read_text())
        assert paired['independent_groups']==2 and len(paired['paired']['8'])==2
        assert report['reported_select_count']==8

    assert torch.count_nonzero(models[0].block.q_proj.b)==0
    assert (args.output/'weighting.pt').is_file()
    with pytest.raises(FileExistsError): run(args)


def test_prepare_excludes_sft_and_previous_groups(tmp_path):
    from argparse import Namespace
    path=tmp_path/'source.jsonl'
    rows=[{'messages':[{'role':'user','content':str(i)}],'reference':'ref'} for i in range(12)]
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    seen=tmp_path/'sft.jsonl'; seen.write_text(json.dumps(rows[0]))
    args=Namespace(input=path,output=tmp_path/'one.json',messages_key='messages',reference_key='reference',groups=1,seed=42,exclude_prompts=[seen])
    prepare(args)
    first=json.loads(args.output.read_text())
    assert first['excluded_prompt_count']==1
    from experiments.mp_opd.real_oracle import identity
    first_ids={identity(row['messages']) for row in first['groups'][0].values()}
    assert identity(rows[0]['messages']) not in first_ids
    args.exclude_prompts.append(args.output)
    args.output=tmp_path/'two.json'; prepare(args)
    second=json.loads(args.output.read_text())
    second_ids={identity(row['messages']) for row in second['groups'][0].values()}
    assert not first_ids & second_ids
    assert second['excluded_prompt_count']==4


def test_sft_conversation_reference_is_not_in_prompt(tmp_path):
    from argparse import Namespace
    path=tmp_path/'sft.jsonl'
    rows=[{'messages':[{'role':'user','content':str(i)},{'role':'assistant','content':'answer'}]} for i in range(6)]
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    excluded=tmp_path/'used.jsonl';excluded.write_text(json.dumps(rows[0]))
    args=Namespace(input=path,output=tmp_path/'out.json',messages_key='messages',reference_key='@last-assistant',groups=1,seed=42,exclude_prompts=[excluded])
    prepare(args)
    result=json.loads(args.output.read_text())
    for row in result['groups'][0].values():
        assert len(row['messages'])==1
        assert row['messages'][0]['content']!='0'
    assert result['groups'][0]['select']['reference']=='answer'


def test_conflict_excludes_whole_prompt_order_independent(tmp_path):
    from argparse import Namespace
    rows=[{'messages':[{'role':'user','content':str(i)}],'reference':'ref'} for i in range(8)]
    rows += [{'messages':rows[0]['messages'],'reference':'alternative'},rows[0]]
    source=tmp_path/'source.jsonl'
    args=Namespace(input=source,output=tmp_path/'default.json',messages_key='messages',reference_key='reference',groups=2,seed=42)
    source.write_text('\n'.join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError,match='conflicting references'): prepare(args)
    args.conflicting_references='exclude'; prepare(args)
    result=json.loads(args.output.read_text())
    assert result['conflicting_prompt_count']==1
    assert all(row['messages'][0]['content']!='0' for group in result['groups'] for row in group.values())
    source.write_text('\n'.join(json.dumps(row) for row in reversed(rows)))
    args.output=tmp_path/'reversed.json';prepare(args)
    reverse=json.loads(args.output.read_text())
    assert reverse['groups']==result['groups']
    assert reverse['conflicting_prompt_ids']==result['conflicting_prompt_ids']


def test_tokenizer_batch_encoding_normalized_before_tensor():
    from collections import UserDict
    from experiments.mp_opd.real_oracle import token_ids, chat_prompt_ids
    class Encoding:
        ids = [2, 7, 9]
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] is False
            return UserDict(input_ids=[2, 7, 9], attention_mask=[1, 1, 1])
    assert token_ids(Encoding()) == [2, 7, 9]
    assert token_ids([], allow_empty=True) == []
    assert chat_prompt_ids(Tokenizer(), [], 3) == [2, 7, 9]
    assert torch.tensor([chat_prompt_ids(Tokenizer(), [], 3)], dtype=torch.long).shape == (1, 3)
    with pytest.raises(ValueError): chat_prompt_ids(Tokenizer(), [], 2)
    for bad in ([], [[2, 7]], [Encoding()], [True], [-1]):
        with pytest.raises(ValueError): token_ids(bad)


def test_multireference_prepare_overlap_and_mean_gradient(tmp_path):
    from argparse import Namespace
    from experiments.mp_opd.real_oracle import reference_mean,select_counts
    rows=[{'messages':[{'role':'user','content':str(i)}],'answer':'ref'} for i in range(30)]
    source=tmp_path/'rows.json';source.write_text(json.dumps(rows))
    args=Namespace(input=source,output=tmp_path/'prepared.json',messages_key='messages',reference_key='answer',groups=2,seed=42,select_references=8,eval_references=4)
    prepare(args);data=json.loads(args.output.read_text())
    assert data['schema']=='mp-oracle-data-v2' and data['split_audit']['rows']==26
    assert select_counts(data['groups'],[1,4,8])==[1,4,8]
    with pytest.raises(ValueError): select_counts(data['groups'],[1,9])
    data['groups'][0]['eval'][0]=data['groups'][0]['select'][0]
    with pytest.raises(ValueError,match='overlapping'): validate_groups(data['groups'])
    x=torch.tensor(2.,requires_grad=True)
    f=reference_mean([lambda ps:ps[0]**2,lambda ps:3*ps[0]**2])
    assert torch.autograd.grad(f((x,)),x)[0].item()==8.


def test_learned_partition_norm_controls_and_energy_learning():
    from kdflow.algorithms._mp_opd_energy import MPAtomEnergy, energy_surrogate_loss
    p,nll,b,w,select,evaluate=fixture()
    net=MPAtomEnergy(10,hidden_dim=4,layers=1).eval()
    features=torch.randn(3,10)
    weighting=AtomWeighting().double()
    original=p[0].detach().clone()
    payload={}
    out=diagnostic(p,nll,b,w,select,evaluate,lr=.1,max_span=2,weighting=weighting,
                   energy=net,energy_features=features,training_payload=payload,norm_controls=True)
    controls=out['controls']
    for name in ('learned_partition','learned_weighting','fixed2','random_oracle_lengths'):
        c=controls[name+'_atomic_norm']
        assert c['norm_match_valid']
        assert c['virtual_update_norm']==pytest.approx(controls['atomic']['virtual_update_norm'],rel=1e-5)
    assert out['learned_partition']['coverage_max_error']<1e-5
    before={k:v.clone() for k,v in net.state_dict().items()}
    opt=torch.optim.AdamW(net.parameters(),lr=.01)
    loss,_=energy_surrogate_loss(net,features,payload['utilities'],max_span_length=2,
                                temperature=1.,virtual_learning_rate=.1,valid_mask=payload['valid'])
    opt.zero_grad();loss.backward();opt.step()
    assert any(not torch.equal(v,before[k]) for k,v in net.state_dict().items())
    assert torch.equal(p[0],original) and p[0].grad is None
    payload2={}
    diagnostic(p,nll,b,w,select,lambda ps:-evaluate(ps),lr=.1,max_span=2,
               energy=net,energy_features=features,training_payload=payload2)
    assert torch.equal(payload['utilities'],payload2['utilities'])
