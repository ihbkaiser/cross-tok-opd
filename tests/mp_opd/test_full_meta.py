import copy
import pytest
import torch
from kdflow.algorithms._mp_opd_full_meta import full_meta_step, adam_value, clipped
from kdflow.algorithms._mp_opd_full_meta import offload_adam_moments_scope
from kdflow.algorithms._mp_opd_full_meta import ForwardParameterBridge


def test_streamed_callbacks_load_lazily_release_and_replay():
    import weakref
    from kdflow.algorithms._mp_opd_full_meta import streamed_inner_losses
    loaded, refs = [], []
    weight = torch.tensor(2., requires_grad=True)
    def loader(index):
        loaded.append(index)
        tensor = torch.tensor([[float(index + 1)]])
        refs.append(weakref.ref(tensor))
        return {'stu_input_ids':tensor}
    def step(batch):
        return {'loss':(weight*batch['stu_input_ids']).square().sum()}
    callbacks = streamed_inner_losses([0,1],loader,step,'cpu')
    assert loaded == []
    gradients=[]
    for callback in callbacks+callbacks:
        loss = callback()
        grad, = torch.autograd.grad(loss,weight,create_graph=True)
        second, = torch.autograd.grad(grad,weight)
        gradients.append((grad.item(),second.item()))
        del loss,grad,second
        assert all(ref() is None for ref in refs)
    assert loaded == [0,1,0,1]
    assert gradients == [(2.,1.),(8.,4.),(2.,1.),(8.,4.)]


def test_forward_bridge_preserves_plain_parameter_gradients():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)).double()
    bridge = ForwardParameterBridge(model)
    try:
        loss = model(torch.ones(2, 3, dtype=torch.double)).square().sum()
        params = tuple(model.parameters())
        expected = torch.autograd.grad(loss, params, retain_graph=True)
        actual = bridge.grad(loss, params)
        for a, b in zip(expected, actual):
            assert torch.equal(a, b)
    finally:
        bridge.close()
    assert not any(m._forward_pre_hooks for m in model.modules())


def test_disconnected_inner_cannot_silently_report_energy_update():
    p = torch.nn.Parameter(torch.ones(2))
    other = torch.nn.Parameter(torch.ones(2))
    e = torch.nn.Linear(2, 1, bias=False)
    opt = torch.optim.AdamW([p], lr=.01)
    eo = torch.optim.AdamW(e.parameters(), lr=.001)
    old = e.weight.detach().clone()
    with pytest.raises(RuntimeError, match='disconnected'):
        full_meta_step([p], opt, e, eo, [lambda: e(other).square().sum()],
                       [lambda: p.square().sum()], max_norm=1.)
    assert torch.equal(old, e.weight)
    assert not eo.state


@pytest.mark.parametrize("max_norm", [0., .1, 10.])
@pytest.mark.parametrize("history", [0, 3])
def test_streamed_hypergradient_matches_direct_unroll(max_norm, history):
    torch.manual_seed(4)
    p=torch.nn.Parameter(torch.tensor([.4,-.8],dtype=torch.double))
    e=torch.nn.Linear(2,1,bias=False).double()
    opt=torch.optim.AdamW([p],lr=.003,betas=(.9,.98),weight_decay=.01)
    eo=torch.optim.SGD(e.parameters(),lr=.01)
    for _ in range(history):
        p.grad=torch.tensor([.2,-.4],dtype=torch.double);opt.step();opt.zero_grad(set_to_none=True)
    x=[torch.tensor([1.,2.],dtype=torch.double),torch.tensor([-.5,1.],dtype=torch.double)]
    inner=[lambda x=x: (e(x).squeeze()*((p*x).sum()).square())/2 for x in x]
    meta=[lambda: (p-torch.tensor([.1,.2],dtype=torch.double)).square().mean()]
    g=torch.autograd.grad(sum(fn() for fn in inner),p,create_graph=True)[0]
    if max_norm>0:
        # Independent differentiable clipping reference.
        g=g*(max_norm/(torch.linalg.vector_norm(g)+1e-6)).clamp(max=1.)
    virtual=adam_value(p,g,opt.state.get(p,{}),opt.param_groups[0])
    objective=(virtual-torch.tensor([.1,.2],dtype=torch.double)).square().mean()
    expected=torch.autograd.grad(objective,tuple(e.parameters()))[0]
    before=p.detach().clone(); original_energy=e.weight.detach().clone(); state=copy.deepcopy(opt.state_dict())
    result=full_meta_step([p],opt,e,eo,inner,meta,max_norm=max_norm)
    assert torch.equal(p,before) and p.grad is None
    assert torch.allclose(e.weight,original_energy-.01*expected,rtol=1e-9,atol=1e-11)
    assert result['mp_opd_meta_nll']==pytest.approx(float(objective.detach()),abs=1e-12)
    for key,value in state['state'].items():
        for name,t in value.items():assert torch.equal(t,opt.state_dict()['state'][key][name])


def test_virtual_adam_matches_real_adam_and_preserves_state():
    p=torch.nn.Parameter(torch.tensor([.4,-.8],dtype=torch.double))
    opt=torch.optim.AdamW([p],lr=.002,betas=(.9,.98),weight_decay=.03)
    for step in range(5):
        g=torch.tensor([.3+step,.1-step],dtype=torch.double)
        virtual=adam_value(p,g,opt.state.get(p,{}),opt.param_groups[0])
        p.grad=g;opt.step();opt.zero_grad(set_to_none=True)
        assert torch.allclose(virtual,p,rtol=1e-14,atol=1e-14)


def test_adam_moment_offload_is_opt_in_and_cpu_safe():
    p = torch.nn.Parameter(torch.tensor([.4, -.8], dtype=torch.double))
    opt = torch.optim.AdamW([p], lr=.002)
    p.grad = torch.tensor([.3, -.1], dtype=torch.double)
    opt.step(); opt.zero_grad(set_to_none=True)
    before = copy.deepcopy(opt.state_dict())
    with offload_adam_moments_scope(opt, [p], enabled=True, device='cpu') as info:
        assert info == {"enabled": True, "offloaded_bytes": 0}
    after = opt.state_dict()
    assert before['param_groups'] == after['param_groups']
    for key, state in before['state'].items():
        for name, value in state.items():
            actual = after['state'][key][name]
            if torch.is_tensor(value):
                assert torch.equal(value, actual)
            else:
                assert value == actual


def test_virtual_failure_rolls_back_student():
    p=torch.nn.Parameter(torch.ones(2,dtype=torch.double)); e=torch.nn.Linear(2,1,bias=False).double()
    opt=torch.optim.AdamW([p],lr=.01);eo=torch.optim.AdamW(e.parameters(),lr=.001)
    before=p.detach().clone()
    def bad():raise RuntimeError('meta forward failed')
    with pytest.raises(RuntimeError,match='meta forward'):
        full_meta_step([p],opt,e,eo,[lambda:e(p).square().sum()],[bad],max_norm=1.)
    assert torch.equal(p,before) and not opt.state
