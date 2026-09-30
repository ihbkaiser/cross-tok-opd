import pytest
import torch
from kdflow.algorithms._mp_opd_energy import MPAtomEnergy, energy_surrogate_loss

@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable locally'))])
def test_eval_gru_backward_without_cudnn_and_dropout(device):
    model=MPAtomEnergy(10,hidden_dim=4,layers=2).to(device).eval()
    features=torch.randn(4,10,device=device,requires_grad=True)
    utilities=torch.randn(4,2,device=device)
    seen=[]
    hook=model.encoder.register_forward_pre_hook(lambda module,args: seen.append((module.training,torch.backends.cudnn.enabled)))
    prior=torch.backends.cudnn.enabled
    loss,_=energy_surrogate_loss(model,features,utilities,max_span_length=2,temperature=1.,virtual_learning_rate=.1)
    loss.backward()
    assert seen==[(False,False)]
    assert torch.backends.cudnn.enabled==prior
    assert features.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        a=model(features,2);b=model(features,2)
    assert torch.equal(a,b)
    assert not model.encoder.training
    hook.remove()
