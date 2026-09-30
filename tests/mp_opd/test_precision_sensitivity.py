"""Small CPU regression for the diagnostic's low-precision addition boundary."""
import torch

def test_nonzero_adapter_gradient_does_not_guarantee_bf16_forward_change():
    b=torch.tensor(0.,requires_grad=True)
    loss=(torch.tensor(1.,dtype=torch.bfloat16)+b.to(torch.bfloat16)).float()
    grad=torch.autograd.grad(loss,b)[0]
    updated=-1e-4*grad
    assert grad.item()!=0 and updated.item()!=0
    assert (torch.tensor(1.,dtype=torch.bfloat16)+updated.to(torch.bfloat16)).item()==1.
    assert (torch.tensor(1.)+updated).item()<1.
