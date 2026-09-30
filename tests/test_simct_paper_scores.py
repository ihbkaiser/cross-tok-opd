from types import SimpleNamespace
import torch
from kdflow.algorithms.span_ctkd import SpanCrossTokenizerKD

def test_paper_scores_match_log_probability_and_row_shift_invariance():
    a=object.__new__(SpanCrossTokenizerKD)
    a.args=SimpleNamespace(kd=SimpleNamespace(span_score_mode='mean_logprob'))
    a.student_overlap_token_ids=torch.tensor([0,1]);a.teacher_overlap_token_ids=torch.tensor([0,1]);a._span_debug_count=1
    s=torch.tensor([[1.,2.,3.],[4.,0.,-1.]],requires_grad=True)
    t=torch.tensor([[3.,0.,2.],[1.,2.,0.]])
    def compute(s,t):return a._build_virtual_vocab_logits([(0,2,0,2)],s,t,[2,0],[0,1])
    sv,tv=compute(s,t)
    sl=s.log_softmax(-1);tl=t.log_softmax(-1)
    torch.testing.assert_close(sv[0],torch.stack([sl[0,0],sl[0,1],(sl[0,2]+sl[1,0])/2]))
    torch.testing.assert_close(tv[0],torch.stack([tl[0,0],tl[0,1],(tl[0,0]+tl[1,1])/2]))
    shifted=compute(s+torch.tensor([[10.],[-5.]]),t+torch.tensor([[-2.],[8.]]))
    torch.testing.assert_close(sv,shifted[0]);torch.testing.assert_close(tv,shifted[1])
    loss=(sv.softmax(-1)*(sv.log_softmax(-1)-tv.log_softmax(-1))).sum();loss.backward()
    assert torch.isfinite(s.grad).all() and s.grad.abs().sum()>0
