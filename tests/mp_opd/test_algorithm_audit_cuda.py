"""Target-runtime checks; no model downloads or production training required."""
import copy
import json
import os

import pytest
import torch

from kdflow.algorithms._mp_opd_atoms import AtomizationResult
from kdflow.algorithms._mp_opd_trust_batch import TrustHeadAccumulator
from kdflow.training_checkpoint import capture_rng, restore_rng
from test_trust_batch_replay import ATOMS, _fit_body, _reference_coefficients
from test_numeric_fallback import _make_algo
from types import SimpleNamespace

pytestmark = pytest.mark.skipif(os.environ.get("AUDIT_TEST_DEVICE") != "cuda",
                                reason="bounded target-runtime CUDA qualification")


def test_gemma2_bf16_b64_m4_full_batch_update_matches_autograd():
    from transformers import Gemma2Config, Gemma2ForCausalLM
    torch.manual_seed(942)
    config = Gemma2Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16,
                         max_position_embeddings=128, sliding_window=64,
                         attention_dropout=.2, final_logit_softcapping=30.,
                         attn_logit_softcapping=50., pad_token_id=0,
                         eos_token_id=255, bos_token_id=1)
    config._attn_implementation = "eager"

    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Gemma2ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
            self.model_config = config
            self.frames = []
        def forward(self, ids, **kwargs):
            out = self.model(ids, attention_mask=kwargs.get("attention_mask"),
                             output_hidden_states=True, use_cache=False)
            self.frames.append(out.logits.detach().clone())
            return {"logits": out.logits, "hidden_states": out.hidden_states}

    model = Student().train()
    reference = copy.deepcopy(model)
    algo = _make_algo("trust_b")
    algo.student = model
    algo.grass_softcap = 30.
    algo.grass_head_bias = False
    algo.teacher_lm_head = torch.nn.Linear(64, 256, bias=False).cuda().to(torch.bfloat16)
    algo.atomizer = SimpleNamespace(atomize=lambda *args, **kwargs:
                                   AtomizationResult(ATOMS, True, None, 3, 3))
    algo.strategy.accumulated_gradient = 16
    batches = []
    for _ in range(16):
        ids = torch.randint(1, 255, (4, 4), device="cuda")
        batches.append(dict(stu_input_ids=ids, tea_input_ids=ids.clone(),
                            stu_loss_mask=torch.tensor([[1, 1, 1, 0]]*4, device="cuda", dtype=torch.bool),
                            tea_loss_mask=torch.tensor([[1, 1, 1, 0]]*4, device="cuda", dtype=torch.bool),
                            stu_attn_mask=torch.ones_like(ids),
                            teacher_hiddens=torch.randn(12, 64, device="cuda", dtype=torch.bfloat16),
                            avg_micro_batch_token_num=torch.tensor(12., device="cuda")))
    initial_rng = capture_rng()
    optim = torch.optim.SGD(model.parameters(), lr=.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=1.)
    actual_gradients = []
    class Strategy:
        step = 0
        accumulated_gradient = 16
        def backward(self, loss, *args):
            self.step = (self.step+1) % 16
            (loss/16).backward()
        def optimizer_step(self, optimizer, student, schedule):
            if self.step == 0:
                actual_gradients.extend(p.grad.detach().clone() for p in student.parameters())
                optimizer.step(); optimizer.zero_grad(); schedule.step()
    actor = SimpleNamespace(student=model, kd_algorithm=algo, optim=optim,
                            scheduler=scheduler, strategy=Strategy(), args=algo.args, _world_size=1)
    status = _fit_body()(actor, batches)
    assert status["optimizer_updates"] == 1
    assert status["mp_opd_trust_calibration_response_count"] == 64
    assert status["mp_opd_numeric_fallback_fraction"] == 0
    assert len(model.frames) == 32
    for first, second in zip(model.frames[:16], model.frames[16:]):
        assert torch.equal(first, second), "Gemma2 no-grad/backward replay changed logits"
    restore_rng(initial_rng)
    for batch in batches:
        logits = reference(batch["stu_input_ids"], attention_mask=batch["stu_attn_mask"])["logits"]
        credits = _reference_coefficients(algo, batch, logits)
        loss = sum(c.rate[0]*c.current_nll[0] + status["mp_opd_trust_lambda"]*c.rate[1]*c.current_nll[1]
                   for c in credits)
        (loss/(12*16)).backward()
    maximum_error = 0.
    maximum_gradient_error = 0.
    for actual, parameter in zip(actual_gradients, reference.parameters()):
        maximum_gradient_error = max(maximum_gradient_error, float((actual.float()-parameter.grad.float()).abs().max()))
        assert torch.equal(actual, parameter.grad), "BF16 accumulated gradients differ"
    # Match optimizer semantics: SGD add_(grad, alpha=-lr) rounds BF16 once.
    # ref - lr*grad uses two BF16 operators and is not the same update oracle.
    torch.optim.SGD(reference.parameters(), lr=.02).step()
    with torch.no_grad():
        for actual, ref in zip(model.parameters(), reference.parameters()):
            maximum_error = max(maximum_error, float((actual.float()-ref.float()).abs().max()))
            assert torch.equal(actual, ref), "BF16 transformer optimizer update differs"
    print("GEMMA2_B64_M4_JSON="+json.dumps({
        "dtype": "bfloat16", "softcap": 30., "optimizer_updates": 1,
        "responses": 64, "lambda": status["mp_opd_trust_lambda"],
        "max_parameter_error": maximum_error, "max_gradient_error": maximum_gradient_error,
        "dropout_replay": "exact",
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
    }))


def test_gemma2_2b_head_shape_accumulator_has_bounded_memory_and_correct_lambda():
    # Actual production head dimensions; identical token deltas and hidden rows
    # give G_M=15.5 G_S, so lambda has an analytic oracle without another 4GiB head.
    vocab, hidden_size, tokens = 256000, 2304, 32
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    logits = torch.zeros(tokens, vocab, device="cuda")
    hidden = torch.ones(tokens, hidden_size, device="cuda")
    labels = torch.ones(tokens, device="cuda", dtype=torch.long)
    selected = torch.full((tokens,), -torch.log(torch.tensor(float(vocab))).item(), device="cuda")
    collector = TrustHeadAccumulator(vocab_chunk=4096)
    collector.add_response(logits, hidden, labels, [(0, 1), (1, tokens)],
                           torch.tensor([1., .5], device="cuda"),
                           torch.tensor([True, False], device="cuda"), selected)
    result = collector.finalize()
    expected_s2 = hidden_size*(1.-1./vocab)
    assert result.lam == pytest.approx(1./15.5, rel=2e-6)
    assert result.gs_norm2 == pytest.approx(expected_s2, rel=2e-6)
    assert result.gm_norm2 == pytest.approx(expected_s2*15.5**2, rel=2e-6)
    peak = (torch.cuda.max_memory_allocated()-before)/2**30
    assert peak < 6., "head accumulator unexpectedly retained token/atom batch matrices"
    assert collector.gs is None and collector.gm is None
    print("GEMMA2_HEAD_CAPACITY_JSON="+json.dumps({
        "vocab": vocab, "hidden": hidden_size, "peak_extra_allocated_gib": peak,
        "lambda": result.lam, "expected_lambda": 1./15.5,
        "gpu": torch.cuda.get_device_name(),
    }))
