"""CUDA integration checks, NOT pretrained-model training or paper reproduction.

Uses unmodified upstream Fp32SparseMM and CrossTokenizerDistillationLossFn.
Synthetic projection/alignment/logits exercise the large-vocabulary CUDA path.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, "/workspace/xtoken-native/NeMo-RL")
from nemo_rl.algorithms.x_token.loss_utils import Fp32SparseMM, LocalizedAlignment
from nemo_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn


def sparse_reference():
    torch.manual_seed(43)
    dense_m = torch.rand(32, 24, device="cuda")
    dense_m[dense_m < 0.75] = 0
    m = dense_m.to_sparse().coalesce()
    x = torch.randn(32, 16, device="cuda", requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", enabled=False):
        reference = dense_m.t() @ reference_x
        reference.square().mean().backward()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = Fp32SparseMM.apply(m, x)
        actual.square().mean().backward()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-5, atol=1e-5)
    return {"pass": True, "output_dtype": str(actual.dtype),
            "max_forward_error": (actual-reference).abs().max().item(),
            "max_backward_error": (x.grad-reference_x.grad).abs().max().item()}


def bf16_sdpa():
    torch.manual_seed(43)
    layer = torch.nn.Linear(256, 768, device="cuda")
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
    x = torch.randn(1, 128, 256, device="cuda")
    before = layer.weight.detach().clone()
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q, k, v = layer(x).reshape(1, 128, 3, 4, 64).permute(2, 0, 3, 1, 4)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            loss = y.float().square().mean()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0
        optimizer.step()
        losses.append(loss.item())
    delta = (layer.weight-before).abs().max().item()
    assert y.dtype == torch.bfloat16 and delta > 0
    return {"pass": True, "losses": losses, "max_weight_change": delta,
            "activation_dtype": str(y.dtype), "parameter_dtype": str(layer.weight.dtype)}


def large_vocab_pkl(out, seq_len):
    torch.manual_seed(43)
    vs, vt = 128256, 151936
    ids = torch.arange(vs, dtype=torch.long)
    indices = torch.stack((ids % vt, (ids + vs) % vt), dim=1)
    weights = torch.tensor([0.7, 0.3]).repeat(vs, 1)
    path = out / f"synthetic-projection-{seq_len}.pt"
    torch.save({"indices": indices, "likelihoods": weights}, path)
    cfg = {"gold_loss": False, "xtoken_loss": False, "temperature": 1.0,
           "vocab_topk": 8192, "uncommon_topk": 8192, "reverse_kl": False,
           "exact_token_match_only": False, "kl_loss_weight": 1.0,
           "ce_loss_scale": 0.1, "dynamic_loss_scaling": False,
           "kd_loss_mode": "sum", "alpha": 1.0, "normalize_teacher_by_vocab": False,
           "student_vocab_size": vs, "teacher_vocab_sizes": [vt],
           "projection_matrix_paths": [str(path)], "teacher_weights": [1.0],
           "teacher_gold_loss": [None], "teacher_xtoken_loss": [None]}
    loss_fn = CrossTokenizerDistillationLossFn(cfg)
    teacher_len = seq_len + 32
    student = torch.nn.Parameter(torch.randn(1, seq_len, vs, device="cuda"))
    teacher = torch.randn(1, teacher_len, vt, device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW([student], lr=0.01)
    before = student[0, 0, :1024].detach().clone()
    alignment = LocalizedAlignment(
        sample_mask=torch.ones(1, device="cuda"),
        student_chunk_id=(torch.arange(seq_len, device="cuda") * 8 // seq_len).unsqueeze(0),
        teacher_chunk_id=(torch.arange(teacher_len, device="cuda") * 8 // teacher_len).unsqueeze(0),
        pair_valid=torch.ones(1, 8, device="cuda", dtype=torch.bool),
        pair_is_correct=torch.ones(1, 8, device="cuda", dtype=torch.bool))
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for step in range(1, 4):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, valid_chunks, accuracy = loss_fn._compute_p_kl(
                student, teacher, alignment, projection_matrix_path=str(path),
                teacher_vocab_size=vt)
            loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(student.grad).all()
        grad_norm = student.grad.norm().item()
        assert grad_norm > 0 and valid_chunks.item() == 8
        optimizer.step()
        torch.cuda.synchronize()
        row = {"step": step, "loss": loss.item(), "grad_norm": grad_norm,
               "elapsed_s": time.perf_counter()-start, "valid_chunks": valid_chunks.item(),
               "projected_accuracy": accuracy.item()}
        print(json.dumps(row), flush=True)
        rows.append(row)
    delta = (student[0, 0, :1024]-before).abs().max().item()
    assert delta > 0
    return {"pass": True, "scope": "synthetic P-KL loss kernel plus optimizer only",
            "student_shape": list(student.shape), "teacher_shape": list(teacher.shape),
            "parameter_dtype": str(student.dtype), "teacher_dtype": str(teacher.dtype),
            "projection_is_synthetic": True, "pretrained_model_used": False,
            "steps": rows, "max_weight_change_sample": delta,
            "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30}


p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
args = p.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.is_bf16_supported()
torch.backends.cuda.matmul.allow_tf32 = False
result = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
          "gpu": torch.cuda.get_device_name(), "bf16_supported": True,
          "sparse_reference": sparse_reference(), "sdpa": bf16_sdpa(), "pkl": []}
for length in (128, 256):
    gc.collect()
    torch.cuda.empty_cache()
    result["pkl"].append(large_vocab_pkl(out, length))
result["pass"] = True
(out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
