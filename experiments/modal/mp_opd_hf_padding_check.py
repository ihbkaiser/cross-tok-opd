"""Isolate why a padded HF batch changes the longest row's logprobs.

The long-cap probe found that HF eager on a padded batch disagrees with the same
row computed alone by up to 5.4 nats in logprob (35% of positions above 0.5).
That is the trainer's situation: the trainer scores a padded micro-batch while the
serving engine scores each request alone. Before blaming the trainer this must be
reduced to a mechanism, because a causal model's longest row has no padding of its
own, so the two forwards *should* be identical.

Variants, all on the same long row, all compared against the same single-row
reference:

  alone_plain        one row, no attention_mask, no position_ids
  alone_mask         one row, attention_mask=ones, position_ids=arange
  alone_mask_nopos   one row, attention_mask=ones, no position_ids
  padded_pos        4 rows padded, attention_mask, position_ids=cumsum-1 (pads->1)
  padded_nopos      4 rows padded, attention_mask, no position_ids
  padded_sdpa       as padded_pos but attn_implementation=sdpa
  padded_long_last  as padded_pos but the long row is the last row
  padded_leftpad    as padded_pos but short rows are LEFT padded
  padded_padid_eos  as padded_pos but the pad token id is the EOS id

Per-band deltas over the long row are printed, because *where* the difference
appears identifies the mechanism: a boundary/window effect is localised, a
batching numeric effect is spread out, and a mask-construction bug starts at a
specific position.

  uvx --from modal modal run experiments/modal/mp_opd_hf_padding_check.py::main
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import modal

APP_NAME = "simct-hf-padding-check-20261003"
IMAGE = ("docker.io/codemaivanngu/simct-b200"
         "@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f")
ASSET_VOLUME_NAME = "simct-parity-longcap-assets-v1"
OUTPUT_VOLUME_NAME = "simct-parity-longcap-results-v1"
PYTHON = "/opt/venvs/simct-b200/bin/python"
LOCAL_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/opt/overlay")

WORKER = r'''
import json, math, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]
OUT = Path(sys.argv[2])
LONG = int(sys.argv[3])
BAND = 128
T = 0.6

tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
base_text = ("The quick brown fox jumps over the lazy dog near the river bank. "
             "Numbers, punctuation and spacing all matter for tokenisation here. ")
ids = tok(base_text * 400, return_tensors=None)["input_ids"]
while len(ids) < LONG:
    ids = (ids + ids)[:LONG]
ids = ids[:LONG]
shorts = [ids[:n] for n in (37, 61, 83)]
print("LONG_IDS=" + str(len(ids)), flush=True)

def stats(a, b, label, positions=None):
    n = min(len(a), len(b))
    d = [abs(a[i] - b[i]) for i in range(n)]
    worst = max(range(n), key=lambda i: d[i])
    bands = []
    for lo in range(0, n, BAND):
        seg = d[lo:lo + BAND]
        bands.append([lo, lo + len(seg), round(sum(seg) / len(seg), 5), round(max(seg), 4)])
    row = {"label": label, "n": n, "mean": sum(d) / n, "max": d[worst],
           "worst_index": worst, "above_0p5": sum(1 for v in d if v > 0.5),
           "bands": bands}
    print("PADCHK=" + json.dumps(row), flush=True)
    return row

def logprobs(logits, prompt_len, out_ids, temperature):
    start = prompt_len - 1
    stop = start + len(out_ids)
    labels = torch.tensor(out_ids, device=logits.device)
    raw, scaled = [], []
    for lo in range(start, stop, 512):
        hi = min(lo + 512, stop)
        block = logits[lo:hi].float()
        lr = torch.log_softmax(block, dim=-1)
        raw.extend(lr.gather(-1, labels[lo - start:hi - start, None]).squeeze(-1).tolist())
        lt = torch.log_softmax(block / temperature, dim=-1)
        scaled.extend(lt.gather(-1, labels[lo - start:hi - start, None]).squeeze(-1).tolist())
        del block, lr, lt
    return raw, scaled

results = {"long_tokens": len(ids), "short_tokens": [len(s) for s in shorts]}
models = {}

def get_model(attn):
    if attn not in models:
        models[attn] = AutoModelForCausalLM.from_pretrained(
            MODEL, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation=attn).to("cuda").eval()
    return models[attn]

reference = None
# Treat the whole long row as "response" behind a one-token virtual prompt, so the
# predicting logit for ids[j] is logits[j-1] for j in 1..L-1.
TARGETS = ids[1:]
with torch.inference_mode():
    for attn in ("eager", "sdpa"):
        model = get_model(attn)
        if attn == "eager":
            t = torch.tensor([ids], device="cuda")
            logits = model(input_ids=t, use_cache=False).logits[0]
            reference, _ = logprobs(logits, 1, TARGETS, T)
            del logits, t
            results["alone_plain_tokens"] = len(reference)
        # single row, explicit mask + positions
        t = torch.tensor([ids], device="cuda")
        a = torch.ones_like(t)
        p = torch.arange(t.shape[1], device="cuda").unsqueeze(0)
        logits = model(input_ids=t, attention_mask=a, position_ids=p, use_cache=False).logits[0]
        got, _ = logprobs(logits, 1, TARGETS, T)
        stats(reference, got, f"alone_mask_{attn}")
        del logits, t, a, p
        # single row, mask but no positions
        t = torch.tensor([ids], device="cuda")
        a = torch.ones_like(t)
        logits = model(input_ids=t, attention_mask=a, use_cache=False).logits[0]
        got, _ = logprobs(logits, 1, TARGETS, T)
        stats(reference, got, f"alone_mask_nopos_{attn}")
        del logits, t, a
        if attn != "eager":
            continue
        # padded batch variants
        for variant in ("padded_pos", "padded_nopos", "padded_long_last",
                        "padded_leftpad", "padded_padid_eos"):
            width = len(ids)
            rows = [ids] + shorts
            pad_id = 0
            if variant == "padded_long_last":
                rows = shorts + [ids]
            if variant == "padded_padid_eos":
                pad_id = int(tok.eos_token_id)
            ids_t = torch.full((len(rows), width), pad_id, dtype=torch.long)
            mask_t = torch.zeros((len(rows), width), dtype=torch.long)
            for r, seq in enumerate(rows):
                n = len(seq)
                if variant == "padded_leftpad" and n < width:
                    ids_t[r, width - n:] = torch.tensor(seq, dtype=torch.long)
                    mask_t[r, width - n:] = 1
                else:
                    ids_t[r, :n] = torch.tensor(seq, dtype=torch.long)
                    mask_t[r, :n] = 1
            pos_t = mask_t.cumsum(-1) - 1
            pos_t.masked_fill_(mask_t == 0, 1)
            long_row = len(rows) - 1 if variant == "padded_long_last" else 0
            kwargs = {"input_ids": ids_t.cuda(), "attention_mask": mask_t.cuda(),
                      "use_cache": False}
            if variant != "padded_nopos":
                kwargs["position_ids"] = pos_t.cuda()
            logits = model(**kwargs).logits[long_row]
            got, _ = logprobs(logits, 1, TARGETS, T)
            stats(reference, got, variant)
            del logits, ids_t, mask_t, pos_t
    for m in models.values():
        del m
    torch.cuda.empty_cache()
OUT.write_text(json.dumps(results, indent=2) + "\n")
print("PADCHK_DONE=1", flush=True)
'''

app = modal.App(APP_NAME)
image = modal.Image.from_registry(IMAGE).entrypoint([]).env({"PYTHONUNBUFFERED": "1"})
assets = modal.Volume.from_name(ASSET_VOLUME_NAME, create_if_missing=False)
outputs = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=False)


def runtime_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(PATH=f"/opt/venvs/simct-b200/bin:{environment.get('PATH','')}",
                       HF_HOME="/assets/hf", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    nvidia = Path("/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia")
    libs = ["/usr/local/cuda/lib64", "/usr/local/nvidia/lib64"]
    libs.extend(str(p) for p in sorted(nvidia.glob("*/lib")))
    environment["LD_LIBRARY_PATH"] = ":".join(libs)
    return environment


@app.function(image=image, gpu="B200", cpu=8, memory=65536, timeout=1800, retries=0,
              max_containers=1, volumes={"/assets": assets, "/runs": outputs})
def run_check(run_id: str, long_tokens: int) -> dict:
    out = Path("/runs") / run_id
    out.mkdir(parents=True, exist_ok=True)
    script = out / "worker.py"
    script.write_text(WORKER)
    done = subprocess.run([PYTHON, str(script), "/assets/student", str(out / "result.json"),
                           str(long_tokens)],
                          env=runtime_environment(), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1700)
    (out / "worker.log").write_text(done.stdout)
    outputs.commit()
    print(done.stdout[-6000:], flush=True)
    if done.returncode:
        raise RuntimeError("padding check failed rc=" + str(done.returncode))
    return json.loads((out / "result.json").read_text())


@app.local_entrypoint()
def main(run_id: str = "hf-padding-20261003-r1", long_tokens: int = 1024):
    print("PADCHK_RESULT=" + json.dumps(run_check.remote(run_id, long_tokens), sort_keys=True))
