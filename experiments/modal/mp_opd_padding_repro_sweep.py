"""Find the minimal variable that makes a padded eager batch change the longest row.

Round 7 finding on the real capture: a padded-batch HF eager forward reproduces the
trainer's stored logprobs EXACTLY (max 0.0), while a per-row forward reproduces the
engine's stored logprobs (max 0.33). The padded-vs-row difference is 0.0414 mean /
1.7597 max on 150 positions -- computed in HF alone, with no SGLang involved.

But an earlier synthetic padding test of mine reported max 0.0 for every padding
variant, which is why this went unfound for several rounds. So this sweeps ONE
variable at a time from a base that copies the capture's shape:

    lengths [506, 224, 4096, 644], longest row at index 2, pad id 0,
    attention_mask long, position_ids = cumsum(mask) - 1 with pads forced to 1,
    model.train(True), padded forward under torch.inference_mode()

Reported per variant: max and count>0.5 of |padded - per_row| on the longest row,
plus |padded - row_masked| so "padding" is separated from "mask/positions".

  MODAL_PROFILE=kieusontung8 uvx --from modal modal run \
    experiments/modal/mp_opd_padding_repro_sweep.py::main
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import modal

APP_NAME = "simct-padding-repro-sweep-20261003"
IMAGE = ("docker.io/codemaivanngu/simct-b200"
         "@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f")
ASSET_VOLUME_NAME = "simct-parity-longcap-assets-v1"
OUTPUT_VOLUME_NAME = "simct-parity-longcap-results-v1"
PYTHON = "/opt/venvs/simct-b200/bin/python"

WORKER = r'''
import json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, OUT = sys.argv[1], Path(sys.argv[2])
TEMPERATURE = 0.6
# The capture's exact row lengths, longest at index 2. Lengths are not internal data.
BASE_LENGTHS = [506, 224, 4096, 644]
LONG_INDEX = 2

tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
FILLER = ("Padding sweep filler text; the words repeat so the token stream has a "
          "period, which is what makes one position of the cycle numerically hard. ")
pool = tok(FILLER * 600, return_tensors=None)["input_ids"]

def row_ids(n, offset=0):
    out = []
    while len(out) < n:
        out.extend(pool)
    return [int(v) for v in out[offset:offset + n]]

model = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager").to("cuda")

def logprobs_at(model, ids, attn, positions, rows_of_interest):
    """logprob of ids[r, j+1] from logits[r, j] for j in the response part of each row."""
    with torch.inference_mode():
        logits = model(input_ids=ids.cuda(),
                       attention_mask=(attn.cuda() if attn is not None else None),
                       position_ids=(positions.cuda() if positions is not None else None),
                       use_cache=False).logits
    got = {}
    for r in rows_of_interest:
        n = int(attn[r].sum().item()) if attn is not None else ids.shape[1]
        vals = []
        for j in range(0, n - 1):
            block = logits[r, j].float()
            lab = int(ids[r, j + 1].item())
            vals.append(float(torch.log_softmax(block / TEMPERATURE, -1)[lab]))
        got[r] = vals
    del logits
    torch.cuda.empty_cache()
    return got

def compare(a, b):
    n = min(len(a), len(b))
    d = [abs(a[i] - b[i]) for i in range(n)]
    return {"n": n, "mean": sum(d) / n, "max": max(d),
            "above_0p5": sum(1 for v in d if v > 0.5)}


def logit_delta(model, ids, attn, positions, lone_ids, row_index):
    """max|delta logit| between the padded forward and the per-row forward.

    This is the content-independent measurement. A logprob comparison cannot see the
    difference at peaked positions -- delta_logprob ~ delta_logit * (1 - p) -- which
    is why a logprob-only sweep reported 0.0 for every variant even though the capture
    shows a 1.76 nat effect at the one flat position of its period-8 loop.
    """
    with torch.inference_mode():
        a = model(input_ids=ids.cuda(), attention_mask=attn.cuda(),
                  position_ids=(positions.cuda() if positions is not None else None),
                  use_cache=False).logits[row_index]
        b = model(input_ids=lone_ids.cuda(), use_cache=False).logits[0]
        n = int(b.shape[0])
        worst, worst_pos = 0.0, -1
        means = []
        for lo in range(0, n, 256):
            hi = min(lo + 256, n)
            d = (a[lo:hi].float() - b[lo:hi].float()).abs()
            means.append(float(d.mean()))
            m = float(d.max())
            if m > worst:
                worst, worst_pos = m, lo + int(d.max(dim=1).values.argmax())
            del d
        del a, b
    torch.cuda.empty_cache()
    return {"max_abs_dlogit": worst, "worst_position": worst_pos,
            "mean_abs_dlogit": sum(means) / len(means), "positions": n}

def build(lengths, pad_id=0):
    width = max(lengths)
    n = len(lengths)
    ids = torch.full((n, width), pad_id, dtype=torch.long)
    attn = torch.zeros((n, width), dtype=torch.long)
    for r, L in enumerate(lengths):
        ids[r, :L] = torch.tensor(row_ids(L, offset=r * 7), dtype=torch.long)
        attn[r, :L] = 1
    return ids, attn

VARIANTS = {
    "base": {},
    "eval_mode": {"eval": True},
    "short_lengths_small": {"lengths": [83, 61, 37, 4096], "long_index": 3},
    "long_first": {"lengths": [4096, 506, 224, 644], "long_index": 0},
    "long_last": {"lengths": [506, 224, 644, 4096], "long_index": 3},
    "no_positions": {"no_positions": True},
    "attn_bool": {"attn_bool": True},
    "padid_eos": {"pad_id_from_eos": True},
    "no_inference_mode": {"no_inference_mode": True},
    "all_rows_long": {"lengths": [4096, 4096, 4096, 4096], "long_index": 0},
    "long_4095": {"lengths": [506, 224, 4095, 644], "long_index": 2},
    "two_rows": {"lengths": [224, 4096], "long_index": 1},
}

results = {}
for name, cfg in VARIANTS.items():
    lengths = cfg.get("lengths", BASE_LENGTHS)
    long_index = cfg.get("long_index", LONG_INDEX)
    model.train(not cfg.get("eval", False))
    pad_id = int(tok.eos_token_id) if cfg.get("pad_id_from_eos") else 0
    ids, attn = build(lengths, pad_id=pad_id)
    if cfg.get("attn_bool"):
        attn = attn.bool()
    pos = attn.long().cumsum(-1) - 1
    pos.masked_fill_(attn == 0, 1)
    positions = None if cfg.get("no_positions") else pos

    ctx = (torch.inference_mode() if not cfg.get("no_inference_mode")
           else __import__("contextlib").nullcontext())
    with ctx:
        padded = logprobs_at(model, ids, attn, positions, [long_index])
    # per-row reference: the longest row alone, no mask, no position_ids
    lone_ids = ids[long_index:long_index + 1, :lengths[long_index]].clone()
    lone_attn = torch.ones_like(lone_ids)
    alone = logprobs_at(model, lone_ids, None, None, [0])
    # per-row but WITH mask + explicit positions (separates padding from mask/pos)
    with ctx:
        masked = logprobs_at(model, lone_ids, lone_attn, None, [0])
    res = {
        "lengths": lengths, "long_index": long_index,
        "padded_vs_row_plain": compare(padded[long_index], alone[0]),
        "padded_vs_row_masked": compare(padded[long_index], masked[0]),
        "row_plain_vs_row_masked": compare(alone[0], masked[0]),
        # content-independent: the logit difference itself
        "logit_delta": logit_delta(model, ids, attn,
                                   positions if positions is not None else pos,
                                   lone_ids, long_index),
    }
    results[name] = res
    print("SWEEP=" + json.dumps({"variant": name,
                                "logit_delta": res["logit_delta"],
                                "padded_vs_row_masked": res["padded_vs_row_masked"]},
                               sort_keys=True), flush=True)

OUT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
print("SWEEP_DONE=1", flush=True)
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


@app.function(image=image, gpu="B200", cpu=8, memory=65536, timeout=3600, retries=0,
              max_containers=1, volumes={"/assets": assets, "/runs": outputs})
def sweep(run_id: str) -> dict:
    root = Path("/runs") / run_id
    root.mkdir(parents=True, exist_ok=True)
    script = root / "worker.py"
    script.write_text(WORKER)
    done = subprocess.run([PYTHON, str(script), "/assets/student", str(root / "sweep.json")],
                          env=runtime_environment(), text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=3500)
    (root / "worker.log").write_text(done.stdout)
    outputs.commit()
    print(done.stdout[-7000:], flush=True)
    if done.returncode:
        raise RuntimeError("sweep failed rc=" + str(done.returncode))
    return json.loads((root / "sweep.json").read_text())


@app.local_entrypoint()
def main(run_id: str = "padding-sweep-20261003-r1"):
    print("SWEEP_RESULT=" + json.dumps(sweep.remote(run_id), sort_keys=True)[:6000])
