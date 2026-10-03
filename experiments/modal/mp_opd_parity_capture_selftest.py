"""Self-test for `diagnose_parity_capture.py` with synthetic captures.

The capture diagnostic is the one deliverable that can still settle
`BUG_parity_logprob_4096cap.md`, and it will be run by hand on a machine I cannot
reach. A silent bug in it would therefore be maximally expensive -- and two
off-by-one bugs of exactly this kind were already found in sibling tools during
this investigation. So it gets positive and negative controls on captures whose
anomaly is injected and known in advance.

Variants:

  clean        behaviour == the temperature-scaled eager reference  -> the detector
               must read ~0, the guard metric must be ~0, no outliers
  raw_stretch  behaviour replaced by the RAW reference over a contiguous middle
               stretch of the longest row (the leading hypothesis for production)
               -> frac_closer_to_raw must jump on that row, and at the outliers
               mean|d_raw| must collapse while mean|d_T| stays large
  noise_band   behaviour = reference + 1.5 over a contiguous middle stretch (a
               synthetic large kernel error) -> outliers must appear as one
               contiguous band and the detector must stay ~0

Run:

  MODAL_PROFILE=kieusontung8 uvx --from modal modal run \
    experiments/modal/mp_opd_parity_capture_selftest.py::main
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import modal

APP_NAME = "simct-parity-capture-selftest-20261003"
IMAGE = ("docker.io/codemaivanngu/simct-b200"
         "@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f")
ASSET_VOLUME_NAME = "simct-parity-longcap-assets-v1"
OUTPUT_VOLUME_NAME = "simct-parity-longcap-results-v1"
PYTHON = "/opt/venvs/simct-b200/bin/python"
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/opt/overlay")

BUILDER = r'''
import json, math, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]
OUT = Path(sys.argv[2])
VARIANT = sys.argv[3]
LONG = int(sys.argv[4])
TEMPERATURE = 0.6
# A period-8 repetition inside the long row, so the injected stretch looks like the
# production trajectory the bug note describes (150 outliers at stride 8).
CYCLE = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]

tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
eos = int(tok.eos_token_id)

def enc(text):
    return [int(v) for v in tok(text, return_tensors=None)["input_ids"]]

# Rows are built with the PRODUCTION trajectory contract, not an ad-hoc shape:
# `kdflow/trajectory.py::trajectory_tokens` does
#     synthetic_eos = not response or response[-1] != eos_id
#     labels = response + ([eos_id] if synthetic_eos else [])
#     ids    = prompt + labels
#     mask   = [False] * (len(prompt) - 1) + [True] * len(labels) + [False]
# and `on_policy_kd_trainer._build_exact_rollout_sample` writes behaviour only over
# [len(prompt) - 1, len(prompt) - 1 + len(sampled)) while the mask covers the
# synthetic terminal too. So a CAPPED row carries exactly one NaN on a masked
# position, and an EOS-terminated row carries none. Getting this wrong is how a
# diagnostic silently mis-aligns, so the self-test reproduces it faithfully.
PROMPT_FILLER = "Context for the self-test prompt; ignore it and answer below. "

long_lead = enc("Explain, in detail, why the following eight words repeat: ")
loop = enc(" ".join(CYCLE) + " ")
body = []
while len(long_lead) + len(body) < LONG:
    body += loop
# Capped: no EOS, so trajectory_tokens appends a synthetic terminal.
long_sampled = (long_lead + body)[:LONG]
short_sampled = [enc("Name three colours. "), enc("What is 2+2? Answer briefly. ")]
prompts = [enc(PROMPT_FILLER * 3), enc(PROMPT_FILLER), enc(PROMPT_FILLER)]
samples = [long_sampled, *[s + [eos] for s in short_sampled]]

rows = []
for prompt, sampled in zip(prompts, samples):
    synthetic = not sampled or sampled[-1] != eos
    labels = sampled + ([eos] if synthetic else [])
    rows.append({"prompt": prompt, "sampled": sampled, "labels": labels,
                 "ids": prompt + labels, "synthetic": synthetic})

model = AutoModelForCausalLM.from_pretrained(
    MODEL, local_files_only=True, dtype=torch.bfloat16,
    attn_implementation="eager").to("cuda").eval()

def row_scores(ids):
    t = torch.tensor([ids], device="cuda")
    logits = model(input_ids=t, use_cache=False).logits[0]
    labels = torch.tensor(ids[1:], device="cuda")
    raw, scaled = [], []
    with torch.inference_mode():
        for lo in range(0, logits.shape[0] - 1, 512):
            hi = min(lo + 512, logits.shape[0] - 1)
            block = logits[lo:hi].float()
            target = labels[lo:hi]
            raw.extend(torch.log_softmax(block, dim=-1).gather(
                -1, target[:, None]).squeeze(-1).tolist())
            scaled.extend(torch.log_softmax(block / TEMPERATURE, dim=-1).gather(
                -1, target[:, None]).squeeze(-1).tolist())
            del block
    del logits, t
    torch.cuda.empty_cache()
    return raw, scaled

width = max(len(r["ids"]) for r in rows)
n = len(rows)
ids_t = torch.zeros((n, width), dtype=torch.long)
attn_t = torch.zeros((n, width), dtype=torch.long)
loss_t = torch.zeros((n, width), dtype=torch.bool)
behav = torch.full((n, width), float("nan"))
trainer = []
positions = []
for r, row in enumerate(rows):
    seq = row["ids"]
    L = len(seq)
    P = len(row["prompt"])
    ids_t[r, :L] = torch.tensor(seq, dtype=torch.long)
    attn_t[r, :L] = 1
    start = P - 1
    # mask covers every label, including the synthetic terminal
    loss_t[r, start:start + len(row["labels"])] = True
    raw, scaled = row_scores(seq)
    for j in range(len(row["labels"])):
        p = start + j
        # behaviour exists only over the sampled tokens; the synthetic terminal
        # position stays NaN -- that is the sentinel the guard filters with isnan
        if j < len(row["sampled"]):
            behav[r, p] = scaled[j]
        trainer.append(scaled[j])
        positions.append((r, p))
    if r == 0 and VARIANT != "clean":
        a = int(len(row["labels"]) * 0.45)
        b = int(len(row["labels"]) * 0.80)
        for j in range(a, b):
            p = start + j
            if VARIANT == "raw_stretch":
                behav[r, p] = raw[j]
            elif VARIANT == "noise_band":
                behav[r, p] = scaled[j] + 1.5
            # keep the trainer side untouched: this is a serving-side anomaly

OUT.mkdir(parents=True, exist_ok=True)
torch.save({
    "stu_input_ids": ids_t,
    "stu_attn_mask": attn_t,
    "stu_loss_mask": loss_t,
    "stu_behavior_log_probs": behav,
    "actual_log_probs": torch.tensor(trainer),
    "positions": torch.tensor(positions),
    "temperature": float(TEMPERATURE),
}, OUT / "batch.pt")
(OUT / "metadata.json").write_text(json.dumps({
    "error": "synthetic self-test capture",
    "per_sample": [],
    "source_commit": "selftest",
    "model_training": False,
    "attention_backend": "eager",
    "torch_version": str(torch.__version__),
    "checkpoint_complete": True,
    "variant": VARIANT,
    "injected_rows": [0] if VARIANT != "clean" else [],
    "synthetic_terminal_rows": [r for r, row in enumerate(rows) if row["synthetic"]],
}, indent=2))
# `student/` is what the diagnostic loads; symlink keeps the 5 GB model out of the
# capture directory.
student = OUT / "student"
if not student.exists():
    student.symlink_to(MODEL)
print("SELFTEST_BUILT=" + json.dumps({
    "variant": VARIANT, "rows": n, "width": width,
    "long_row_tokens": len(long_ids), "capture": str(OUT)}), flush=True)
'''

app = modal.App(APP_NAME)
image = (
    modal.Image.from_registry(IMAGE)
    .entrypoint([])
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_file(
        str(ROOT / "experiments/runai/diagnose_parity_capture.py"),
        "/opt/overlay/experiments/runai/diagnose_parity_capture.py")
)
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
def selftest(run_id: str, variants: str, long_len: int) -> dict:
    root = Path("/runs") / run_id
    root.mkdir(parents=True, exist_ok=True)
    script = root / "builder.py"
    script.write_text(BUILDER)
    report: dict = {"run_id": run_id, "variants": {}}
    for variant in [v.strip() for v in variants.split(",") if v.strip()]:
        capture = root / ("capture-" + variant)
        built = subprocess.run(
            [PYTHON, str(script), "/assets/student", str(capture), variant, str(long_len)],
            env=runtime_environment(), text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=900)
        (root / ("build-" + variant + ".log")).write_text(built.stdout)
        if built.returncode:
            report["variants"][variant] = {"built": False, "log": built.stdout[-1500:]}
            continue
        diag = subprocess.run(
            [PYTHON, "/opt/overlay/experiments/runai/diagnose_parity_capture.py",
             str(capture), "--out", str(root / ("diag-" + variant + ".json")), "--engine"],
            env=runtime_environment(), text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=900)
        (root / ("diag-" + variant + ".log")).write_text(diag.stdout)
        outputs.commit()
        parsed = {}
        for line in diag.stdout.replace("\r", "\n").splitlines():
            s = line.strip()
            for key in ("fraction closer to HF(T=1)", "at ", "first outlier positions",
                        "gap histogram", "corr(|d|, 1-p_sampled)",
                        "guard_fresh_T_vs_behavior", "per row:",
                        "mean d_T", "PARITY_DIAG_OUT", "PARITY_DIAG_DONE",
                        "engine_prefill_raw_vs_hf_raw", "stored_scaled_vs_hf_T",
                        "VERDICT:", "ENGINE_CHECK_FAILED", "engine_alignment_warning"):
                if key in s:
                    parsed.setdefault(key, []).append(s[:240])
        report["variants"][variant] = {"built": True, "returncode": diag.returncode,
                                      "highlights": parsed}
        print("SELFTEST_VARIANT=" + json.dumps(
            {"variant": variant, "rc": diag.returncode, "highlights": parsed},
            sort_keys=True), flush=True)

        # Third run: simulate the capture whose 6.4 GB student checkpoint was
        # already cleaned up (the bug note flags a cleanup policy for it). The
        # model-free path must still report the stored-pair structure, and its
        # guard delta must agree with the model path's stored-pair numbers.
        student = capture / "student"
        if student.exists() or student.is_symlink():
            student.unlink()
        # CUDA_VISIBLE_DEVICES=-1 hides every GPU. If the model-free path still
        # produces its numbers, it provably needs no GPU -- which is what makes it
        # safe to run on a shared node without touching another tenant's card.
        nomodel_env = runtime_environment()
        nomodel_env["CUDA_VISIBLE_DEVICES"] = "-1"
        nomodel = subprocess.run(
            [PYTHON, "/opt/overlay/experiments/runai/diagnose_parity_capture.py",
             str(capture), "--out", str(root / ("diag-nomodel-" + variant + ".json"))],
            env=nomodel_env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=900)
        (root / ("diag-nomodel-" + variant + ".log")).write_text(nomodel.stdout)
        outputs.commit()
        picked = {}
        for line in nomodel.stdout.replace("\r", "\n").splitlines():
            s = line.strip()
            for key in ("guard delta from the capture itself", "row 0:", "shift test",
                        "PARITY_DIAG_MODEL", "periodicity:", "Traceback", "Error"):
                if key in s:
                    picked.setdefault(key, []).append(s[:240])
        report["variants"][variant]["nomodel"] = {"returncode": nomodel.returncode,
                                                  "highlights": picked}
        print("SELFTEST_NOMODEL=" + json.dumps(
            {"variant": variant, "rc": nomodel.returncode, "highlights": picked},
            sort_keys=True), flush=True)
    (root / "selftest.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    outputs.commit()
    return report


@app.local_entrypoint()
def main(run_id: str = "capture-selftest-20261003-r1",
         variants: str = "clean,raw_stretch,noise_band", long_len: int = 512):
    result = selftest.remote(run_id, variants, long_len)
    print("SELFTEST_RESULT=" + json.dumps(result, sort_keys=True, default=str)[:4000])
