"""Numeric-guardrail tests on Modal, on CPU.

Covers the downgrade of numeric guardrails to warnings for the geometry modes
(grass, grass_chunk, align, trust_r): the parity tripwire flag, the atomic
fallback path, and the mode-wiring invariant. Same cost rules as the other CPU
runners: no GPU, torch from the CPU index, explicit file list.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/work")
RUN = "mp-opd-guardrails-cpu-tests-20261008-r1"

MODULES = (
    "_mp_opd_trust.py",
    "_mp_opd_align.py",
    "_mp_opd_grass_chunk.py",
    "_mp_opd_grass_span.py",
    "_mp_opd_credit.py",
    "_mp_opd_credit_transform.py",
    "_mp_opd_energy.py",
    "_mp_opd_gbv_span.py",
    "_mp_opd_airs.py",
    "_mp_opd_oracle.py",
    "_mp_opd_semimarkov.py",
    "_mp_opd_training_diagnostics.py",
    "_mp_opd_atoms.py",
    "_parity_capture.py",
)
# Importing mp_opd executes kdflow/loss/__init__.py, which globs the whole
# directory: every loss file must be present or the import dies. All of them
# are pure torch, so the cost is bytes, not dependencies.
LOSS_MODULES = (
    "adaptive_kl_div.py",
    "cross_entropy.py",
    "hierarchical_ranking_loss.py",
    "js_div.py",
    "kl_div.py",
    "reverse_kl_div.py",
    "skewed_kl_div.py",
    "skewed_rkl_div.py",
    "top1_ce.py",
    "tvd.py",
)
TESTS = (
    "test_parity_capture.py",
    "test_grass_chunk.py",
    "test_align.py",
    "test_trust.py",
    "test_numeric_fallback.py",
)

# mp_opd.py is shipped but never imported: the suites read it as source text.
SOURCE_ONLY = ("mp_opd.py",)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pytest", "numpy")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_file(ROOT / "kdflow" / "__init__.py", "/work/kdflow/__init__.py")
    .add_local_file(
        ROOT / "kdflow" / "algorithms" / "__init__.py",
        "/work/kdflow/algorithms/__init__.py",
    )
    .add_local_file(
        ROOT / "kdflow" / "arguments" / "distillation_args.py",
        "/work/kdflow/arguments/distillation_args.py",
    )
)

for _name in MODULES:
    image = image.add_local_file(
        ROOT / "kdflow" / "algorithms" / _name, f"/work/kdflow/algorithms/{_name}"
    )
image = image.add_local_file(
    ROOT / "kdflow" / "energy_cadence.py", "/work/kdflow/energy_cadence.py"
)
image = image.add_local_file(
    ROOT / "kdflow" / "loss" / "__init__.py", "/work/kdflow/loss/__init__.py"
)
for _name in LOSS_MODULES:
    image = image.add_local_file(
        ROOT / "kdflow" / "loss" / _name, f"/work/kdflow/loss/{_name}"
    )
image = image.add_local_file(
    ROOT / "kdflow" / "fused_logprob.py", "/work/kdflow/fused_logprob.py"
)
for _name in SOURCE_ONLY:
    image = image.add_local_file(
        ROOT / "kdflow" / "algorithms" / _name, f"/work/kdflow/algorithms/{_name}"
    )
for _name in TESTS:
    image = image.add_local_file(
        ROOT / "tests" / "mp_opd" / _name, f"/work/tests/mp_opd/{_name}"
    )

app = modal.App(RUN, image=image)


@app.function(image=image, cpu=2.0, memory=4096, timeout=1800)
def run_guardrail_tests():
    import hashlib

    import torch

    os.environ["KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT"] = "1"

    digests = {}
    for name in MODULES:
        path = Path("/work/kdflow/algorithms") / name
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    print(
        "GUARDRAIL_ENV_JSON="
        + json.dumps(
            {
                "run": RUN,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "module_sha256_16": digests,
            }
        ),
        flush=True,
    )

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *[f"/work/tests/mp_opd/{t}" for t in TESTS],
         "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        cwd="/work",
        env={**os.environ, "PYTHONPATH": "/work"},
    )
    print(completed.stdout[-12000:], flush=True)
    if completed.stderr.strip():
        print("STDERR_TAIL=" + completed.stderr[-4000:], flush=True)
    print(
        "GUARDRAIL_SUMMARY_JSON="
        + json.dumps(
            {
                "exit_code": completed.returncode,
                "passed": completed.stdout.count(" passed"),
                "failed": completed.stdout.count(" failed"),
            }
        ),
        flush=True,
    )
    return completed.returncode


@app.local_entrypoint()
def main():
    run_guardrail_tests.remote()
