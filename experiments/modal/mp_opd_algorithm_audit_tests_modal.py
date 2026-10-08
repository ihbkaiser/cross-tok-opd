"""Algorithm regressions on Modal CPU or the pinned B200 runtime.

Set AUDIT_GPU=1 to test the same suite and the B64/M4 actor update on CUDA.
The CUDA variant uses the existing runtime digest with a source-only overlay.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/work")
RUN = "mp-opd-algorithm-audit-20261009-r1"
GPU = os.environ.get("AUDIT_GPU", "0") == "1"
GPU_REQUEST = os.environ.get("AUDIT_GPU_TYPE", "B200,H100").split(",")
RUNTIME = "docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f"
PYTHON = "/opt/venvs/simct-b200/bin/python" if GPU else None

MODULES = (
    "_mp_opd_trust.py",
    "_mp_opd_trust_batch.py",
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
    "test_grass_span.py",
    "test_algorithm_audit_regressions.py",
    "test_trust_batch_replay.py",
    "test_sparse_metric_reduction.py",
    "test_parity_capture.py",
    "test_grass_chunk.py",
    "test_align.py",
    "test_trust.py",
    "test_numeric_fallback.py",
    "test_algorithm_audit_cuda.py",
)

# Production loss code is imported through the lightweight registry.
SOURCE_ONLY = ("mp_opd.py",)

image = (
    modal.Image.from_registry(RUNTIME).entrypoint([]) if GPU else
    modal.Image.debian_slim(python_version="3.12").pip_install("pytest", "numpy")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
)
image = (
    image
    .env({"AUDIT_GPU": "1" if GPU else "0", "AUDIT_GPU_TYPE": ",".join(GPU_REQUEST)})
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

for name in ("training_checkpoint.py", "metric_reduction.py"):
    image = image.add_local_file(ROOT / "kdflow" / name, f"/work/kdflow/{name}")
image = image.add_local_file(ROOT / "kdflow/ray/train/student_actor.py",
                             "/work/kdflow/ray/train/student_actor.py")
image = image.add_local_file(ROOT / "experiments/modal/vendor/xtoken_upstream_token_aligner.py",
                             "/work/experiments/modal/vendor/xtoken_upstream_token_aligner.py")

app = modal.App(RUN, image=image)


@app.function(image=image, cpu=2.0, memory=8192 if GPU else 4096,
              gpu=GPU_REQUEST if GPU else None, timeout=900 if GPU else 1800)
def run_guardrail_tests():
    import hashlib

    os.environ["KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT"] = "1"

    digests = {}
    for name in MODULES:
        path = Path("/work/kdflow/algorithms") / name
        digests["kdflow/algorithms/"+name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("kdflow/algorithms/mp_opd.py", "kdflow/ray/train/student_actor.py",
                 "kdflow/metric_reduction.py", "kdflow/arguments/distillation_args.py",
                 "kdflow/training_checkpoint.py", "kdflow/fused_logprob.py",
                 *["tests/mp_opd/"+t for t in TESTS]):
        digests[name] = hashlib.sha256((Path("/work")/name).read_bytes()).hexdigest()
    print(
        "GUARDRAIL_ENV_JSON="
        + json.dumps(
            {
                "run": RUN,
                "python": sys.version.split()[0],
                "runtime_digest": RUNTIME if GPU else "torch-cpu",
                "device": "cuda" if GPU else "cpu",
                "source_sha256": digests,
            }
        ),
        flush=True,
    )

    environment = {**os.environ, "PYTHONPATH": "/work/experiments/modal/vendor:/work",
                   "AUDIT_TEST_DEVICE": "cuda" if GPU else "cpu",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    if GPU:
        environment["LD_LIBRARY_PATH"] = ":".join([
            "/usr/local/cuda/lib64", "/usr/local/nvidia/lib64",
            *[str(p) for p in Path("/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia").glob("*/lib")],
        ])
    probe = subprocess.run([PYTHON or sys.executable, "-c",
        "import json,sys,torch; print('RUNTIME_ENV_JSON='+json.dumps(dict(python=sys.version.split()[0],torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)))"],
        text=True, capture_output=True, env=environment)
    print(probe.stdout, flush=True)
    if probe.returncode:
        print(probe.stderr, flush=True)
        return probe.returncode

    completed = subprocess.run(
        [PYTHON or sys.executable, "-m", "pytest", *[f"/work/tests/mp_opd/{t}" for t in TESTS],
         "-q", *( ["-s"] if GPU else [] ), "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        cwd="/work",
        env=environment,
    )
    print(completed.stdout, flush=True)
    if completed.stderr.strip():
        print("STDERR_TAIL=" + completed.stderr[-4000:], flush=True)
    print(
        "GUARDRAIL_SUMMARY_JSON="
        + json.dumps(
            {
                "exit_code": completed.returncode,
                **{key: int(match.group(1)) if (match := re.search(r"(\d+) " + key, completed.stdout)) else 0
                   for key in ("passed", "failed", "skipped", "error")},
            }
        ),
        flush=True,
    )
    return completed.returncode


@app.local_entrypoint()
def main():
    code = run_guardrail_tests.remote()
    if code:
        raise SystemExit(code)
