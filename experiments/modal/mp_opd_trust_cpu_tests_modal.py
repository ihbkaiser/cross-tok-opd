"""TRUST unit tests on Modal, on CPU.

Purpose is narrow: run `tests/mp_opd/test_trust.py` on a real torch, not on
numpy. The calibration math was verified against a NumPy transcription of the
method note locally (kappa never exceeds one over 4000 random PSD problems,
the aligned pair gives lambda 1.8, opposing and empty sides give zero), and the
Gram-correspondence assertions need the real `chunk_head_gram` routine. Those
cannot be settled without torch, and shipping the tests to a node to get them
would cost a GPU slot nobody needs for arithmetic on small matrices.

**CPU, deliberately.** TRUST is fp32 coefficient arithmetic on a handful of
atoms plus one small exact Gram. Nothing here touches CUDA, so asking for a
B200 would spend GPU money to produce a result identical to this one. The GPU
claim is a separate question and is not answered by this run.

Cost control: one CPU container, two cores, no volumes, no model download. The
image build pulls a CPU-only torch, so the nvidia-* wheels never enter it.

Source layout matters. `test_trust.py` and `test_grass_chunk.py` locate the
sources they audit with `parents[2] / "kdflow" / ...`, so the files have to
land at /work/tests/mp_opd/ and /work/kdflow/algorithms/ for those paths to
resolve. Copying them one level shallower would make the source-scanning tests
read a missing file and pass or fail for the wrong reason.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/work")
RUN = "mp-opd-trust-cpu-tests-20261007-r1"

# TRUST is what this run is for. GRASS-Chunk is dragged along because the same
# edit added "trust_r"/"trust_b" to the mode choices in distillation_args and
# rewired the mode sets in mp_opd, and test_grass_chunk audits those files by
# reading them as source text - including the invariant that every dispatched
# mode is advertised.
MODULES = (
    "_mp_opd_trust.py",
    "_mp_opd_grass_chunk.py",
    "_mp_opd_grass_span.py",
    "_mp_opd_credit.py",
    "_mp_opd_atoms.py",
)
TESTS = ("test_trust.py", "test_grass_chunk.py")

# mp_opd.py is shipped but never imported. The suite reads it as source text to
# prove that every mode mp_opd dispatches is advertised to the user, which is
# the invariant that would catch "trust_r" landing in one of those two lists
# and not the other.
SOURCE_ONLY = ("mp_opd.py",)

# pytest and numpy from PyPI, torch from the CPU index so the build never pulls
# the multi-gigabyte nvidia-* dependency set that the default PyPI wheel drags
# in. Every path is an explicit add_local_file: add_local_dir would ship
# unrelated algorithms whose imports this image cannot satisfy.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pytest", "numpy")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_file(ROOT / "kdflow" / "__init__.py", "/work/kdflow/__init__.py")
    .add_local_file(
        ROOT / "kdflow" / "algorithms" / "__init__.py",
        "/work/kdflow/algorithms/__init__.py",
    )
    # Shipped for read_text only. Its package __init__ is NOT shipped, so
    # anything that tries to import it fails loudly instead of against a stub.
    .add_local_file(
        ROOT / "kdflow" / "arguments" / "distillation_args.py",
        "/work/kdflow/arguments/distillation_args.py",
    )
)

for _name in MODULES:
    image = image.add_local_file(
        ROOT / "kdflow" / "algorithms" / _name, f"/work/kdflow/algorithms/{_name}"
    )
# _mp_opd_grass_span imports this one transitively; it is pure Python over
# torch, so it costs one file rather than a compiled extension.
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
def run_trust_tests():
    import hashlib

    import torch

    # kdflow/algorithms/__init__.py imports every non-underscore module in the
    # directory unless this is set, which would pull mp_opd and the whole
    # training stack. The suites here do not need it.
    os.environ["KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT"] = "1"

    digests = {}
    for name in MODULES:
        path = Path("/work/kdflow/algorithms") / name
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    print(
        "TRUST_ENV_JSON="
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
        "TRUST_SUMMARY_JSON="
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
    run_trust_tests.remote()
