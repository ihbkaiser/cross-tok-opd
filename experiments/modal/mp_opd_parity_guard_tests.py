"""Run the MP-OPD parity-guard tests inside the pinned runtime (CPU only).

Local Windows has no torch and installing one is not allowed, so the guard's
regression tests are executed in the same `simct-b200` image the production runs
use. CPU-only and bounded, so it costs cents.

  MODAL_PROFILE=kieusontung8 uvx --from modal modal run \
    experiments/modal/mp_opd_parity_guard_tests.py::main
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import modal

APP_NAME = "simct-parity-guard-tests-20261003"
IMAGE = ("docker.io/codemaivanngu/simct-b200"
         "@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f")
PYTHON = "/opt/venvs/simct-b200/bin/python"
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/opt/overlay")

TARGETS = (
    "tests/mp_opd/test_energy_and_integration.py",
    "tests/mp_opd/test_parity_capture.py",
    "tests/mp_opd/test_forward_position_ids.py",
)

app = modal.App(APP_NAME)
image = (
    modal.Image.from_registry(IMAGE)
    .entrypoint([])
    .env({"PYTHONPATH": "/opt/overlay/experiments/modal/vendor:/opt/overlay",
          "PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(ROOT / "kdflow"), "/opt/overlay/kdflow",
                   ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(str(ROOT / "tests"), "/opt/overlay/tests",
                   ignore=["**/__pycache__/**", "**/*.pyc"])
    # The algorithm registry imports every algorithm and xtoken needs the vendored
    # aligner, so this directory is not optional.
    .add_local_dir(str(ROOT / "experiments/modal"), "/opt/overlay/experiments/modal",
                   ignore=["**/__pycache__/**", "**/*.pyc"])
    # Several tests/mp_opd files import the ladder/queue modules by bare name
    # (`import queue_fixed_span_ladder`), so runai has to be importable for the
    # suite to collect at all.
    .add_local_dir(str(ROOT / "experiments/runai"), "/opt/overlay/experiments/runai",
                   ignore=["**/__pycache__/**", "**/*.pyc"])
)


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0)
def run_tests(commit: str, targets: str) -> dict:
    environment = dict(os.environ)
    environment.update(PATH=f"/opt/venvs/simct-b200/bin:{environment.get('PATH','')}",
                       PYTHONPATH="/opt/overlay/experiments/modal/vendor:/opt/overlay",
                       PYTHONUNBUFFERED="1",
                       KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT="1")
    nvidia = Path("/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia")
    libs = ["/usr/local/cuda/lib64", "/usr/local/nvidia/lib64"]
    libs.extend(str(p) for p in sorted(nvidia.glob("*/lib")))
    environment["LD_LIBRARY_PATH"] = ":".join(libs)
    command = [PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               *[t for t in targets.split(",") if t.strip()]]
    done = subprocess.run(command, cwd="/opt/overlay", env=environment, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1700)
    tail = done.stdout.strip().splitlines()[-12:]
    summary = {"commit": commit, "targets": targets, "returncode": done.returncode,
               "passed": done.returncode == 0, "tail": tail}
    print("PARITY_GUARD_TESTS=" + json.dumps(summary, sort_keys=True), flush=True)
    return summary


@app.local_entrypoint()
def main(targets: str = ",".join(TARGETS)):
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    print("GUARD_TESTS_COMMIT=" + commit + " dirty=" + ("yes" if dirty else "no"), flush=True)
    result = run_tests.remote(commit, targets)
    if not result["passed"]:
        raise SystemExit("guard tests failed rc=" + str(result["returncode"]))
    print("GUARD_TESTS_ALL_PASSED=1", flush=True)
