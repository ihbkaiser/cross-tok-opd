"""Bounded production pipeline-fix qualification on the existing pinned runtime."""
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time
import modal

RUN = os.environ.get("PIPELINE_FIX_RUN", "pipeline-fix-20261011-r5")
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/opt/pipeline_fix")
PY = "/opt/venvs/simct-b200/bin/python"
IMAGE = "docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f"
app = modal.App(RUN)
image = modal.Image.from_registry(IMAGE).entrypoint([])
for directory in ("kdflow", "tests"):
    image = image.add_local_dir(str(ROOT / directory), "/opt/pipeline_fix/" + directory,
                                ignore=["**/__pycache__/**", "**/*.pyc"])
image = image.add_local_file(str(ROOT / "experiments/modal/pipeline_fix_worker.py"), "/opt/pipeline_fix/pipeline_fix_worker.py")
image = image.add_local_file(str(ROOT / "experiments/runai/launch_algorithm_audit_fix.py"), "/opt/pipeline_fix/experiments/runai/launch_algorithm_audit_fix.py")
for name in ("run_single_gpu.py", "run_single_gpu.sh", "ray_socket_layout.py"):
    image = image.add_local_file(str(ROOT / "experiments/runai" / name), "/opt/pipeline_fix/experiments/runai/" + name)
image = image.add_local_file(str(ROOT / "experiments/modal/mp_opd_phi_gemma_50.py"), "/opt/pipeline_fix/experiments/modal/mp_opd_phi_gemma_50.py")
image = image.add_local_dir(str(ROOT / "experiments/modal/vendor"), "/opt/pipeline_fix/experiments/modal/vendor",
                            ignore=["**/__pycache__/**", "**/*.pyc"])
results = modal.Volume.from_name(RUN, create_if_missing=True)
volumes = {"/results": results, "/warmup": modal.Volume.from_name("phi-qwen-warmup-assets-20261011-v1"),
           "/prior": modal.Volume.from_name("phi-qwen-warmup-debug-20261011-r1"),
           "/phi_teacher": modal.Volume.from_name("phi-gemma-debug-20261010-r1"),
           "/qwen_teacher": modal.Volume.from_name("qwen-pipeline-teacher-20261011-v1")}


def execute(stage, seconds):
    log = Path("/results") / (stage + ".log")
    if log.exists(): raise RuntimeError("Attempt exists; inspect it before selecting a new run ID")
    env = dict(os.environ)
    env.update(PYTHONPATH="/opt/pipeline_fix/experiments/modal/vendor:/opt/pipeline_fix", PYTHONUNBUFFERED="1", PATH=str(Path(PY).parent)+":"+env.get("PATH", ""),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4", KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT="1", KDFLOW_TRUST_REMOTE_CODE="0")
    libs = ["/usr/local/cuda/lib64", "/usr/local/nvidia/lib64"]
    libs += [str(p) for p in Path("/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia").glob("*/lib")]
    env["LD_LIBRARY_PATH"] = ":".join(libs)
    start = time.monotonic(); pos = 0; error = None; proc = None
    try:
        with log.open("x") as f:
            proc = subprocess.Popen([PY, "-u", "/opt/pipeline_fix/pipeline_fix_worker.py", stage], cwd="/opt/pipeline_fix",
                env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                if time.monotonic()-start > seconds: raise TimeoutError("Bounded pipeline qualification timeout")
                with log.open() as reader: reader.seek(pos); chunk = reader.read(); pos = reader.tell()
                if chunk: print(chunk, flush=True)
                results.commit(); time.sleep(5)
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        receipt = {"stage": stage, "exit_code": proc.returncode if proc else None, "error": error,
                   "elapsed_seconds": time.monotonic()-start, "image": IMAGE}
        source = Path("/opt/pipeline_fix")
        receipt["source_sha256"] = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ("kdflow", "tests") for p in sorted((source / directory).rglob("*.py"))}
        receipt["source_sha256"]["pipeline_fix_worker.py"] = hashlib.sha256((source/"pipeline_fix_worker.py").read_bytes()).hexdigest()
        (Path("/results") / (stage+"-receipt.json")).write_text(json.dumps(receipt))
        results.commit()
    with log.open() as reader: reader.seek(pos); print(reader.read(), flush=True)
    return receipt


@app.function(image=image, cpu=4, memory=16384, timeout=480, retries=0, volumes=volumes)
def cpu(stage="cpu"): return execute(stage, 420)


@app.function(image=image, gpu="B200", cpu=8, memory=65536, timeout=960, retries=0, max_containers=1, volumes=volumes)
def gpu(arm): return execute(arm, 900)


@app.local_entrypoint()
def main(stage: str = "cpu"):
    if stage not in {"cpu", "cache", "cache-fixed", "phi", "qwen"}: raise ValueError("unknown stage")
    receipt = cpu.remote(stage) if stage in {"cpu", "cache", "cache-fixed"} else gpu.remote(stage)
    print("PIPELINE_FIX_RECEIPT=" + json.dumps({k: v for k, v in receipt.items() if k != "source_sha256"}))
    if receipt["exit_code"] != 0 or receipt["error"]: raise RuntimeError("Read persisted failure before retrying")
