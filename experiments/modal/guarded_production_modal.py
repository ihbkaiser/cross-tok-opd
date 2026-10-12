"""Bounded real-SFT ALIGN-NEW/TRUST-P canary using the node shell launcher.

Source is a committed Git bundle. Assets are persistent remote volumes; the
existing public warmup range importer verifies checkpoint SHA256 and ZIP CRC.
No runtime dependencies, model weights or data are installed on the laptop.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import modal

RUN = os.environ.get("GUARDED_CANARY_RUN", "guarded-production-20261012-r1")
COMMIT = os.environ["GUARDED_CANARY_SOURCE_COMMIT"]
BUNDLE = Path(os.environ.get("GUARDED_CANARY_BUNDLE", "/tmp/unused"))
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/work")
IMAGE = "docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f"
PY = "/opt/venvs/simct-b200/bin/python"


def validate_local_source(root, bundle, expected, *, windows_checkout=False):
    """Reject stale/mistyped source references before creating any paid app."""
    def git(*args):
        # WSL Git otherwise reports the Windows-native CRLF checkout as dirty.
        # Match its confirmed Windows Git EOL policy for this command only;
        # real edits still fail and no repository/global config is modified.
        prefix = ["git", "-c", "core.autocrlf=true"] if windows_checkout else ["git"]
        return subprocess.check_output([*prefix, "-C", str(root), *args], text=True).strip()
    if git("rev-parse", "HEAD") != expected:
        raise ValueError("GUARDED_CANARY_SOURCE_COMMIT differs from local HEAD")
    # Compare tracked content (including staged changes) to HEAD. Git status
    # can retain an EOL-only dirty stat-cache entry after another Git client.
    if git("diff", "--name-only", "HEAD", "--"):
        raise ValueError("Commit tracked source changes before submitting")
    if git("bundle", "list-heads", str(bundle)) != expected+" refs/heads/vdt/ops/b200-portable":
        raise ValueError("Canary bundle ref differs from requested source commit")


if modal.is_local():
    validate_local_source(ROOT, BUNDLE, COMMIT,
                          windows_checkout=os.environ.get("GUARDED_CANARY_WINDOWS_CHECKOUT") == "1")
app = modal.App(RUN)
image = (modal.Image.from_registry(IMAGE).entrypoint([])
         .env({"GUARDED_CANARY_SOURCE_COMMIT": COMMIT})
         .add_local_file(str(BUNDLE), "/work/source.bundle")
         .add_local_file(str(ROOT/"experiments/modal/guarded_production_worker.py"), "/work/worker.py"))
assets = modal.Volume.from_name("guarded-warmup-assets-20261012-v1", create_if_missing=True)
results = modal.Volume.from_name(RUN, create_if_missing=True)
volumes = {"/warmup": assets, "/results": results,
           "/qwen_assets": modal.Volume.from_name("simct-qwen7b-gemma2-assets-20260915"),
           "/phi_assets": modal.Volume.from_name("simct-phi-gemma-assets")}


def execute(stage, seconds):
    path = Path("/results")/(stage+".log")
    if path.exists():
        raise RuntimeError("Attempt exists; read its evidence before selecting a new run ID")
    env = dict(os.environ)
    env.update(PATH=str(Path(PY).parent)+":"+env.get("PATH", ""),
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
               KDFLOW_TRUST_REMOTE_CODE="0", HF_HOME="/warmup/hf")
    libs = ["/usr/local/cuda/lib64", "/usr/local/nvidia/lib64"]
    libs += [str(p) for p in Path("/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia").glob("*/lib")]
    env["LD_LIBRARY_PATH"] = ":".join(libs)
    start = time.monotonic(); pos = 0; error = None; process = None
    try:
        with path.open("x") as output:
            process = subprocess.Popen([PY, "-u", "/work/worker.py", stage], env=env,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                if time.monotonic()-start > seconds:
                    raise TimeoutError("Bounded production qualification timeout")
                with path.open() as reader:
                    reader.seek(pos); chunk = reader.read(); pos = reader.tell()
                if chunk: print(chunk, flush=True)
                results.commit(); time.sleep(5)
    except Exception as exc:
        error = type(exc).__name__+": "+str(exc)
    finally:
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        receipt = {"stage": stage, "exit_code": process.returncode if process else None,
                   "error": error, "elapsed_seconds": time.monotonic()-start,
                   "runtime_digest": IMAGE, "source_commit": COMMIT,
                   "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (Path("/results")/(stage+"-receipt.json")).write_text(json.dumps(receipt, indent=2))
        assets.commit(); results.commit()
    with path.open() as reader:
        reader.seek(pos); print(reader.read(), flush=True)
    return receipt


@app.function(image=image, cpu=4, memory=16384, timeout=1260, retries=0,
              max_containers=1, volumes=volumes)
def prepare(): return execute("prepare", 1200)


@app.function(image=image, gpu="B200", cpu=16, memory=131072, timeout=960,
              retries=0, max_containers=1, volumes=volumes)
def canary(stage): return execute(stage, 900)


@app.local_entrypoint()
def main(stage: str = "prepare"):
    if stage not in {"prepare", "qwen-align_new", "qwen-trust_p", "phi-align_new", "phi-trust_p"}:
        raise ValueError("Unknown canary stage")
    receipt = prepare.remote() if stage == "prepare" else canary.remote(stage)
    print("GUARDED_PRODUCTION_RECEIPT_JSON="+json.dumps(receipt), flush=True)
    if receipt["exit_code"] != 0 or receipt["error"]:
        raise RuntimeError("Canary failed; read persisted logs before any retry")
