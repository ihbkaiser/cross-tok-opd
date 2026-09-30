"""Bounded, single-attempt command runner. Never retries a workload."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--timeout", type=int, default=1800)
p.add_argument("--cwd", default="/workspace/xtoken-native/NeMo-RL")
p.add_argument("--root", default="/workspace/xtoken-native/artifacts")
p.add_argument("command", nargs=argparse.REMAINDER)
args = p.parse_args()
cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
if not cmd:
    p.error("command required")
stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out = Path(args.root) / (stamp + "-" + args.name)
out.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
meta = {"name": args.name, "command": cmd, "cwd": args.cwd,
        "started_at": stamp, "timeout_seconds": args.timeout, "attempt": 1}
(out / "command.json").write_text(json.dumps(meta, indent=2) + "\n")
env = dict(os.environ)
env.update(UV_NO_CACHE="false", UV_CACHE_DIR="/workspace/xtoken-native/uv-cache",
           UV_LINK_MODE="hardlink", PYTHONUNBUFFERED="1", WANDB_MODE="disabled",
           DO_NOT_TRACK="1", HF_HUB_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="4")
for key in ("WANDB_API_KEY", "KAGGLE_KEY", "KAGGLE_API_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
    env.pop(key, None)
print(f"EVIDENCE_DIR={out}", flush=True)
with (out / "stdout.log").open("wb") as log, (out / "gpu.csv").open("wb") as gpu:
    proc = subprocess.Popen(cmd, cwd=args.cwd, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    deadline = started + args.timeout
    timed_out = False
    last_size = 0
    last_growth = started
    while proc.poll() is None:
        tick = time.monotonic()
        probe = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,name,memory.used,utilization.gpu,power.draw,temperature.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, timeout=10)
        gpu.write(probe.stdout)
        gpu.flush()
        size = (out / "stdout.log").stat().st_size
        if size != last_size:
            last_size, last_growth = size, tick
        print(json.dumps({"elapsed_s": round(tick-started), "log_bytes": size,
                          "no_log_growth_s": round(tick-last_growth), "running": True}), flush=True)
        if tick >= deadline:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            break
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
    rc = proc.wait()
meta.update(exit_code=rc, timed_out=timed_out, elapsed_seconds=time.monotonic()-started,
            finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
(out / "result.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta), flush=True)
sys.exit(124 if timed_out else rc)
