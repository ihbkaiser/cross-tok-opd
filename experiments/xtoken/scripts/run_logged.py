"""Single-attempt POSIX runner for the fixed, credential-free setup commands."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid


class BestEffortOutput:
    """Console loss is advisory; errors writing durable evidence are not."""

    def __init__(self, stream):
        self.stream = stream
        self.disconnected = False

    def _disconnect(self):
        self.disconnected = True
        # A real TextIOWrapper may retain bytes after EPIPE. Redirect its fd so
        # interpreter shutdown does not turn an otherwise successful run into 120.
        try:
            descriptor = self.stream.fileno()
        except (AttributeError, OSError, ValueError):
            return
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), descriptor)

    def write(self, text):
        if not self.disconnected:
            try:
                self.stream.write(text)
            except BrokenPipeError:
                self._disconnect()
        return len(text)

    def flush(self):
        if not self.disconnected:
            try:
                self.stream.flush()
            except BrokenPipeError:
                self._disconnect()


def workload_environment() -> dict[str, str]:
    # This harness has no authenticated model-download action. Do not forward keys.
    return {key: value for key, value in os.environ.items()
            if not any(marker in key.upper() for marker in
                       ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "KAGGLE_KEY"))}


def stop_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The parent may have exited while descendants remain in the same group.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run_logged(command: list[str], *, cwd: Path, root: Path, name: str,
               timeout: float, env: dict[str, str]) -> tuple[int, Path]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError("invalid evidence name")
    if not command or timeout <= 0:
        raise ValueError("command and positive timeout required")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = root / f"{stamp}-{name}-{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    meta = {"name": name, "command": command, "cwd": str(cwd), "attempt": 1,
            "started_at": stamp, "timeout_seconds": timeout,
            "timed_out": False, "interrupted": False, "telemetry_samples": 0,
            "telemetry_errors": [], "exit_code": None}
    (out / "command.json").write_text(json.dumps(meta, indent=2) + "\n")
    child_env = dict(env, XTOKEN_RUN_DIR=str(out))
    console = BestEffortOutput(sys.stdout)
    print(f"EVIDENCE_DIR={out}", file=console, flush=True)
    started = time.monotonic()
    deadline = started + timeout
    proc = None
    rc = 1
    try:
        with (out / "stdout.log").open("wb") as log, (out / "gpu.csv").open("wb") as gpu:
            proc = subprocess.Popen(command, cwd=cwd, env=child_env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            next_probe = started
            while proc.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    meta["timed_out"] = True
                    stop_process_group(proc)
                    break
                if now >= next_probe:
                    try:
                        probe = subprocess.run(
                            ["nvidia-smi", "--query-gpu=timestamp,name,memory.used,utilization.gpu,power.draw,temperature.gpu",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, timeout=min(2.0, deadline - now), env=child_env,
                        )
                        if probe.returncode == 0 and probe.stdout.strip():
                            gpu.write(probe.stdout)
                            gpu.flush()
                            meta["telemetry_samples"] += 1
                        elif "nonzero_or_empty" not in meta["telemetry_errors"]:
                            meta["telemetry_errors"].append("nonzero_or_empty")
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        if type(exc).__name__ not in meta["telemetry_errors"]:
                            meta["telemetry_errors"].append(type(exc).__name__)
                    progress = {"elapsed_s": round(time.monotonic() - started, 1),
                                "log_bytes": (out / "stdout.log").stat().st_size,
                                "running": proc.poll() is None,
                                "pid": proc.pid, "console_disconnected": console.disconnected}
                    pending = out / "progress.pending.json"
                    pending.write_text(json.dumps(progress) + "\n")
                    pending.replace(out / "progress.json")
                    print(json.dumps(progress), file=console, flush=True)
                    next_probe = time.monotonic() + 15
                try:
                    proc.wait(timeout=max(0.001, min(1.0, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            meta["child_exit_code"] = proc.wait()
            rc = 124 if meta["timed_out"] else (proc.returncode if proc.returncode >= 0 else 128 - proc.returncode)
    except KeyboardInterrupt:
        meta["interrupted"] = True
        rc = 130
    except OSError as exc:
        meta["error_type"] = type(exc).__name__
        rc = 127
    finally:
        if proc is not None:
            # Also remove leftover descendants after normal parent exit.
            stop_process_group(proc)
        meta.update(exit_code=rc, elapsed_seconds=time.monotonic() - started,
                    console_disconnected=console.disconnected,
                    finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        (out / "result.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps({"exit_code": rc, "timed_out": meta["timed_out"],
                          "telemetry_samples": meta["telemetry_samples"]}), file=console, flush=True)
        meta["console_disconnected"] = console.disconnected
        (out / "result.json").write_text(json.dumps(meta, indent=2) + "\n")
    return rc, out
