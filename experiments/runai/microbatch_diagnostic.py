"""Bounded (micro_B, micro_M) capacity diagnostic against the production env builder.

Why this exists: the campaign pins micro_B=1 because the first real micro4 run OOM'd in
the second-order inner gradient (168.29 GiB allocated) and micro2 was never verified on
real data. The launcher admits (2,4) for limits 1-30 only. This driver reuses
queue_full_alternating.run_command - the same function the campaign uses - so the
diagnostic exercises the real serving contract, the real offload flag and the real
wrapper, and changes exactly one thing: the microbatch pair.

Usage (company host, one GPU at a time):

    python3.12 experiments/runai/microbatch_diagnostic.py <arm> <micro_B> <micro_M> <updates> [plain|expandable]

It writes <case>/diagnostics/<arm>-<stamp>/ plus sibling <arm>-<stamp>.attempt-*.log files
and prints DIAG_PEAK / DIAG_SUMMARY, or DIAG_FAILED with the log tail on OOM.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
PARENT = Path(os.environ.get("DIAG_PARENT", "/workspace/storage-shared/nlp/tungks/borrow8-8MgodXcM"))
for entry in (str(SRC), str(SRC / "experiments/runai"), str(SRC / "experiments/modal/vendor")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import queue_full_alternating as F          # noqa: E402
import queue_split_alternating as Q         # noqa: E402

SERVING = {"deterministic": True, "random_seed": 42,
           "attention_backend": "triton", "disable_radix_cache": True}
OPERATIONAL_KEYS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "CUDA_LAUNCH_BLOCKING",
                    "NCCL_CUMEM_HOST_ENABLE", "NCCL_IB_DISABLE", "NCCL_NET_GDR_LEVEL",
                    "NCCL_P2P_DISABLE", "OMP_NUM_THREADS")

# Metrics lines look like: ..., gpu_peak_memory_allocated_gib: 148.45, gpu_peak_memory_reserved_gib: 172.53, ...
PEAK = re.compile(r"gpu_peak_memory_allocated_gib[^0-9]+([0-9.]+)"
                  r".*?gpu_peak_memory_reserved_gib[^0-9]+([0-9.]+)")
TRACE = re.compile(r"FULL_META_MEMORY (\{.*\})")


def diag_config(micro_b: int, micro_m: int, variant: str = "ALT-main-s43") -> dict:
    """The campaign run recipe with only the microbatch pair replaced."""
    config = dict(next(r for r in F.configurations() if r["id"] == variant))
    config["micro_B"] = int(micro_b)
    config["micro_M"] = int(micro_m)
    return config


def case_root() -> Path:
    if (PARENT / "campaign.json").is_file():
        return PARENT
    head = subprocess.check_output(["git", "-C", str(SRC), "rev-parse", "--short", "HEAD"],
                                   text=True).strip()
    return PARENT / ("microbatch-diag-" + head)


def ensure_case(case: Path) -> None:
    if (case / "campaign.json").is_file():
        return
    try:
        Q.initialize_case(case, qualification_policy="advisory", extra_gpu_count=4,
                          qualification_timeout=7200)
    except FileNotFoundError as error:
        raise SystemExit(
            "DIAG_CASE_UNAVAILABLE " + str(error.filename) + ": case templates live under "
            + ": the case template is missing; run this on the host that owns the shared workspace"
        )
    path = case / "campaign.json"
    campaign = json.loads(path.read_text())
    campaign["offload_adam_moments"] = True
    campaign["serving"] = dict(SERVING)
    path.write_text(json.dumps(campaign, indent=2) + "\n")
    print("CASE_READY", case, flush=True)


def operational_environment() -> dict:
    return {key: os.environ[key] for key in OPERATIONAL_KEYS if os.environ.get(key)}


def summarise(log: Path, run: Path) -> dict:
    text = log.read_text(errors="replace") if log.is_file() else ""
    peak = PEAK.search(text)
    traces = [json.loads(match) for match in TRACE.findall(text)]
    record = {
        "log": str(log),
        "directory": str(run),
        "gpu_peak_memory_allocated_gib": float(peak.group(1)) if peak else None,
        "gpu_peak_memory_reserved_gib": float(peak.group(2)) if peak else None,
        "full_meta_trace_lines": len(traces),
        "full_meta_arithmetic_peak_gib": round(max(t["peak_allocated_bytes"] for t in traces) / 2**30, 3)
        if traces else None,
        "launch_config": (run / "launch-config.json").is_file(),
    }
    summary_path = run / "checkpoint" / "run-summary.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text())
        for key in ("optimizer_updates", "energy_updates", "status", "session_fit_seconds"):
            record[key] = payload.get(key)
    return record


def main() -> int:
    if len(sys.argv) not in (5, 6):
        print(__doc__)
        return 2
    arm, micro_b, micro_m, updates = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                                      int(sys.argv[4]))
    allocator = sys.argv[5] if len(sys.argv) == 6 else "plain"
    if not 1 <= updates <= 30:
        raise ValueError("diagnostics are bounded to 1-30 updates by the launcher contract")
    if allocator == "expandable":
        # Operational setting, not a hyperparameter: recorded in launch-config.json and in
        # the sidecar so this arm cannot be mistaken for the baseline it is not.
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    elif allocator != "plain":
        raise ValueError("allocator must be plain or expandable")

    case = case_root()
    ensure_case(case)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = arm + "-" + stamp
    out = case / "diagnostics" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    config = diag_config(micro_b, micro_m)
    print("DIAG", json.dumps({"arm": arm, "case": str(case), "out": str(out),
                              "micro_B": config["micro_B"], "micro_M": config["micro_M"],
                              "updates": updates, "allocator": allocator,
                              "energy_every": config["energy_every"]}), flush=True)
    sidecar = out.parent / (name + ".operational-environment.json")
    if os.environ.get("DIAG_PLAN_ONLY") == "1":
        # CPU-only: proves the case, the recipe override and the target path without a GPU.
        print("DIAG_PLAN", json.dumps({"case": str(case), "out": str(out), "updates": updates,
                                       "allocator": allocator,
                                       "config": diag_config(micro_b, micro_m)},
                                      sort_keys=True), flush=True)
        print("DIAG_PLAN_READY", out, flush=True)
        return 0
    sidecar.write_text(json.dumps(operational_environment(), indent=2, sort_keys=True) + "\n")
    status = "completed"
    try:
        F.run_command(case, config, out, limit=updates)
    except Exception as error:                       # OOM and friends: report, do not retry
        status = "failed:" + type(error).__name__
        print("DIAG_FAILED", status, str(error)[:400], flush=True)
    log = max(out.parent.glob(name + ".attempt-*.log"), default=None,
              key=lambda path: path.stat().st_mtime)
    print("DIAG_ARTIFACT", sidecar, sidecar.is_file(), flush=True)
    print("DIAG_ARTIFACT", out / "launch-config.json", (out / "launch-config.json").is_file(),
          flush=True)
    if log is not None:
        record = summarise(log, out)
        record["status"] = status
        print("DIAG_SUMMARY", json.dumps(record, sort_keys=True), flush=True)
        print("DIAG_LOG_TAIL", flush=True)
        print("\n".join(log.read_text(errors="replace").splitlines()[-25:]), flush=True)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
