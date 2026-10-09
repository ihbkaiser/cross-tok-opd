"""Plan, then explicitly launch the audited algorithms on the user's GPU 3.

Read asset paths from an existing campaign/launch manifest. The reference run is
never resumed or modified. A node-specific runtime wrapper must be supplied.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MODES = ("grass", "grass_chunk", "grass_chunk_temporal", "align", "trust_r", "trust_b")


def asset_paths(reference):
    data = json.loads(Path(reference).read_text())
    if "options" in data:
        options = data["options"]
        return dict(student=options["student_name_or_path"],
                    teacher=options["teacher_name_or_path"],
                    dataset=options["train_dataset_path"])
    return {key: data[key] for key in ("student", "teacher", "dataset")}


def require_idle_gpu():
    def query(*args):
        return subprocess.check_output(["nvidia-smi", *args], text=True)
    gpu = next(csv.reader(query("--id=3", "--query-gpu=uuid,memory.used",
                                "--format=csv,noheader,nounits").splitlines()))
    uuid, used = gpu[0].strip(), int(gpu[1])
    holders = [row[1].strip() for row in csv.reader(query(
        "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader").splitlines())
        if row and row[0].strip() == uuid]
    if holders or used >= 2000:
        raise RuntimeError(f"GPU 3 is occupied: uuid={uuid}, memory_mib={used}, pids={holders}")
    # Sleeping engines may release their CUDA context while retaining the GPU
    # assignment. Inspect only their CUDA selector, never print environment data.
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc/"cmdline").read_bytes()
            if not any(name in command for name in (b"sglang", b"vllm", b"run_single_gpu.py")):
                continue
            env = (proc/"environ").read_bytes().split(b"\0")
            selectors = next((v.split(b"=", 1)[1].decode().split(",") for v in env
                              if v.startswith(b"CUDA_VISIBLE_DEVICES=")), [])
            if "3" in selectors or uuid in selectors:
                raise RuntimeError(f"GPU 3 is assigned to a serving/training process: pid={proc.name}")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return uuid


def build_plan(args):
    paths = asset_paths(args.reference)
    for name, raw in paths.items():
        if not Path(raw).exists():
            raise FileNotFoundError(f"{name} asset does not exist: {raw}")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory; refusing to resume: {output}")
    if output.with_name(output.name+".audit-launch.json").exists():
        raise FileExistsError(f"An audit launch receipt already exists for: {output}")
    wrapper = Path(args.runtime_wrapper).resolve()
    if not wrapper.is_file():
        raise FileNotFoundError(f"Node runtime wrapper does not exist: {wrapper}")
    receipt_path = ROOT/"ALGORITHM_AUDIT_RECEIPT.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("Use the qualified source bundle with ALGORITHM_AUDIT_RECEIPT.json")
    receipt = json.loads(receipt_path.read_text())
    source_id = receipt.get("commit") or "uncommitted-tree:"+receipt["staged_tree"]
    for relative, expected in receipt["source_sha256"].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Qualified source has changed: {relative}")
    environment = {k: v for k, v in os.environ.items()
                   if not k.startswith("MP_") or k in {"MP_RUNTIME_DIR", "MP_SHARED_ROOT"}}
    temporal = args.mode == "grass_chunk_temporal"
    chunk_source = getattr(args, "chunk_source", None) or ("run" if temporal else "xtoken")
    run_length = getattr(args, "run_length", 2)
    start_step = getattr(args, "temporal_start_step", 20)
    end_step = getattr(args, "temporal_end_step", 160)
    if run_length < 1 or not 0 <= start_step < end_step:
        raise ValueError("Require positive run_length and 0 <= temporal start < end")
    settings = dict(
        CUDA_VISIBLE_DEVICES="3", MP_STUDENT_PATH=paths["student"],
        MP_TEACHER_PATH=paths["teacher"], MP_DATASET_PATH=paths["dataset"],
        MP_SEED=str(args.seed), MP_PARTITION_SEED="43", MP_ALGORITHM="mp_opd",
        MP_MICRO_TRAIN_BATCH_SIZE="4", MP_TRUST_SCOPE="batch",
        MP_GRASS_CHUNK_SOURCE=chunk_source, MP_GRASS_CHUNK_STRADDLE="singleton",
        MP_GRASS_CHUNK_RUN_LENGTH=str(run_length),
        MP_GRASS_CHUNK_TEMPORAL_START_STEP=str(start_step),
        MP_GRASS_CHUNK_TEMPORAL_END_STEP=str(end_step),
        MP_ATTN_IMPLEMENTATION="eager", MP_MAX_SPAN_LENGTH="2",
        MP_RESUME="0", MP_TENSORBOARD="1", MP_EVAL_ON_CKPT="0",
        MP_RAY_TMP="/tmp/af"+str(os.getpid()),
        MP_SOURCE_COMMIT=source_id,
        MP_SOURCE_DIRTY="" if receipt.get("commit") else "qualified staged source; GPG commit blocked",
        KDFLOW_ROLLOUT_PORT_BASE="18000", KDFLOW_ROUTER_PORT_BASE="26000",
        KDFLOW_ROUTER_PROMETHEUS_PORT="23000",
        PYTHONPATH=f"{ROOT}/experiments/modal/vendor:{ROOT}",
    )
    environment.update(settings)
    command = ["bash", str(wrapper), str(ROOT/"experiments/runai/run_single_gpu.py"),
               args.mode, str(args.updates), str(output)]
    plan = dict(node="embed-8b-training-v2-0-1", gpu=3, mode=args.mode,
                updates=args.updates, scheduler_horizon=312, batch=64, microbatch=4,
                assets=paths, source_commit=receipt.get("commit"),
                source_tree=receipt.get("staged_tree"), output=str(output),
                reference=str(Path(args.reference).resolve()),
                runtime_wrapper=str(wrapper), command=command,
                trust_scope="batch", chunk_source=chunk_source, chunk_run_length=run_length,
                temporal_pooling=dict(start_step=start_step, end_step=end_step) if temporal else None,
                seed=args.seed,
                tensorboard=True, evaluation="external", resume=False)
    return plan, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Existing launch-config.json or campaign.json")
    parser.add_argument("--runtime-wrapper", required=True, help="Existing, qualified wrapper for this node")
    parser.add_argument("--output", required=True, help="New run directory")
    parser.add_argument("--mode", choices=MODES, default="trust_b")
    parser.add_argument("--updates", type=int, choices=range(1, 313), default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-source", choices=("xtoken", "run"), default=None)
    parser.add_argument("--run-length", type=int, default=2)
    parser.add_argument("--temporal-start-step", type=int, default=20)
    parser.add_argument("--temporal-end-step", type=int, default=160)
    parser.add_argument("--execute", action="store_true", help="Without this flag, only print the plan")
    args = parser.parse_args()
    plan, environment = build_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return 0
    uuid = require_idle_gpu()
    import fcntl
    with open(f"/tmp/simct-algorithm-audit-{uuid}.lock", "w") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_idle_gpu()
        record = Path(args.output).resolve().with_name(Path(args.output).name+".audit-launch.json")
        record.parent.mkdir(parents=True, exist_ok=True)
        plan.update(gpu_uuid=uuid, started=time.time())
        record.write_text(json.dumps(plan, indent=2)+"\n")
        with record.with_suffix(".log").open("x") as stream:
            result = subprocess.run(plan["command"], cwd=ROOT, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT)
        plan.update(returncode=result.returncode, finished=time.time())
        record.write_text(json.dumps(plan, indent=2)+"\n")
        print("AUDIT_LAUNCH_RECEIPT="+str(record), flush=True)
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
