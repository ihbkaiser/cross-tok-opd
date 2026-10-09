#!/usr/bin/env python3
"""Gen/score worker for the mathplus extension.

Mirrors the eval_queue conventions (plan/state/cells layout, SGLang server
per checkpoint via python-b200-host.sh, per-item stable request seeds) but
validates against its own profile/source, so the main 4-bench queues are
untouched. New file only.
"""
import argparse
import contextlib
import fcntl
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mathplus_data as D
import contract_eval as E  # read-only reuse; never modified by this module

MODEL_ID = "eval-mathplus"
CONTEXT_LENGTH = 8192


class Deadline(Exception):
    pass


def atomic_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


@contextlib.contextmanager
def locked(path, blocking=True):
    path = Path(path)
    fh = path.open("w")
    flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
    try:
        fcntl.flock(fh.fileno(), flags)
    except OSError:
        fh.close()
        yield None
        return
    try:
        yield fh
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def digest(obj):
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def stop_group(process):
    try:
        os.killpg(os.getpgid(process.pid), 15)
    except (ProcessLookupError, PermissionError):
        pass


def grade_math(text, gold):
    from evaluation import (extract_boxed_answer, extract_number_from_answer,
                            strip_thinking_content, try_parse_number)
    stripped, _, _ = strip_thinking_content(text or "")
    pred = extract_boxed_answer(stripped)
    pred_num = try_parse_number(pred) if pred else None
    gold_num = try_parse_number(str(gold))
    if pred_num is not None and gold_num is not None:
        if abs(pred_num - gold_num) < 1e-6:
            return True, pred or str(pred_num)
    if not pred:
        fb = extract_number_from_answer(stripped)
        if fb is not None and gold_num is not None and abs(fb - gold_num) < 1e-6:
            return True, str(fb)
    return False, pred


def grade_gpqa(text, gold):
    pred = D.extract_boxed_letter(text or "")
    return (pred == gold), pred


def cmd_prepare(a):
    root = a.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cache = a.shared / "simct-eval-data" / "mathplus-cache"
    cache.mkdir(parents=True, exist_ok=True)
    data = {}
    benches = a.benches.split(",") if a.benches else list(D.BENCHES)
    for bench in benches:
        if bench not in D.BENCHES:
            raise ValueError(f"unknown bench {bench}")
        items, source = D.acquire(bench, cache, a.proxy)
        path = root / (bench + ".json")
        path.write_text(json.dumps(
            {"benchmark": bench, "items": items, "source": source,
             "profile": D.PROFILE}, indent=2), encoding="utf-8")
        data[bench] = {"path": str(path), "sha256": D.file_hash(path),
                       "count": len(items), "source": source}
        print(f"PREPARED {bench} {len(items)}", flush=True)
    jobs = []
    for spec in a.checkpoint:
        ckpt = Path(spec).resolve()
        ident = E.checkpoint_identity(str(ckpt))
        jobs.append({"id": f"mathplus-{ckpt.parent.parent.name}-{ckpt.name}",
                     "checkpoint": ident, "tier": 0})
    plan = {"schema": "mathplus-queue-v1", "profile": D.PROFILE,
            "seeds": list(D.SEEDS), "data": data, "jobs": jobs,
            "hours": a.hours, "temperature": a.temperature,
            "top_p": a.top_p, "n": a.n, "protocol": a.protocol,
            "source": D.script_hashes(),
            "protocol_detail": {
                "context_length": CONTEXT_LENGTH, "n": 1, "caps": D.CAPS,
                "math": "boxed numeric, QWEN_MATH_SYSTEM_PROMPT (simct) "
                        "or Question/Answer (harness)",
                "gpqa": "MCQ A-D shuffled seed 42, boxed letter",
                "code_execution": "none (exact match only)",
                "scope": "exploratory extension beside the 4-bench contract"}}
    (root / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("PLAN_READY=" + str(root / "plan.json"), flush=True)


def read_state(root, plan_hash, plan):
    for name in ("generation-state.json", "state.json"):
        path = root / name
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            if state["plan_sha256"] != plan_hash:
                raise ValueError("queue plan changed")
            return state, path
    start = time.time()
    state = {"plan_sha256": plan_hash, "started": start,
             "deadline": start + plan["hours"] * 3600, "jobs": {}, "durations": []}
    return state, root / "generation-state.json"


def launch_server(ckpt_path, gpu, port, deadline):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
               PYTHONUNBUFFERED="1",
               PYTHONPATH=str(HERE.parents[1] / "experiments/modal/vendor") + ":"
               + str(HERE.parents[1]))
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy",
                 "https_proxy", "all_proxy", "WANDB_API_KEY", "HF_TOKEN"):
        env.pop(name, None)
    host_sh = HERE.parents[1] / "experiments/runai/python-b200-host.sh"
    command = ["bash", str(host_sh), "-m", "sglang.launch_server",
               "--model-path", ckpt_path, "--served-model-name", MODEL_ID,
               "--host", "127.0.0.1", "--port", str(port),
               "--tp-size", "1", "--mem-fraction-static", "0.8",
               "--context-length", str(CONTEXT_LENGTH),
               "--attention-backend", "triton", "--disable-cuda-graph",
               "--random-seed", "42"]
    log = open(f"/tmp/mathplus-gpu{gpu}-server.log", "ab")
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    watchdog = threading.Timer(max(0., deadline - time.time()),
                               stop_group, args=(process,))
    watchdog.daemon = True
    watchdog.start()
    try:
        startup = min(deadline, time.time() + 900)
        while True:
            if time.time() >= deadline:
                raise Deadline("wall budget")
            if time.time() >= startup:
                raise RuntimeError("server readiness exceeded 900s")
            if process.poll() is not None:
                raise RuntimeError("SGLang exited")
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(f"http://127.0.0.1:{port}/health", timeout=2):
                    pass
                break
            except (OSError, ValueError):
                time.sleep(2)
        info = E.verify_server(f"http://127.0.0.1:{port}", ckpt_path, MODEL_ID)
        return process, watchdog, info
    except Exception:
        watchdog.cancel()
        stop_group(process)
        raise


def request_seed(seed, item_id, rep=0):
    return int(digest([seed, item_id, rep])[:8], 16) % (2 ** 31)


def cmd_gen(a):
    plan_path = a.plan.resolve()
    root = plan_path.parent
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_hash = D.file_hash(plan_path)
    if plan.get("profile") != D.PROFILE or plan.get("schema") != "mathplus-queue-v1":
        raise ValueError("not a mathplus plan")
    if plan.get("source") != D.script_hashes():
        raise ValueError("mathplus source changed")
    for data in plan["data"].values():
        if D.file_hash(data["path"]) != data["sha256"]:
            raise ValueError("prepared data changed")
    own_lock = Path(tempfile.gettempdir()) / f"mathplus-gpu{a.gpu}.lock"
    main_lock = Path(tempfile.gettempdir()) / f"simct-eval-gpu{a.gpu}.lock"
    with locked(own_lock, blocking=False) as own:
        if own is None:
            raise RuntimeError("another mathplus worker owns this GPU")
        with locked(main_lock, blocking=False) as main:
            if main is None:
                raise RuntimeError("another eval worker owns this GPU")
            state, state_path = read_state(root, plan_hash, plan)
            atomic_json(state_path, state)
            port = 32000 + 1000 * a.gpu
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError("server port already occupied")
            for job in plan["jobs"]:
                if state["jobs"].get(job["id"], {}).get("status") == "completed":
                    continue
                actual = E.checkpoint_identity(job["checkpoint"]["path"])
                if actual != job["checkpoint"]:
                    raise ValueError("checkpoint changed after plan")
                with locked(root / (job["id"] + ".lock"), blocking=False) as jl:
                    if jl is None:
                        continue
                    state["jobs"][job["id"]] = {"status": "running", "gpu": a.gpu,
                                                "started": time.time()}
                    atomic_json(state_path, state)
                    started = time.time()
                    try:
                        run_job(root, plan, job, a, port, state["deadline"])
                        state["jobs"][job["id"]] = {"status": "completed",
                                                    "seconds": time.time() - started}
                    except Deadline as exc:
                        state["jobs"][job["id"]] = {"status": "partial",
                                                    "reason": str(exc)}
                    except Exception as exc:
                        state["jobs"][job["id"]] = {"status": "failed",
                                                    "error": type(exc).__name__ + ": " + str(exc)}
                    atomic_json(state_path, state)
                    print("JOB", job["id"],
                          json.dumps(state["jobs"][job["id"]]), flush=True)
                    if state["jobs"][job["id"]]["status"] != "completed":
                        return
            print("QUEUE_STOP", flush=True)


def run_job(root, plan, job, a, port, deadline):
    data_items = {}
    for bench, data in plan["data"].items():
        payload = json.loads(Path(data["path"]).read_text(encoding="utf-8"))
        data_items[bench] = payload["items"]
    process, watchdog, server = launch_server(job["checkpoint"]["path"], a.gpu,
                                              port, deadline)
    try:
        for seed in plan["seeds"]:
            for bench in plan["data"]:
                cell = root / "cells" / job["id"] / bench / str(seed)
                cell.mkdir(parents=True, exist_ok=True)
                if (cell / "generation-complete.json").exists():
                    continue
                out = cell / "generation.jsonl"
                found = set()
                n = int(plan.get("n", 1))
                if out.exists():
                    for line in out.read_text().splitlines():
                        row = json.loads(line)
                        found.add((row["id"], row.get("rep", 0)))
                base = f"http://127.0.0.1:{port}"
                for item in data_items[bench]:
                    for rep in range(n):
                        if (item["id"], rep) in found:
                            continue
                        if time.time() >= deadline:
                            raise Deadline("wall budget")
                        msgs = (item["messages"] if plan.get("protocol", "simct") == "simct"
                                else item["messages_harness"])
                        payload = {"model": MODEL_ID, "messages": msgs,
                                   "temperature": plan.get("temperature", 0.0),
                                   "top_p": plan.get("top_p", 1.0),
                                   "max_tokens": D.CAPS[bench], "n": 1,
                                   "seed": request_seed(seed, item["id"], rep),
                                   "chat_template_kwargs": {"enable_thinking": False}}
                    req = urllib.request.Request(
                        base + "/v1/chat/completions",
                        data=json.dumps(payload).encode(),
                        headers={"Content-Type": "application/json"})
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    with opener.open(req, timeout=600) as resp:
                        result = json.load(resp)
                    choices = result.get("choices", [])
                    if result.get("error") or len(choices) != 1 or choices[0].get(
                            "finish_reason") not in {"stop", "length"}:
                        raise ValueError("invalid inference response")
                    content = choices[0]["message"]["content"]
                    if not isinstance(content, str):
                        raise ValueError("missing content")
                    with out.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"id": item["id"], "seed": seed, "rep": rep,
                                            "request_seed": payload["seed"], "text": content}) + "\n")
                (cell / "generation-complete.json").write_text(json.dumps(
                    {"job": job["id"], "benchmark": bench, "seed": seed,
                     "server": server, "count": len(data_items[bench])}), encoding="utf-8")
                print("GENERATION_COMPLETE", job["id"], bench, seed,
                      len(data_items[bench]), flush=True)
        if E.verify_server(f"http://127.0.0.1:{port}",
                           job["checkpoint"]["path"], MODEL_ID) != server:
            raise ValueError("server identity drift")
    finally:
        watchdog.cancel()
        stop_group(process)


def cmd_score(a):
    plan_path = a.plan.resolve()
    root = plan_path.parent
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("profile") != D.PROFILE or plan.get("schema") != "mathplus-queue-v1":
        raise ValueError("not a mathplus plan")
    if plan.get("source") != D.script_hashes():
        raise ValueError("mathplus source changed")
    data_items = {}
    for bench, data in plan["data"].items():
        if D.file_hash(data["path"]) != data["sha256"]:
            raise ValueError("prepared data changed")
        payload = json.loads(Path(data["path"]).read_text(encoding="utf-8"))
        data_items[bench] = {x["id"]: x for x in payload["items"]}
    for job in plan["jobs"]:
        for seed in plan["seeds"]:
            for bench in plan["data"]:
                cell = root / "cells" / job["id"] / bench / str(seed)
                key = f"{job['id']}/{bench}/{seed}"
                complete = cell / "generation-complete.json"
                if not complete.exists():
                    print("SKIP", key, "(no generation)", flush=True)
                    continue
                if (cell / "metrics.json").exists():
                    print("SKIP", key, "(done)", flush=True)
                    continue
                print("SCORE_START", key, flush=True)
                texts = {}
                for line in (cell / "generation.jsonl").read_text().splitlines():
                    row = json.loads(line)
                    texts.setdefault(row["id"], {})[row.get("rep", 0)] = row["text"]
                n = int(plan.get("n", 1))
                correct_reps, passed, detail = [0] * n, 0, []
                for item_id, item in data_items[bench].items():
                    reps = texts.get(item_id, {})
                    hits = 0
                    for rep in range(n):
                        text = reps.get(rep, "")
                        if bench == "gpqa-diamond":
                            ok, pred = grade_gpqa(text, item["gold"])
                        else:
                            ok, pred = grade_math(text, item["gold"])
                        correct_reps[rep] += bool(ok)
                        hits += bool(ok)
                        detail.append({"id": item_id, "rep": rep, "gold": item["gold"],
                                       "pred": pred, "correct": bool(ok)})
                    passed += (hits > 0)
                total = len(data_items[bench])
                avgs = [c / total for c in correct_reps]
                with locked(cell / "scoring.lock", blocking=False) as own:
                    if own is None:
                        print("BUSY", key, flush=True)
                        continue
                    (cell / "predictions.json").write_text(
                        json.dumps(detail, indent=2), encoding="utf-8")
                    (cell / "metrics.json").write_text(json.dumps(
                        {"mean": sum(avgs) / n, "avg_at_n": sum(avgs) / n,
                         "pass_at_n": passed / total, "n": n,
                         "rep_avgs": avgs, "passed": passed,
                         "correct": correct_reps[0], "total": total,
                         "n_present": 1, "eval_n": 1,
                         "seeds": {str(seed): sum(avgs) / n}},
                        indent=2), encoding="utf-8")
                print("SCORE_DONE", key,
                      f"avg@{n}={sum(avgs) / n:.4f} pass@{n}={passed / total:.4f}",
                      flush=True)
    print("SCORING_PASS_DONE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--checkpoint", action="append", default=[],
                   help="checkpoint dir (repeatable)")
    q.add_argument("--shared", type=Path,
                   default=Path("/workspace/storage-shared/nlp/tungks"))
    q.add_argument("--proxy", default="http://10.30.154.118:80")
    q.add_argument("--hours", type=float, default=20.)
    q.add_argument("--temperature", type=float, default=0.0)
    q.add_argument("--top_p", type=float, default=1.0)
    q.add_argument("--n", type=int, default=1,
                   help="samples per item; pass@n needs temperature>0")
    q.add_argument("--protocol", choices=("simct", "harness"), default="simct")
    q.add_argument("--benches", default="",
                   help="comma subset, e.g. aime24,aime25 (default: all 5)")
    q.set_defaults(func=cmd_prepare)
    q = sub.add_parser("gen")
    q.add_argument("--plan", type=Path, required=True)
    q.add_argument("--gpu", type=int, choices=range(8), required=True)
    q.set_defaults(func=cmd_gen)
    q = sub.add_parser("score")
    q.add_argument("--plan", type=Path, required=True)
    q.set_defaults(func=cmd_score)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
