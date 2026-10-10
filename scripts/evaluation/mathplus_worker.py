#!/usr/bin/env python3
"""Gen/score worker for the mathplus extension.

Mirrors the eval_queue conventions (plan/state/cells layout, SGLang server
per checkpoint via python-b200-host.sh, per-item stable request seeds) but
validates against its own profile/source, so the main 4-bench queues are
untouched. New file only.
"""
import argparse
import concurrent.futures as cf
import contextlib
import fcntl
import json
import urllib.error
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


MILESTONES = ("312", "280", "240", "200", "160", "120", "80", "40")


def resolve_checkpoints(specs):
    """Expand runs-file lines / --checkpoint specs to (ckptdir, steps).

    A runs-file line is ``RUNDIR [step ...]``; without steps the milestone
    steps that exist on disk are used, newest first. ``#`` starts a comment.
    """
    out = []
    for spec in specs:
        parts = spec.split()
        if not parts or parts[0].startswith("#"):
            continue
        run = Path(parts[0])
        if not run.is_absolute():
            run = Path("/workspace/storage-shared/nlp/tungks") / run
        steps = [p for p in parts[1:] if (run / "checkpoint" / f"step{p}").is_dir()]
        if len(parts) == 1:
            steps = [s for s in MILESTONES
                     if (run / "checkpoint" / f"step{s}").is_dir()]
        for s in steps:
            out.append(run / "checkpoint" / f"step{s}")
    return out


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
    specs = list(a.checkpoint)
    queue_lines = []
    if a.runs_file:
        lines = Path(a.runs_file).read_text().splitlines()
        queue_lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        (root / "runs.queue").write_text("\n".join(queue_lines) + "\n", encoding="utf-8")
        (root / "runs.evaluated").write_text("", encoding="utf-8")
        specs = [str(p) for p in resolve_checkpoints(queue_lines)]
    for spec in specs:
        ckpt = Path(spec).resolve()
        ident = E.checkpoint_identity(str(ckpt))
        era, rule = D.era_of(str(ckpt))
        print(f"MATH_ERA {ckpt.parent.name}/{ckpt.name} -> {era}"
              + (f" ({rule})" if rule else " (no rule, explicit unknown)"), flush=True)
        jobs.append({"id": f"mathplus-{ckpt.parent.parent.name}-{ckpt.name}",
                     "run": ckpt.parent.name,
                     "checkpoint": ident, "tier": 0, "math": era})
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
    data_ok, note = D.source_matches(plan.get("source"))
    if not data_ok:
        raise ValueError("mathplus data contract changed")
    print("SOURCE", note, flush=True)
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
                pending = [(item, rep) for item in data_items[bench]
                           for rep in range(n)
                           if (item["id"], rep) not in found]
                proto = plan.get("protocol", "simct")
                temp = plan.get("temperature", 0.0)
                topp = plan.get("top_p", 1.0)

                def one(task):
                    item, rep = task
                    msgs = (item["messages"] if proto == "simct"
                            else item["messages_harness"])
                    payload = {"model": MODEL_ID, "messages": msgs,
                               "temperature": temp, "top_p": topp,
                               "max_tokens": D.CAPS[bench], "n": 1,
                               "seed": request_seed(seed, item["id"], rep),
                               "chat_template_kwargs": {"enable_thinking": False}}
                    req = urllib.request.Request(
                        base + "/v1/chat/completions",
                        data=json.dumps(payload).encode(),
                        headers={"Content-Type": "application/json"})
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    try:
                        with opener.open(req, timeout=600) as resp:
                            result = json.load(resp)
                    except urllib.error.HTTPError as exc:
                        body = exc.read().decode(errors="replace")[:500]
                        raise ValueError(f"HTTP {exc.code}: {body}")
                    choices = result.get("choices", [])
                    if result.get("error") or len(choices) != 1 or choices[0].get(
                            "finish_reason") not in {"stop", "length"}:
                        raise ValueError("invalid inference response")
                    content = choices[0]["message"]["content"]
                    if not isinstance(content, str):
                        raise ValueError("missing content")
                    return (item["id"], rep, payload["seed"], content)

                workers = min(a.concurrency, len(pending) or 1)
                with cf.ThreadPoolExecutor(max_workers=workers) as pool:
                    future_map = {pool.submit(one, t): t for t in pending}
                    done = 0
                    for fut in cf.as_completed(future_map):
                        if time.time() >= deadline:
                            pool.shutdown(wait=False, cancel_futures=True)
                            raise Deadline("wall budget")
                        item_id, rep, rseed, content = fut.result()
                        with out.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(
                                {"id": item_id, "seed": seed, "rep": rep,
                                 "request_seed": rseed, "text": content}) + "\n")
                        done += 1
                        if done % 50 == 0:
                            print("PROGRESS", job["id"], bench, seed,
                                  f"{done}/{len(pending)}", flush=True)
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
    data_ok, note = D.source_matches(plan.get("source"))
    if not data_ok:
        raise ValueError("mathplus data contract changed")
    print("SOURCE", note, flush=True)
    data_items = {}
    for bench, data in plan["data"].items():
        if D.file_hash(data["path"]) != data["sha256"]:
            raise ValueError("prepared data changed")
        payload = json.loads(Path(data["path"]).read_text(encoding="utf-8"))
        data_items[bench] = {x["id"]: x for x in payload["items"]}
    nshard = max(1, int(a.shard_count))
    ishhard = int(a.shard_index) % nshard
    print(f"SHARD {ishhard}/{nshard}", flush=True)
    idx = 0
    for job in plan["jobs"]:
        for seed in plan["seeds"]:
            for bench in plan["data"]:
                mine = (idx % nshard) == ishhard
                idx += 1
                cell = root / "cells" / job["id"] / bench / str(seed)
                key = f"{job['id']}/{bench}/{seed}"
                if not mine:
                    continue
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
                         "math": job.get("math", "unknown"),
                         "checkpoint_sha256": job["checkpoint"]["sha256"],
                         "n_present": 1, "eval_n": 1,
                         "seeds": {str(seed): sum(avgs) / n}},
                        indent=2), encoding="utf-8")
                print("SCORE_DONE", key,
                      f"avg@{n}={sum(avgs) / n:.4f} pass@{n}={passed / total:.4f}",
                      flush=True)
    print("SCORING_PASS_DONE", flush=True)


def cmd_mark_done(a):
    """Move a fully scored run from runs.queue to runs.evaluated.

    A run counts as done only when every one of its jobs has metrics.json
    for every bench x seed cell. Refuses otherwise (prints what is missing).
    """
    root = a.plan.resolve().parent
    plan = json.loads(a.plan.resolve().read_text(encoding="utf-8"))
    queue_path = root / "runs.queue"
    done_path = root / "runs.evaluated"
    if not queue_path.exists():
        raise ValueError("no runs.queue (plan was not prepared with --runs-file)")
    targets = [j for j in plan["jobs"] if a.run in j["id"]]
    if not targets:
        raise ValueError(f"no jobs match {a.run!r}")
    missing = []
    for job in targets:
        for seed in plan["seeds"]:
            for bench in plan["data"]:
                if not (root / "cells" / job["id"] / bench / str(seed)
                        / "metrics.json").exists():
                    missing.append(f"{job['id']}/{bench}/{seed}")
    if missing:
        print(f"NOT DONE {a.run}: {len(missing)} cells missing, e.g. {missing[:3]}",
              flush=True)
        return
    lines = queue_path.read_text().splitlines()
    keep, moved = [], []
    for ln in lines:
        if ln.strip() and not ln.strip().startswith("#") and a.run in ln:
            moved.append(ln)
        else:
            keep.append(ln)
    queue_path.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
    with done_path.open("a", encoding="utf-8") as f:
        for ln in moved:
            f.write(ln + "\n")
    print(f"MARKED DONE {a.run}: {len(targets)} jobs, moved {len(moved)} queue lines",
          flush=True)


def cmd_loadtest(a):
    """VRAM/latency probe: serve a checkpoint and fire concurrent fixed
    requests. Reports p50/p95 latency, error count and GPU memory."""
    import statistics
    ckpt = Path(a.checkpoint).resolve()
    ident = E.checkpoint_identity(str(ckpt))
    deadline = time.time() + a.minutes * 60
    port = 32100 + 1000 * a.gpu
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError("server port already occupied")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), HF_HUB_OFFLINE="1",
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
               "--model-path", str(ckpt), "--served-model-name", MODEL_ID,
               "--host", "127.0.0.1", "--port", str(port),
               "--tp-size", "1", "--mem-fraction-static", "0.8",
               "--context-length", str(a.context_length),
               "--attention-backend", "triton", "--disable-cuda-graph",
               "--random-seed", "42"]
    log = open(f"/tmp/mathplus-loadtest-gpu{a.gpu}.log", "ab")
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
        E.verify_server(f"http://127.0.0.1:{port}", str(ckpt), MODEL_ID)
        if a.items_from:
            payload = json.loads(Path(a.items_from).read_text(encoding="utf-8"))
            pool_items = [x["messages"] for x in payload["items"][:a.bench_items]]
            if not pool_items:
                raise ValueError("no items in items-from file")
        else:
            prompt = ("Solve step by step and put the final answer in \\boxed{}. "
                      "If a train travels 120 km in 2 hours, what is its average speed in km/h? "
                      "Show the formula, substitute the numbers, and compute carefully. " * 4)
            pool_items = [[{"role": "user", "content": prompt}]]

        def one(i):
            msgs = pool_items[i % len(pool_items)]
            payload = {"model": MODEL_ID, "messages": msgs,
                       "temperature": 0.6, "top_p": 0.95,
                       "max_tokens": a.max_tokens, "n": 1, "seed": i}
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            t0 = time.time()
            try:
                with opener.open(req, timeout=900) as resp:
                    result = json.load(resp)
                ok = bool(result.get("choices")) and result["choices"][0].get(
                    "finish_reason") in {"stop", "length"}
                return (time.time() - t0, ok, "")
            except Exception as exc:
                return (time.time() - t0, False,
                        f"{type(exc).__name__}: {str(exc)[:120]}")

        results = []
        with cf.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            futs = [pool.submit(one, i) for i in range(a.requests)]
            for fut in cf.as_completed(futs):
                results.append(fut.result())
                if len(results) % 20 == 0:
                    print(f"PROGRESS {len(results)}/{len(futs)}", flush=True)
        lat = sorted(r[0] for r in results)
        ok = sum(1 for r in results if r[1])
        errs = {}
        for _, good, msg in results:
            if not good:
                errs[msg] = errs.get(msg, 0) + 1
        print(json.dumps({
            "gpu": a.gpu, "context_length": a.context_length,
            "max_tokens": a.max_tokens, "concurrency": a.concurrency,
            "requests": len(results), "ok": ok,
            "lat_p50": statistics.median(lat),
            "lat_p95": lat[max(0, int(len(lat) * 0.95) - 1)],
            "lat_max": lat[-1], "errors": errs}, indent=2), flush=True)
    finally:
        watchdog.cancel()
        stop_group(process)


def cmd_migrate_source(a):
    """Re-stamp an existing plan with the current source record without
    touching jobs/cells: used when only the worker (never the data contract)
    changed. Records the previous stamp for provenance."""
    root = a.plan.resolve().parent
    plan = json.loads(a.plan.resolve().read_text(encoding="utf-8"))
    prev = plan.get("source")
    plan["source"] = D.script_hashes()
    plan.setdefault("source_migrations", []).append(
        {"previous": prev, "time": time.time()})
    atomic_json(a.plan.resolve(), plan)
    print("MIGRATED", a.plan, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--checkpoint", action="append", default=[],
                   help="checkpoint dir (repeatable)")
    q.add_argument("--runs-file", default="",
                   help="queue file: one RUNDIR [steps...] per line")
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
    q.add_argument("--concurrency", type=int, default=128)
    q.set_defaults(func=cmd_gen)
    q = sub.add_parser("score")
    q.add_argument("--plan", type=Path, required=True)
    q.add_argument("--shard-index", type=int, default=0)
    q.add_argument("--shard-count", type=int, default=1)
    q.set_defaults(func=cmd_score)
    q = sub.add_parser("mark-done")
    q.add_argument("--plan", type=Path, required=True)
    q.add_argument("--run", required=True,
                   help="substring of job id, e.g. a run-dir name")
    q.set_defaults(func=cmd_mark_done)
    q = sub.add_parser("migrate-source")
    q.add_argument("--plan", type=Path, required=True)
    q.set_defaults(func=cmd_migrate_source)
    q = sub.add_parser("loadtest")
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--gpu", type=int, choices=range(8), required=True)
    q.add_argument("--context-length", type=int, default=16384)
    q.add_argument("--max-tokens", type=int, default=8192)
    q.add_argument("--concurrency", type=int, default=128)
    q.add_argument("--requests", type=int, default=64)
    q.add_argument("--minutes", type=float, default=30.)
    q.add_argument("--items-from", default="",
                   help="prepared {bench}.json for real workload (default: synthetic prompt)")
    q.add_argument("--bench-items", type=int, default=30)
    q.set_defaults(func=cmd_loadtest)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
