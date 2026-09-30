#!/usr/bin/env python3
"""Explicit SimCT eval contracts. CPU prepare/score; generation uses an existing server.

Never starts a GPU server, installs packages, or uploads results.
"""
from __future__ import annotations

import argparse
import ast
import base64
import concurrent.futures
import functools
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
AUTHOR_SHA = "21378cfb1aa1d2f3ddab684a1bcb671fd588919c76fd410fac424bd062db2839"
AUTHOR_COMMIT = "cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e"
CAPS = {"gsm8k": 4096, "math500": 4096, "mbpp": 2048, "live-code-bench-v6": 4096}
COUNTS = {"gsm8k": 1319, "math500": 500, "mbpp": 500, "live-code-bench-v6": 1055}
PROFILES = ("author-code", "paper-spec")
SEEDS = (42, 43, 44, 45, 46)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


@functools.lru_cache(maxsize=1)
def author_helpers():
    """Compile only audited author helpers, avoiding optional inference imports."""
    source = (HERE / "evaluation.py").read_bytes()
    if digest(source) != AUTHOR_SHA:
        raise ValueError("Author evaluator changed: audit and repin before evaluation")
    names = {"strip_thinking_content", "extract_number_from_answer", "normalize_answer_string",
             "extract_boxed_answer", "try_parse_number", "is_math_equivalent", "_extract_code_block",
             "_run_mbpp_tests", "_run_lcb_tests", "QWEN_MATH_SYSTEM_PROMPT",
             "GSM8K_SYSTEM_PROMPT", "MBPP_SYSTEM_PROMPT", "LCB_SYSTEM_PROMPT"}
    nodes = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            nodes.append(node)
    namespace = dict(re=re, json=json, List=List, Dict=Dict, Any=Any, Optional=Optional, Tuple=Tuple)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(HERE / "evaluation.py"), "exec"), namespace)
    return namespace


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Private-test pickle may not reference globals")


def test_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            result = json.loads(value)
        except json.JSONDecodeError:
            result = RestrictedUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(value)))).load()
            if isinstance(result, str):
                result = json.loads(result)
        if isinstance(result, list):
            return result
    raise ValueError("Invalid test-list encoding; no silent dropping of private tests")


def normalize(benchmark, row, profile):
    if benchmark == "gsm8k":
        prompt, gold = row["question"], row["answer"]
        item = {"id": str(row.get("id", digest(prompt.encode()))), "prompt": prompt, "gold": gold}
    elif benchmark == "math500":
        prompt, gold = row["problem"], row["answer"]
        item = {"id": str(row.get("unique_id", row.get("id", digest(prompt.encode())))), "prompt": prompt, "gold": gold}
    elif benchmark == "mbpp":
        tests = row["test_list"]
        if not tests or not all(isinstance(t, str) and t.strip() for t in tests):
            raise ValueError("MBPP requires nonempty assertion tests")
        item = {"id": str(row["task_id"]), "prompt": row["text"], "tests": tests,
                "setup": row.get("test_setup_code", "")}
    else:
        # Company prompt/ground_truth alone does not prove release or hidden-test coverage.
        public = test_list(row["public_test_cases"])
        private = test_list(row["private_test_cases"])
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        fn_name = metadata.get("func_name")
        cases = public + private
        if not cases:
            raise ValueError("LCB has no tests")
        for case in cases:
            if not isinstance(case.get("input"), str) or not isinstance(case.get("output"), str):
                raise ValueError("LCB inputs/outputs must preserve original strings")
            if case.get("testtype") == "functional" and not fn_name:
                raise ValueError("Functional LCB case missing func_name")
        item = {"id": str(row["question_id"]), "prompt": row["question_content"],
                "tests": cases, "fn_name": fn_name, "private_test_count": len(private)}
    if not item["id"] or not isinstance(item["prompt"], str) or not item["prompt"].strip():
        raise ValueError("Empty ID/prompt")
    return item


def messages(benchmark, item, profile, system_role):
    h = author_helpers()
    constant = {"gsm8k": "GSM8K_SYSTEM_PROMPT", "math500": "QWEN_MATH_SYSTEM_PROMPT",
                "mbpp": "MBPP_SYSTEM_PROMPT", "live-code-bench-v6": "LCB_SYSTEM_PROMPT"}[benchmark]
    if profile == "paper-spec" and benchmark == "gsm8k":
        constant = "QWEN_MATH_SYSTEM_PROMPT"
    system = h[constant]
    prompt = item["prompt"]
    if benchmark == "mbpp":
        prompt += "\n\nYour code should pass these tests:\n" + "\n".join(item["tests"])
    if system_role == "merge":
        return [{"role": "user", "content": system + "\n\n" + prompt}]
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def prepare(args):
    smoke = getattr(args, "smoke", False)
    if args.split != "test":
        raise ValueError("Evaluation contract requires the test split")
    if not smoke and args.expected_count != COUNTS[args.benchmark]:
        raise ValueError("Full evaluation requires reference count; use --smoke for explicit small diagnostics")
    if args.format == "jsonl":
        rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    else:
        # Offline only: use company snapshots/cache; no network bootstrap.
        os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        from datasets import load_dataset, load_from_disk, DatasetDict
        if args.format == "hf-cache":
            rows = load_dataset(args.data, args.config, split=args.split, revision=args.revision)
        else:
            rows = load_from_disk(args.data)
            if isinstance(rows, DatasetDict):
                rows = rows[args.split]
        rows = list(rows)
    if len(rows) != args.expected_count or not rows:
        raise ValueError(f"Expected {args.expected_count} rows, got {len(rows)}")
    if args.benchmark == "live-code-bench-v6" and args.release != "release_v6":
        raise ValueError("LCB requires explicit release_v6 provenance")
    items = [normalize(args.benchmark, row, args.profile) for row in rows]
    if len({x["id"] for x in items}) != len(items):
        raise ValueError("Duplicate dataset IDs")
    if args.benchmark == "mbpp" and not smoke and {x["id"] for x in items} != {str(i) for i in range(11, 511)}:
        raise ValueError("MBPP full test must contain task IDs 11..510; sanitized is a different contract")
    for item in items:
        item["messages"] = messages(args.benchmark, item, args.profile, args.system_role)
    write_new(args.out, {"schema": 1, "benchmark": args.benchmark, "profile": args.profile, "scope": "smoke" if smoke else "full",
                        "source": {"data": args.data, "revision": args.revision, "split": args.split,
                                   "config": args.config, "release": args.release,
                                   "provenance": "user-supplied revision; content hash pins the actual rows"},
                        "author_commit": AUTHOR_COMMIT, "author_sha256": AUTHOR_SHA,
                        "system_role": args.system_role, "items_sha256": digest(encoded(items)), "items": items})


def load_prepared(path):
    prepared = read_json(path)
    if prepared["profile"] not in PROFILES or prepared["benchmark"] not in CAPS:
        raise ValueError("Unknown contract")
    if prepared["author_sha256"] != AUTHOR_SHA or digest(encoded(prepared["items"])) != prepared["items_sha256"]:
        raise ValueError("Prepared data integrity mismatch")
    if not prepared["items"] or len({x["id"] for x in prepared["items"]}) != len(prepared["items"]):
        raise ValueError("Empty data or duplicate IDs")
    if prepared["scope"] not in {"smoke", "full"}:
        raise ValueError("Unknown evaluation scope")
    if prepared["scope"] == "full" and len(prepared["items"]) != COUNTS[prepared["benchmark"]]:
        raise ValueError("Full evaluation count mismatch")
    for item in prepared["items"]:
        if item["messages"] != messages(prepared["benchmark"], item, prepared["profile"], prepared["system_role"]):
            raise ValueError("Prompt differs from selected contract")
    return prepared


def checkpoint_identity(path, expected_updates=None):
    root = Path(path).resolve(strict=True)
    required = [root / "config.json", root / "tokenizer_config.json"]
    if not any((root / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise ValueError("Checkpoint missing tokenizer")
    indices = list(root.glob("*.index.json"))
    if indices:
        for index in indices:
            for name in set(read_json(index)["weight_map"].values()):
                shard = (root / name).resolve()
                if not shard.is_relative_to(root) or not shard.is_file():
                    raise ValueError("Missing or escaping checkpoint shard")
    weights = list(root.glob("*.safetensors")) + list(root.glob("pytorch_model*.bin"))
    if not weights or not all(p.is_file() for p in required):
        raise ValueError("Checkpoint incomplete")
    if expected_updates is not None:
        summary = read_json(root / "run-summary.json")
        exitfile = Path(str(root.parent) + ".exitcode")
        if summary.get("status") != "completed" or summary.get("optimizer_updates") != expected_updates:
            raise ValueError("Training summary is not completed at required update budget")
        if not exitfile.is_file() or exitfile.read_text().strip() != "0":
            raise ValueError("Missing successful training exitcode")
    # Hash after training; includes tokenizer and model files without touching caches.
    files = sorted(set(required + weights + indices + list(root.glob("tokenizer*")) + list(root.glob("special_tokens*"))))
    hashes = {p.name: file_hash(p) for p in files if p.is_file()}
    return {"path": str(root), "files_sha256": hashes, "sha256": digest(encoded(hashes))}


def request_json(base, suffix, payload=None):
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Only local inference endpoints are accepted")
    req = urllib.request.Request(base.rstrip("/") + suffix,
                                 data=encoded(payload) if payload is not None else None,
                                 headers={"Content-Type": "application/json"})
    # Loopback requests must not use the company outbound proxy.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=600) as response:
        return json.load(response)


def verify_server(base, checkpoint, model):
    info = request_json(base, "/get_server_info")
    path = info.get("model_path") or info.get("server_args", {}).get("model_path")
    if not path or Path(path).resolve() != Path(checkpoint).resolve():
        raise ValueError("SGLang model_path does not match checkpoint; restart with the right checkpoint")
    models = request_json(base, "/v1/models")
    if model not in {item["id"] for item in models["data"]}:
        raise ValueError("Served model ID mismatch")
    keys = ("model_path", "dtype", "tp_size", "dp_size", "attention_backend", "random_seed",
            "disable_cuda_graph", "context_length")
    server_args = info.get("server_args", info)
    return {"version": info.get("version"), "args": {key: server_args.get(key) for key in keys}}


def generation_payload(model, item, benchmark, seed):
    # Stable per-item seed: independent of request order/concurrency and equal across models.
    request_seed = int(digest(encoded([seed, item["id"]]))[:8], 16) % (2**31)
    payload = {"model": model, "messages": item["messages"], "temperature": 0.6, "top_p": 0.95,
               "max_tokens": CAPS[benchmark], "n": 1, "seed": request_seed,
               "chat_template_kwargs": {"enable_thinking": False}}
    return payload


def validate_response(result):
    choices = result.get("choices", [])
    if result.get("error") or len(choices) != 1 or choices[0].get("finish_reason") not in {"stop", "length"}:
        raise ValueError("Invalid inference response (no silent empty-result fallback)")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("Missing content")


def generate_one(base, model, item, benchmark, seed):
    payload = generation_payload(model, item, benchmark, seed)
    result = request_json(base, "/v1/chat/completions", payload)
    validate_response(result)
    return {"id": item["id"], "seed": seed, "request_seed": payload["seed"],
            "request_sha256": digest(encoded(payload)), "response": result}


def load_records(path, expected):
    found = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if row["id"] not in expected or row["id"] in found:
                raise ValueError("Unexpected or duplicate result ID")
            found[row["id"]] = row
    return found


def validate_records(found, prepared, contract):
    for item in prepared["items"]:
        if item["id"] not in found:
            continue
        row = found[item["id"]]
        payload = generation_payload(contract["model"], item, prepared["benchmark"], contract["seed"])
        if row["seed"] != contract["seed"] or row["request_seed"] != payload["seed"] or row["request_sha256"] != digest(encoded(payload)):
            raise ValueError("Saved response request/seed mismatch")
        validate_response(row["response"])


def generate(args):
    prepared = load_prepared(args.prepared)
    identity = checkpoint_identity(args.checkpoint, args.expected_updates)
    server_info = verify_server(args.base_url, identity["path"], args.model)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    contract = {"prepared_sha256": file_hash(args.prepared), "checkpoint": identity,
                "profile": prepared["profile"], "benchmark": prepared["benchmark"], "scope": prepared["scope"],
                "seed": args.seed, "model": args.model, "schema": 1, "server": server_info,
                "runner_sha256": file_hash(__file__), "temperature": 0.6, "top_p": 0.95,
                "max_tokens": CAPS[prepared["benchmark"]], "seed_rule": "sha256([repeat_seed,id])[:8] modulo 2**31"}
    lock = root / ".generating"
    with lock.open("x"):
        pass
    try:
        manifest_path = root / "generation.json"
        if manifest_path.exists():
            if read_json(manifest_path) != contract:
                raise ValueError("Resume contract mismatch")
        else:
            if any(root.glob("*.jsonl")):
                raise ValueError("Orphan result files")
            write_new(manifest_path, contract)
        output = root / "responses.jsonl"
        ids = {x["id"] for x in prepared["items"]}
        found = load_records(output, ids)
        validate_records(found, prepared, contract)
        pending = [x for x in prepared["items"] if x["id"] not in found]
        with output.open("a") as stream, concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(generate_one, args.base_url, args.model, item, prepared["benchmark"], args.seed) for item in pending]
            for future in concurrent.futures.as_completed(futures):
                try:
                    row = future.result()  # API failures stop the cell; saved responses remain resumable.
                except Exception:
                    for pending_future in futures:
                        pending_future.cancel()
                    raise
                stream.write(encoded(row).decode() + "\n")
                stream.flush()
        if len(load_records(output, ids)) != len(ids):
            raise ValueError("Incomplete generation")
        if verify_server(args.base_url, identity["path"], args.model) != server_info:
            raise ValueError("Server configuration changed during generation")
        complete = {"generation_sha256": file_hash(manifest_path), "responses_sha256": file_hash(output), "count": len(ids)}
        if (root / "complete.json").exists():
            if read_json(root / "complete.json") != complete:
                raise ValueError("Completion receipt mismatch")
        else:
            write_new(root / "complete.json", complete)
        print(f"GENERATION_COMPLETE: {len(ids)} items, seed={args.seed}")
    finally:
        lock.unlink()


def sandbox_command(python, runtime_roots=()):
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bubblewrap unavailable; no unsandboxed code-scoring fallback")
    cmd = [bwrap, "--die-with-parent", "--unshare-all", "--new-session", "--clearenv"]
    for path in ("/usr", "/lib", "/lib64", "/etc/ld.so.cache", "/etc/alternatives", *runtime_roots):
        p = Path(path).resolve()
        if not p.exists():
            continue
        if str(p) in {"/", "/home", "/root", "/workspace", "/mnt", "/opt"}:
            raise ValueError("Runtime mount must be narrowly scoped")
        cmd += ["--ro-bind", str(p), str(path)]
    cmd += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/dev/shm", "--tmpfs", "/tmp", "--dir", "/work",
            "--ro-bind", str(HERE), "/app", "--chdir", "/work",
            "--setenv", "OPENBLAS_NUM_THREADS", "1", "--setenv", "OMP_NUM_THREADS", "1",
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--setenv", "HOME", "/work",
            python, "/app/contract_worker.py"]
    return cmd


def sandbox_job(command, job):
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        done = subprocess.run(command, input=encoded(job), stdout=stdout, stderr=stderr,
                              timeout=job.get("outer_timeout", 180))
        stdout.seek(0)
        text = stdout.read(1024 * 1024)
        if done.returncode:
            try:
                message = json.loads(text).get("error", "worker failed")
            except (ValueError, UnicodeDecodeError):
                message = "sandbox/runtime startup failed"
            raise RuntimeError(f"Scorer exit {done.returncode}: {message}; cell not scored")
        result = json.loads(text)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result


def preflight(args):
    command = sandbox_command(args.python, args.runtime_root)
    results = {}
    failed = False
    for benchmark in args.benchmarks:
        try:
            results[benchmark] = [sandbox_job(command, {"qualify": True, "correct": correct,
                                 "profile": args.profile, "benchmark": benchmark}) for correct in (True, False)]
        except Exception as exc:
            failed = True
            results[benchmark] = {"error": str(exc)}
    print(json.dumps({"status": "not_ready" if failed else "cpu_scorers_qualified",
                      "profile": args.profile, "results": results}, indent=2))
    if failed:
        raise SystemExit(1)


def score(args):
    prepared = load_prepared(args.prepared)
    root = Path(args.run)
    contract = read_json(root / "generation.json")
    if contract["prepared_sha256"] != file_hash(args.prepared):
        raise ValueError("Generation data mismatch")
    ids = {x["id"] for x in prepared["items"]}
    found = load_records(root / "responses.jsonl", ids)
    if set(found) != ids:
        raise ValueError("Cannot score incomplete generation")
    validate_records(found, prepared, contract)
    receipt = read_json(root / "complete.json")
    if receipt != {"generation_sha256": file_hash(root / "generation.json"),
                   "responses_sha256": file_hash(root / "responses.jsonl"), "count": len(ids)}:
        raise ValueError("Missing or invalid completion receipt")
    output = root / "metrics.json"
    if output.exists():
        raise ValueError("Metrics already exist; refusing overwrite")
    command = sandbox_command(args.python, args.runtime_root)
    def worker(job):
        return sandbox_job(command, job)
    qualification = [worker({"qualify": True, "correct": correct, "profile": prepared["profile"],
                             "benchmark": prepared["benchmark"]}) for correct in (True, False)]
    details = []
    for item in prepared["items"]:
        record = found[item["id"]]
        response = record["response"]["choices"][0]
        result = worker({"profile": prepared["profile"], "benchmark": prepared["benchmark"],
                         "item": item, "text": response["message"]["content"],
                         "outer_timeout": max(180, 10 * (len(item.get("tests", [])) + 2) + 30)})
        if type(result.get("passed")) is not bool:
            raise ValueError("Scorer did not return a boolean pass")
        details.append({"id": item["id"], "passed": result["passed"], "finish_reason": response["finish_reason"]})
    write_new(output, {"contract": contract, "status": "completed", "qualification": qualification,
                       "scoring_runner_sha256": file_hash(__file__), "worker_sha256": file_hash(HERE / "contract_worker.py"),
                       "responses_sha256": file_hash(root / "responses.jsonl"), "total": len(details),
                       "score": sum(x["passed"] for x in details) / len(details), "details": details,
                       "limits": "paper-spec is a documented reconstruction, not an exact author harness reproduction"})
    print(f"SCORE_COMPLETE: {output}")


def summarize(args):
    cells = [read_json(path) for path in args.metrics]
    if any(x["contract"].get("scope") != "full" for x in cells):
        raise ValueError("Smoke results cannot produce a full benchmark summary")
    signatures = {(x["contract"]["checkpoint"]["sha256"], x["contract"]["profile"]) for x in cells}
    if len(signatures) != 1:
        raise ValueError("Summary must contain one checkpoint and one profile")
    result = {}
    for benchmark in CAPS:
        rows = [x for x in cells if x["contract"]["benchmark"] == benchmark]
        if len(rows) != 5 or {x["contract"]["seed"] for x in rows} != set(SEEDS):
            raise ValueError(f"Need exactly seeds {SEEDS} for {benchmark}")
        if len({x["contract"]["prepared_sha256"] for x in rows}) != 1:
            raise ValueError("Dataset/prompt mismatch across repeats")
        if len({digest(encoded([x["qualification"], x["worker_sha256"], x["scoring_runner_sha256"]])) for x in rows}) != 1:
            raise ValueError("Scorer version differs across repeats")
        if any(x["status"] != "completed" for x in rows):
            raise ValueError("Incomplete cell")
        values = [x["score"] for x in rows]
        result[benchmark] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values),
                             "population_std": statistics.pstdev(values), "scores": values}
    if len(cells) != 20:
        raise ValueError("Unexpected cells")
    write_new(args.out, {"checkpoint_sha256": next(iter(signatures))[0], "profile": next(iter(signatures))[1],
                        "benchmarks": result, "average": statistics.mean(x["mean"] for x in result.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--profile", choices=PROFILES, required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--runtime-root", action="append", default=[])
    p.add_argument("--benchmarks", choices=CAPS, nargs="+", default=list(CAPS))
    p.set_defaults(func=preflight)
    p = sub.add_parser("prepare")
    p.add_argument("--profile", choices=PROFILES, required=True)
    p.add_argument("--benchmark", choices=CAPS, required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--format", choices=("jsonl", "hf-disk", "hf-cache"), required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--config")
    p.add_argument("--release")
    p.add_argument("--expected-count", type=int, required=True)
    p.add_argument("--smoke", action="store_true", help="Explicit small diagnostic; excluded from full summaries")
    p.add_argument("--system-role", choices=("merge", "native"), required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser("generate")
    p.add_argument("--prepared", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--expected-updates", type=int)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True, help="SGLang base URL, without /v1")
    p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    p.add_argument("--concurrency", type=int, choices=range(1, 65), default=8)
    p.add_argument("--out", required=True)
    p.set_defaults(func=generate)
    p = sub.add_parser("score")
    p.add_argument("--prepared", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--runtime-root", action="append", default=[])
    p.set_defaults(func=score)
    p = sub.add_parser("summarize")
    p.add_argument("--metrics", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
