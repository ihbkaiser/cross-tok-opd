"""Probe the rollout engine's behavior logprobs without training.

The trainer dies at ``_build_exact_rollout_sample`` when the engine reports a
sampled-token log-probability of exactly 0.0 (p=1), which cannot happen for a real
sampled token. This probe answers three questions that a 7-minute training run
cannot separate: does stock SGLang return zeros at all, which request field makes
it, and does it need a long generation or a batch to show up.

It mirrors production exactly on purpose -- same server flags, same
``/generate`` payload shape as ``rollout_group._generate_one`` (``input_ids``,
``return_logprob=True``, ``logprob_start_len=-1``) -- so a clean probe exonerates
this harness rather than a lookalike one. Only the rollout engine is started:
no teacher, no optimizer, no checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "experiments" / "modal" / "vendor") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments" / "modal" / "vendor"))

from kdflow.rollout_serving import build_extra_server_args  # noqa: E402


# Every arm keeps the production payload and changes exactly one request field, so
# the zeros are attributable. `baseline` is the arm the trainer actually uses.
ARMS = {
    "baseline": {},
    "logprob_start_len0": {"logprob_start_len": 0},
    "sampling_logprobs0": {"sampling_logprobs": 0},
    "no_return_logprob": {"return_logprob": False},
    "greedy": {"temperature": 0.0},
    "temperature1": {"temperature": 1.0},
}


def serving_flags() -> dict:
    """Build the same engine flags the launcher hands to SGLang."""
    from types import SimpleNamespace

    return build_extra_server_args(SimpleNamespace(rollout=SimpleNamespace(
        rollout_disable_piecewise_cuda_graph=os.environ.get("MP_DISABLE_PIECEWISE_CUDA_GRAPH", "1") == "1",
        rollout_attention_backend=os.environ.get("MP_ROLLOUT_ATTENTION_BACKEND", ""),
        rollout_deterministic_inference=os.environ.get("MP_ROLLOUT_DETERMINISTIC", "0") == "1",
        rollout_disable_radix_cache=os.environ.get("MP_ROLLOUT_DISABLE_RADIX_CACHE", "0") == "1",
        rollout_random_seed=int(os.environ.get("MP_ROLLOUT_SEED", "-1")),
    )))


def load_prompt_ids(dataset: Path, rows: int, max_len: int) -> list[list[int]]:
    """Encode real prompts so the prefill shape matches the training run."""
    import pandas as pd
    from transformers import AutoTokenizer

    frame = pd.read_parquet(dataset)
    # The phi dataset ships as chat rows (source, messages); other datasets ship a
    # flat text column. Accept both so the probe runs against whatever the campaign
    # case points at instead of dying on a column name.
    column = next((name for name in ("prompt", "question", "text", "input", "messages") if name in frame.columns), None)
    if column is None:
        raise SystemExit("no prompt-like column in dataset: " + ", ".join(map(str, frame.columns)))
    print(f"PROBE_PROMPT_COLUMN={column}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(os.environ["MP_STUDENT_PATH"])
    encoded: list[list[int]] = []
    for value in frame[column].head(rows).tolist():
        if column == "messages":
            turns = [turn.get("content", "") for turn in value if turn.get("role") == "user"]
            if not turns:
                continue
            text = turns[0]
        else:
            text = value if isinstance(value, str) else str(value)
        if not text.strip():
            continue
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True,
            tokenize=True, return_dict=False)
        # return_dict=False can still yield a nested list for batched templates.
        if len(ids) and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        encoded.append([int(token) for token in ids][:max_len])
    if not encoded:
        raise SystemExit("dataset produced no usable prompts")
    return encoded


def start_engine(model: Path, port: int, extra: dict) -> subprocess.Popen:
    command = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", str(model),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--tp-size", "1",
        "--trust-remote-code",
        "--skip-server-warmup",
        "--log-level", "warning",
        "--log-level-http", "warning",
    ]
    for name, value in extra.items():
        flag = name.replace("_", "-")
        command += [f"--{flag}"] + ([] if isinstance(value, bool) and value else [str(value)])
    print("PROBE_ENGINE_CMD=" + " ".join(command), flush=True)
    # Own process group: SGLang spawns worker children, and signalling only the
    # launcher leaves them holding GPU memory, which then blocks the next probe.
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True)
    for _ in range(180):
        if process.poll() is not None:
            raise SystemExit("engine exited early with code " + str(process.returncode))
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                print("PROBE_ENGINE_READY", flush=True)
                return process
        except requests.RequestException:
            pass
        time.sleep(2)
    raise SystemExit("engine never became healthy")


def stop_engine(process: subprocess.Popen) -> None:
    """Terminate the engine and every child it spawned, then confirm it released VRAM."""
    if process.poll() is not None:
        return
    try:
        group = os.getpgid(process.pid)
    except ProcessLookupError:
        group = None
    for sig, grace in ((signal.SIGTERM, 90), (signal.SIGKILL, 60)):
        if process.poll() is not None:
            break
        if group is not None:
            os.killpg(group, sig)
        else:
            process.send_signal(sig)
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue
    print("PROBE_ENGINE_STOPPED", flush=True)


def make_payload(shape: str, ids: list[int], text: str, overrides: dict, max_new_tokens: int) -> dict:
    """Build one request; `shape` picks which optional field is present."""
    sampling = {"max_new_tokens": max_new_tokens, "temperature": 0.6, "top_p": 0.95}
    sampling.update({k: v for k, v in overrides.items() if k in {"temperature", "top_p", "logprobs"}})
    if shape == "text_form":
        payload: dict = {"text": text}
    else:
        payload = {"input_ids": ids}
        if shape == "logprobs_field":
            sampling.setdefault("logprobs", 0)
    payload["sampling_params"] = sampling
    payload["return_logprob"] = overrides.get("return_logprob", True)
    if shape != "no_start_len":
        payload["logprob_start_len"] = overrides.get("logprob_start_len", -1)
    return payload


# Ordered most-faithful first. The native /generate endpoint rejected the production
# shape outright, so the probe must report which shape the server accepts instead of
# dying on raise_for_status() with the server's explanation discarded.
SHAPES = ("production", "no_start_len", "logprobs_field", "text_form")


def negotiate_shape(ids: list[int], text: str, port: int) -> str:
    for shape in SHAPES:
        payload = make_payload(shape, ids, text, {}, 8)
        response = requests.post(f"http://127.0.0.1:{port}/generate", json=payload, timeout=300)
        print(f"PROBE_SHAPE {shape} http={response.status_code} {response.text[:240]}", flush=True)
        if response.status_code == 200:
            print(f"PROBE_SHAPE_ACCEPTED={shape}", flush=True)
            return shape
    raise SystemExit("engine rejected every payload shape")


def one_request(ids: list[int], text: str, overrides: dict, port: int, shape: str, max_new_tokens: int) -> dict:
    payload = make_payload(shape, ids, text, overrides, max_new_tokens)
    response = requests.post(f"http://127.0.0.1:{port}/generate", json=payload, timeout=900)
    if response.status_code != 200:
        raise SystemExit(f"engine rejected {shape}: {response.status_code} {response.text[:400]}")
    return response.json()


def summarize(rows: list[dict], arm: str, concurrency: int) -> dict:
    tokens = zeros = 0
    first_zero = None
    lowest = 0.0
    highest = 0.0
    missing = 0
    for row_index, output in enumerate(rows):
        # SGLang puts the sampled ids at the top level; only the logprobs live in meta_info.
        sampled = list(output.get("output_ids") or output.get("meta_info", {}).get("output_token_ids") or [])
        series = output.get("meta_info", {}).get("output_token_logprobs")
        if series is None:
            missing += 1
            continue
        ids = [int(entry[1]) for entry in series]
        if ids != sampled[:len(ids)]:
            raise SystemExit(f"arm {arm} row {row_index}: logprob ids do not align with sampled ids")
        for position, entry in enumerate(series):
            value = float(entry[0])
            tokens += 1
            if value == 0.0:
                zeros += 1
                if first_zero is None:
                    first_zero = [row_index, position]
            lowest = min(lowest, value)
            highest = max(highest, value)
    report = {
        "arm": arm, "rows": len(rows), "concurrency": concurrency,
        "tokens": tokens, "zeros": zeros,
        "zero_fraction": round(zeros / tokens, 4) if tokens else None,
        "first_zero": first_zero, "min_logprob": lowest, "max_logprob": highest,
        "rows_missing_output_logprobs": missing,
    }
    print(
        "[probe] arm={arm} rows={rows} tokens={tokens} zeros={zeros} frac={frac} "
        "first_zero={first_zero} min={min_logprob} max={max_logprob} missing={rows_missing_output_logprobs}"
        .format(**report),
        flush=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--port", type=int, default=24800)
    parser.add_argument("--arms", default="baseline,logprob_start_len0,sampling_logprobs0,greedy")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = Path(os.environ["MP_STUDENT_PATH"])
    prompts = load_prompt_ids(Path(args.dataset), args.rows, args.max_prompt_len)
    print(f"PROBE_PROMPTS={len(prompts)} lens={[len(ids) for ids in prompts]}", flush=True)

    extra = serving_flags()
    print("PROBE_SERVING_ARGS=" + json.dumps(extra, sort_keys=True), flush=True)

    engine = start_engine(model, args.port, extra)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(os.environ["MP_STUDENT_PATH"])
    texts = [tokenizer.decode(ids) for ids in prompts]
    shape = negotiate_shape(prompts[0], texts[0], args.port)
    result = {"serving_args": extra, "rows": len(prompts), "shape": shape, "arms": []}
    try:
        for arm in args.arms.split(","):
            arm = arm.strip()
            if arm not in ARMS:
                raise SystemExit("unknown arm " + arm)
            overrides = ARMS[arm]
            # The trainer drives the router with a whole batch at once, and SGLang's
            # shared logprob offset advances per forward batch. Probe in parallel so a
            # batch-dependent fault cannot hide behind one request at a time.
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                outputs = list(pool.map(
                    lambda pair: one_request(pair[0], pair[1], overrides, args.port, shape, args.max_new_tokens),
                    list(zip(prompts, texts))))
            result["arms"].append(summarize(outputs, arm, args.concurrency))
    finally:
        stop_engine(engine)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("PROBE_JSON=" + str(out), flush=True)
    for report in result["arms"]:
        print("PROBE_VERDICT arm={arm} zeros={zeros} tokens={tokens}".format(**report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())