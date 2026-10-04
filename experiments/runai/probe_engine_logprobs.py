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
    column = next((name for name in ("prompt", "question", "text", "input") if name in frame.columns), None)
    if column is None:
        raise SystemExit("no prompt-like column in dataset: " + ", ".join(map(str, frame.columns)))
    print(f"PROBE_PROMPT_COLUMN={column}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(os.environ["MP_STUDENT_PATH"])
    encoded: list[list[int]] = []
    for value in frame[column].head(rows).tolist():
        text = value if isinstance(value, str) else str(value)
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        encoded.append(list(ids)[:max_len])
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
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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


def summarize(rows: list[dict], arm: str) -> dict:
    tokens = zeros = 0
    first_zero = None
    lowest = 0.0
    highest = 0.0
    missing = 0
    for row_index, output in enumerate(rows):
        sampled = list(output.get("meta_info", {}).get("output_token_ids") or [])
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
        "arm": arm, "rows": len(rows), "tokens": tokens, "zeros": zeros,
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
    result = {"serving_args": extra, "rows": len(prompts), "arms": []}
    try:
        for arm in args.arms.split(","):
            arm = arm.strip()
            if arm not in ARMS:
                raise SystemExit("unknown arm " + arm)
            overrides = ARMS[arm]
            outputs = []
            for ids in prompts:
                payload = {
                    "input_ids": ids,
                    "sampling_params": {
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": 0.6,
                        "top_p": 0.95,
                    },
                    "return_logprob": True,
                    "logprob_start_len": -1,
                }
                payload["sampling_params"].update(
                    {k: v for k, v in overrides.items() if k in {"temperature", "top_p", "logprobs"}})
                payload["logprob_start_len"] = overrides.get("logprob_start_len", -1)
                payload["return_logprob"] = overrides.get("return_logprob", True)
                response = requests.post(
                    f"http://127.0.0.1:{args.port}/generate", json=payload, timeout=900)
                response.raise_for_status()
                outputs.append(response.json())
            result["arms"].append(summarize(outputs, arm))
    finally:
        engine.send_signal(signal.SIGTERM)
        try:
            engine.wait(timeout=90)
        except subprocess.TimeoutExpired:
            engine.kill()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("PROBE_JSON=" + str(out), flush=True)
    for report in result["arms"]:
        print("PROBE_VERDICT arm={arm} zeros={zeros} tokens={tokens}".format(**report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())