"""Measure Gemma-2 SGLang/HF log-probability parity on one GPU."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import gc
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any


TEMPERATURE = 0.6
TOP_P = 0.95
MAX_NEW_TOKENS = 900
PROMPTS = (
    "Solve carefully and explain every step: A steel bar is 20 kg heavier than a "
    "90 kg copper bar and weighs twice as much as a tin bar. Find the total mass "
    "of 20 bars of each metal.",
    "Write a Python function that finds the longest strictly increasing subsequence "
    "of a list, explain its invariants, complexity, edge cases, and include tests.",
)


def _values(entries: list[list[Any]]) -> list[float]:
    return [float(item[0]) for item in entries]


def _ids(entries: list[list[Any]]) -> list[int]:
    return [int(item[1]) for item in entries]


def _stats(left: list[float], right: list[float]) -> dict[str, Any]:
    import torch

    assert len(left) == len(right) and left
    delta = (torch.tensor(left, dtype=torch.float64) - torch.tensor(right, dtype=torch.float64)).abs()
    worst = int(delta.argmax().item())
    return {
        "tokens": int(delta.numel()),
        "mean": float(delta.mean().item()),
        "p99": float(torch.quantile(delta, 0.99).item()),
        "p999": float(torch.quantile(delta, 0.999).item()),
        "max": float(delta[worst].item()),
        "above_0p1": int((delta > 0.1).sum().item()),
        "above_0p5": int((delta > 0.5).sum().item()),
        "worst_flat_position": worst,
        "left": float(left[worst]),
        "right": float(right[worst]),
    }


def _flatten(values: list[list[float]]) -> list[float]:
    return [item for row in values for item in row]


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)) or any(
        isinstance(item, (list, tuple, dict)) for item in value
    ):
        raise TypeError(f"Expected one flat token-ID sequence, got {type(value).__name__}")
    return [int(item) for item in value]


def _engine_kwargs(
    model_path: str,
    gpu_name: str,
    backend: str,
    rl_on_policy_target: str | None,
) -> dict[str, Any]:
    fraction = 0.52 if "A10" in gpu_name else 0.25
    result: dict[str, Any] = {
        "model_path": model_path,
        "trust_remote_code": False,
        "tp_size": 1,
        "mem_fraction_static": fraction,
        "skip_server_warmup": True,
        "disable_piecewise_cuda_graph": True,
        "log_level": "warning",
    }
    if backend != "default":
        result["attention_backend"] = backend
    if rl_on_policy_target is not None:
        result["rl_on_policy_target"] = rl_on_policy_target
    return result


def _server_info(engine: Any) -> dict[str, Any]:
    args = getattr(engine, "server_args", None)
    result = {}
    for name in (
        "attention_backend",
        "sampling_backend",
        "enable_deterministic_inference",
        "rl_on_policy_target",
        "disable_cuda_graph",
        "disable_piecewise_cuda_graph",
        "mem_fraction_static",
        "dtype",
    ):
        if args is not None and hasattr(args, name):
            value = getattr(args, name)
            result[name] = value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
    return result


def _hf_scores(
    model_path: str, tokenized: list[dict[str, Any]], backend: str
) -> dict[str, list[list[float]]]:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=backend,
    ).to("cuda").eval()
    scores: dict[str, list[list[float]]] = {"raw": [], "temperature": []}
    with torch.inference_mode():
        for record in tokenized:
            prompt_ids = record["prompt_ids"]
            output_ids = record["output_ids"]
            all_ids = torch.tensor([prompt_ids + output_ids], device="cuda")
            logits = model(input_ids=all_ids, use_cache=False).logits[0]
            response_logits = logits[len(prompt_ids) - 1 : len(prompt_ids) + len(output_ids) - 1]
            labels = torch.tensor(output_ids, device="cuda")
            for name, temperature in (("raw", 1.0), ("temperature", TEMPERATURE)):
                selected = (
                    (response_logits.float() / temperature)
                    .log_softmax(dim=-1)
                    .gather(-1, labels[:, None])
                    .squeeze(-1)
                )
                scores[name].append(selected.cpu().tolist())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return scores


def run(
    model_path: str,
    output_path: Path,
    backends: list[str],
    rl_on_policy_target: str | None,
) -> dict[str, Any]:
    import sglang
    import torch
    import transformers
    from transformers import AutoTokenizer

    started = time.monotonic()
    gpu_name = torch.cuda.get_device_name(0)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    prompt_ids = [
        _token_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        ))
        for prompt in PROMPTS
    ]
    environment = {
        "gpu": gpu_name,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "sglang": getattr(sglang, "__version__", "unknown"),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "return_original_logprob": os.environ.get("SGLANG_RETURN_ORIGINAL_LOGPROB"),
        "requested_rl_on_policy_target": rl_on_policy_target,
    }

    generated: list[dict[str, Any]] | None = None
    sglang_results: dict[str, Any] = {}
    for backend_index, backend in enumerate(backends):
        engine = None
        try:
            engine = sglang.Engine(
                **_engine_kwargs(model_path, gpu_name, backend, rl_on_policy_target)
            )
            backend_result: dict[str, Any] = {"server_args": _server_info(engine)}
            if backend_index == 0:
                outputs = engine.generate(
                    input_ids=prompt_ids,
                    sampling_params={
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "ignore_eos": True,
                    },
                    return_logprob=True,
                    logprob_start_len=-1,
                )
                outputs = outputs if isinstance(outputs, list) else [outputs]
                generated = []
                for index, output in enumerate(outputs):
                    response_ids = [int(value) for value in output["output_ids"]]
                    entries = output["meta_info"]["output_token_logprobs"]
                    assert _ids(entries) == response_ids
                    generated.append(
                        {
                            "sample": index,
                            "prompt_ids": prompt_ids[index],
                            "output_ids": response_ids,
                            "decode_logprobs": _values(entries),
                            "finish_reason": output["meta_info"].get("finish_reason"),
                        }
                    )
            assert generated is not None
            full_ids = [record["prompt_ids"] + record["output_ids"] for record in generated]
            batch_scores_by_temperature: dict[str, list[list[float]]] = {}
            for name, temperature in (("temperature", TEMPERATURE), ("raw_probe", 1.0)):
                rescored = engine.generate(
                    input_ids=full_ids,
                    sampling_params={"temperature": temperature, "top_p": TOP_P, "max_new_tokens": 0},
                    return_logprob=True,
                    logprob_start_len=0,
                )
                rescored = rescored if isinstance(rescored, list) else [rescored]
                batch_scores: list[list[float]] = []
                for record, output in zip(generated, rescored):
                    entries = output["meta_info"]["input_token_logprobs"]
                    ids = _ids(entries)
                    assert ids == record["prompt_ids"] + record["output_ids"]
                    start = len(record["prompt_ids"])
                    batch_scores.append(_values(entries[start:]))
                batch_scores_by_temperature[name] = batch_scores

            batch_scores = batch_scores_by_temperature["temperature"]

            individual_scores: list[list[float]] = []
            for record, ids in zip(generated, full_ids):
                engine.flush_cache()
                output = engine.generate(
                    input_ids=ids,
                    sampling_params={"temperature": TEMPERATURE, "top_p": TOP_P, "max_new_tokens": 0},
                    return_logprob=True,
                    logprob_start_len=0,
                )
                entries = output["meta_info"]["input_token_logprobs"]
                assert _ids(entries) == ids
                individual_scores.append(_values(entries[len(record["prompt_ids"]):]))

            backend_result["prefill_batch_logprobs"] = batch_scores
            backend_result["prefill_raw_probe_logprobs"] = batch_scores_by_temperature["raw_probe"]
            backend_result["prefill_individual_logprobs"] = individual_scores
            backend_result["prefill_temperature_vs_raw_probe"] = _stats(
                _flatten(batch_scores),
                _flatten(batch_scores_by_temperature["raw_probe"]),
            )
            backend_result["prefill_batch_vs_individual"] = _stats(
                _flatten(batch_scores), _flatten(individual_scores)
            )
            if backend_index == 0:
                backend_result["decode_vs_prefill_batch"] = _stats(
                    _flatten([record["decode_logprobs"] for record in generated]),
                    _flatten(batch_scores),
                )
            sglang_results[backend] = backend_result
        finally:
            if engine is not None:
                engine.shutdown()
            gc.collect()
            torch.cuda.empty_cache()

    assert generated is not None
    hf_results = {
        "eager": _hf_scores(model_path, generated, "eager"),
        "sdpa": _hf_scores(model_path, generated, "sdpa"),
    }
    comparisons: dict[str, Any] = {}
    for backend, values in sglang_results.items():
        for hf_backend, score_sets in hf_results.items():
            for scale_name, scores in score_sets.items():
                comparisons[f"sglang_{backend}_prefill_vs_hf_{hf_backend}_{scale_name}"] = _stats(
                    _flatten(values["prefill_individual_logprobs"]), _flatten(scores)
                )
    for scale_name in ("raw", "temperature"):
        comparisons[f"hf_eager_vs_sdpa_{scale_name}"] = _stats(
            _flatten(hf_results["eager"][scale_name]),
            _flatten(hf_results["sdpa"][scale_name]),
        )
    first = backends[0]
    for scale_name, scores in hf_results["eager"].items():
        comparisons[f"sglang_{first}_decode_vs_hf_eager_{scale_name}"] = _stats(
            _flatten([record["decode_logprobs"] for record in generated]),
            _flatten(scores),
        )

    result = {
        "status": "completed",
        "environment": environment,
        "backends": backends,
        "samples": generated,
        "sglang": sglang_results,
        "hf": hf_results,
        "comparisons": comparisons,
        "elapsed_seconds": time.monotonic() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    compact = {
        "status": result["status"],
        "environment": environment,
        "backends": backends,
        "comparisons": comparisons,
        "elapsed_seconds": result["elapsed_seconds"],
    }
    print("MP_PARITY_SUMMARY_JSON=" + json.dumps(compact, sort_keys=True), flush=True)
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", required=True)
    parser.add_argument("--rl-on-policy-target", choices=("none", "fsdp"), default="none")
    args = parser.parse_args()
    backends = [value.strip() for value in args.backends.split(",") if value.strip()]
    if not backends or backends[0] != "default":
        raise ValueError("The first backend must be default so it owns the sampled trajectory")
    target = None if args.rl_on_policy_target == "none" else args.rl_on_policy_target
    run(args.model_path, args.output, backends, target)


if __name__ == "__main__":
    main()
