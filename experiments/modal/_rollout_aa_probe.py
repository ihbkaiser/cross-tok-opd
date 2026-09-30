"""Rollout-only A/A, staged: hot repeat -> sleep/wake -> engine restart.

Production RolloutActorGroup path, Gemma2 student only, no teacher, no optimizer.
Compares token sequences AND behavior logprobs separately, and scores a fixed
prefix with prompt logprobs (logprob_start_len=0) at every boundary.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

R5_CONTROL = Path("/runs/p1-hostmask-ab-20260916-r5-control")
OUT_DIR = Path("/runs/p1-parity-20260917")
STUDENT = "/assets/student"

MODES = {
    "baseline": {"deterministic": False, "backend": "", "seed": -1},
    "deterministic": {"deterministic": True, "backend": "", "seed": 42},
    "triton": {"deterministic": True, "backend": "triton", "seed": 42},
}

SHORT_PREFIX_TOKENS = 32
TRACE_WINDOW = 3


def digest_ids(ids) -> str:
    return hashlib.sha256(",".join(str(item) for item in ids).encode()).hexdigest()[:32]


def digest_floats(values) -> str:
    return hashlib.sha256(",".join(f"{float(value):.9g}" for value in values).encode()).hexdigest()[:32]


def build_requests(rows: int):
    source = R5_CONTROL / "checkpoint/rollout_data/1.jsonl"
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    records = records[:rows]
    return ([record["prompt_ids"] for record in records],
            [record.get("output_ids") for record in records])


def main() -> int:
    mode = os.environ.get("P1_AA_MODE", "baseline")
    rows = int(os.environ.get("P1_AA_ROWS", "64"))
    if mode not in MODES:
        raise SystemExit(f"unknown P1_AA_MODE={mode}")
    settings = MODES[mode]

    sys.path.insert(0, "/opt/repo")
    sys.path.insert(0, "/opt/repo/experiments/modal/vendor")

    result: dict = {
        "schema": "simct-p1-rollout-aa-v2",
        "mode": mode, "rows": rows, "settings": settings,
        "status": "starting", "stages": [], "boundaries": [], "probes": [],
        "source": str(R5_CONTROL / "checkpoint/rollout_data/1.jsonl"),
    }

    import ray
    from kdflow.logprob_probe import parse_probe_batch
    from kdflow.ray.placement_group import create_placement_group
    from kdflow.ray.rollout.rollout_group import RolloutActorGroup
    from kdflow.trajectory import bounded_sampling_params
    from kdflow.training_checkpoint import seeded_sampling

    student_config = Path(STUDENT) / "config.json"
    result["student_identity"] = {
        "path": STUDENT,
        "config_sha256": hashlib.sha256(student_config.read_bytes()).hexdigest(),
        "config": json.loads(student_config.read_text()),
        "shards": sorted((p.name, p.stat().st_size) for p in Path(STUDENT).glob("*.safetensors")),
    }
    result["teacher_loaded"] = False

    prompt_ids, reference = build_requests(rows)
    base_params = {"max_new_tokens": 4096, "temperature": 0.6, "top_p": 0.95}
    params = seeded_sampling(
        bounded_sampling_params(prompt_ids, base_params, 4096), seed=42, step=1, count=len(prompt_ids)
    )
    result["request_params"] = [dict(item) for item in params[:3]]
    result["prompt_ids_sha"] = [digest_ids(item) for item in prompt_ids]

    from types import SimpleNamespace

    from kdflow.cli.train_kd_on_policy import build_extra_server_args

    rollout_args = SimpleNamespace(
        rollout_disable_piecewise_cuda_graph=True,
        rollout_deterministic_inference=settings["deterministic"],
        rollout_random_seed=settings["seed"],
        rollout_attention_backend=settings["backend"],
        rollout_disable_radix_cache=settings["deterministic"],
    )
    extra_server_args = build_extra_server_args(SimpleNamespace(rollout=rollout_args))
    result["driver_rollout_args"] = vars(rollout_args)
    result["extra_server_args"] = extra_server_args

    try:
        from sglang.srt.server_args import ServerArgs

        launch = {
            "model_path": STUDENT, "trust_remote_code": True, "host": "127.0.0.1",
            "port": 15000, "nccl_port": 15001, "dist_init_addr": "127.0.0.1:15002",
            "tp_size": 1, "base_gpu_id": 0, "gpu_id_step": 1, "node_rank": 0, "nnodes": 1,
            "enable_memory_saver": True, "enable_weights_cpu_backup": True,
            "skip_server_warmup": True, "log_level": "warning", "log_level_http": "warning",
        }
        launch.update(extra_server_args)
        normalized = ServerArgs(**launch)
        result["server_args_normalized"] = {
            name: str(getattr(normalized, name, "<absent>")) for name in (
                "enable_deterministic_inference", "random_seed", "sampling_backend",
                "attention_backend", "prefill_attention_backend", "decode_attention_backend",
                "chunked_prefill_size", "disable_radix_cache", "cuda_graph_max_bs",
                "disable_cuda_graph", "disable_piecewise_cuda_graph", "dtype",
            )
        }
        result["backend_consistency"] = {
            "sampling_backend_is_pytorch": str(getattr(normalized, "sampling_backend", "")) == "pytorch",
            "prefill_backend_none": getattr(normalized, "prefill_attention_backend", None) in (None, "None"),
            "decode_backend_none": getattr(normalized, "decode_attention_backend", None) in (None, "None"),
            "attention_backend": str(getattr(normalized, "attention_backend", "")),
        }
    except Exception as exc:
        result["server_args_normalized_error"] = repr(exc)

    ray.init(
        runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
        include_dashboard=False,
    )
    pg = create_placement_group(1)

    def make_group():
        return RolloutActorGroup(
            model_path=STUDENT,
            extra_server_args=dict(extra_server_args),
            num_actors=1, tp_size=1, num_gpus_per_node=1,
            enable_memory_saver=True, mem_fraction_static=0.25,
            num_gpus_per_actor=0.3, pg=pg,
        )

    def generate(group, tag: str) -> dict:
        started = time.time()
        outputs = group.generate(["" for _ in prompt_ids], params, None, prompt_ids)
        elapsed = time.time() - started
        entries = []
        for index, output in enumerate(outputs):
            if not output:
                entries.append({"row_index": index, "missing": True})
                continue
            meta = output.get("meta_info") or {}
            ids = output.get("output_ids") or []
            logprobs = meta.get("output_token_logprobs") or []
            values = [row[0] for row in logprobs if isinstance(row, (list, tuple)) and row]
            entries.append({
                "row_index": index,
                "output_ids": ids,
                "output_ids_sha": digest_ids(ids),
                "output_ids_len": len(ids),
                "output_logprobs": values,
                "output_logprobs_sha": digest_floats(values) if values else None,
                "finish_reason": meta.get("finish_reason"),
                "weight_version": meta.get("weight_version"),
                "cached_tokens": meta.get("cached_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "prompt_tokens": meta.get("prompt_tokens"),
            })
        print(f"AA_GENERATE tag={tag} seconds={elapsed:.2f}", flush=True)
        return {"tag": tag, "seconds": elapsed, "entries": entries}

    def score_prefix(group, tag: str, prefix_ids, seed: int) -> dict:
        probe_params = [{"max_new_tokens": 1, "temperature": 0.6, "top_p": 0.95,
                         "sampling_seed": seed}]
        outputs = group.generate([""], probe_params, None, [list(prefix_ids)],
                                 logprob_start_len=0)
        verdict = parse_probe_batch(outputs, [list(prefix_ids)])
        meta = (outputs[0] or {}).get("meta_info") or {}
        series = meta.get("input_token_logprobs") or []
        values = [row[0] for row in series
                  if isinstance(row, (list, tuple)) and row and row[0] is not None]
        record = {
            "tag": tag, "prefix_tokens": len(prefix_ids), "verdict": verdict,
            "entries": len(series),
            "fingerprint": digest_floats(values) if values else None,
            "sample": [round(float(value), 6) for value in values[:4]],
            "weight_version": meta.get("weight_version"),
        }
        print(f"AA_PROBE tag={tag} verdict={verdict['status']} reason={verdict.get('reason')}", flush=True)
        return record

    def compare(left: dict, right: dict) -> dict:
        left_entries = {entry["row_index"]: entry for entry in left.get("entries", [])}
        right_entries = {entry["row_index"]: entry for entry in right.get("entries", [])}
        shared = sorted(set(left_entries) & set(right_entries))
        token_exact = 0
        logprob_exact = 0
        missing = len(set(range(rows)) - set(shared))
        first_token_divergence = None
        first_logprob_divergence = None
        max_logprob_delta = 0.0
        for index in shared:
            a, b = left_entries[index], right_entries[index]
            if a.get("missing") or b.get("missing"):
                missing += 1
                continue
            if a["output_ids"] == b["output_ids"]:
                token_exact += 1
            elif first_token_divergence is None:
                common = 0
                for x, y in zip(a["output_ids"], b["output_ids"]):
                    if x != y:
                        break
                    common += 1
                first_token_divergence = {"row_index": index, "token_position": common}
            lv_a, lv_b = a.get("output_logprobs") or [], b.get("output_logprobs") or []
            if lv_a and lv_a == lv_b:
                logprob_exact += 1
            else:
                for x, y in zip(lv_a, lv_b):
                    try:
                        max_logprob_delta = max(max_logprob_delta, abs(float(x) - float(y)))
                    except (TypeError, ValueError):
                        continue
                if first_logprob_divergence is None:
                    common = 0
                    for x, y in zip(lv_a, lv_b):
                        if x != y:
                            break
                        common += 1
                    first_logprob_divergence = {
                        "row_index": index, "logprob_position": common,
                        "left_len": len(lv_a), "right_len": len(lv_b),
                        "left_sha": a.get("output_logprobs_sha"), "right_sha": b.get("output_logprobs_sha"),
                    }
        return {
            "left": left["tag"], "right": right["tag"],
            "rows_in_batch": rows, "rows_compared": len(shared), "rows_missing": missing,
            "token_exact_rows": token_exact,
            "logprob_exact_rows": logprob_exact,
            "max_abs_logprob_delta": max_logprob_delta,
            "first_token_divergence": first_token_divergence,
            "first_logprob_divergence": first_logprob_divergence,
        }

    def trace(group, rows_index: int, common_prefix: int) -> dict:
        """Re-score the fixed common prefix of the first divergent row."""
        prefix = list(prompt_ids[rows_index])
        return {"row_index": rows_index, "common_prefix": common_prefix,
                "prompt_sha": digest_ids(prompt_ids[rows_index]),
                "seed": params[rows_index]["sampling_seed"],
                "prefix_probe": score_prefix(group, "trace", prefix, params[rows_index]["sampling_seed"])}

    stage_records = []
    generations = {}
    stopped_at = None
    try:
        group = make_group()
        group.sleep()
        result["status"] = "engine_ready"

        group.wakeup()
        result["boundaries"].append({
            "boundary": "initial_wakeup", "ok": True,
            "weight_version": None, "engine_count": 1,
        })
        short_prefix = prompt_ids[0][:SHORT_PREFIX_TOKENS]
        result["probes"].append(score_prefix(group, "short_prefix_pre_hot", short_prefix,
                                             params[0]["sampling_seed"]))
        generations["A0"] = generate(group, "A0")
        generations["A1"] = generate(group, "A1")
        hot = compare(generations["A0"], generations["A1"])
        stage_records.append({"stage": "hot_repeat", "comparison": hot, "sleep_between": False})
        result["probes"].append(score_prefix(group, "short_prefix_post_hot", short_prefix,
                                             params[0]["sampling_seed"]))
        hot_pass = hot["token_exact_rows"] == rows and hot["logprob_exact_rows"] == rows
        if not hot_pass and hot["first_token_divergence"]:
            stage_records[-1]["first_divergence_trace"] = trace(
                group,
                hot["first_token_divergence"]["row_index"],
                hot["first_token_divergence"]["token_position"],
            )
        if hot_pass:
            group.sleep()
            group.wakeup()
            result["boundaries"].append({"boundary": "sleep_wake", "ok": True, "engine_count": 1})
            result["probes"].append(score_prefix(group, "short_prefix_post_sleepwake", short_prefix,
                                                 params[0]["sampling_seed"]))
            generations["A2"] = generate(group, "A2")
            sleepwake = compare(generations["A0"], generations["A2"])
            stage_records.append({"stage": "sleep_wake", "comparison": sleepwake, "sleep_between": True})
            if sleepwake["token_exact_rows"] == rows and sleepwake["logprob_exact_rows"] == rows:
                group.shutdown()
                group = make_group()
                group.sleep()
                group.wakeup()
                result["boundaries"].append({"boundary": "restart", "ok": True, "engine_count": 1})
                result["probes"].append(score_prefix(group, "short_prefix_post_restart", short_prefix,
                                                     params[0]["sampling_seed"]))
                generations["A3"] = generate(group, "A3")
                restart = compare(generations["A0"], generations["A3"])
                stage_records.append({"stage": "restart", "comparison": restart, "sleep_between": True})
                if restart["token_exact_rows"] != rows or restart["logprob_exact_rows"] != rows:
                    stopped_at = "restart"
            else:
                stopped_at = "sleep_wake"
        else:
            stopped_at = "hot_repeat"
        try:
            group.shutdown()
        except Exception:
            pass
        result["stopped_at_first_failed_stage"] = stopped_at
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc()[-4000:])
    finally:
        result["stages"] = stage_records
        trimmed = []
        for tag, generation in generations.items():
            trimmed.append({
                "tag": tag, "seconds": generation.get("seconds"),
                "entries": [{k: v for k, v in entry.items()
                             if k not in {"output_ids", "output_logprobs"}}
                            for entry in generation.get("entries", [])],
                "output_ids": [entry.get("output_ids") for entry in generation.get("entries", [])],
                "output_logprobs": [entry.get("output_logprobs") for entry in generation.get("entries", [])],
            })
        result["generations"] = trimmed
        if reference:
            result["reference_sha"] = [digest_ids(item) if item else None for item in reference]
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / f"rollout-aa-{mode}-v2.json").write_text(
                json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
        except OSError as exc:
            print(f"AA_WRITE_WARNING {exc!r}", flush=True)
        try:
            ray.shutdown()
        except Exception:
            pass

    marker = {key: value for key, value in result.items()
              if key not in {"generations", "probes", "boundaries"}}
    marker["stage_summaries"] = [
        {"stage": stage["stage"], "comparison": stage["comparison"]} for stage in result["stages"]
    ]
    marker["probe_verdicts"] = [
        {"tag": probe["tag"], "verdict": probe["verdict"]["status"],
         "reason": probe["verdict"].get("reason"), "entries": probe["entries"],
         "fingerprint": probe["fingerprint"]}
        for probe in result.get("probes", [])
    ]
    print("ROLLOUT_AA_JSON=" + json.dumps(marker, sort_keys=True, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
