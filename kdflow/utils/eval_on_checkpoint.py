"""Generate benchmark cells from the policy resident in the rollout server.

`eval_queue.py worker` always launches its own SGLang server and loads the
checkpoint from disk, because that is the only place it runs. During training
the policy for the current step is already loaded in the trainer's rollout
server, so re-serialising tens of GiB and reading it back only to produce the
same weights is pure overhead.

Generation goes through `rollout_group.generate`, the exact request path the
training rollout uses (same router, same pacing, same batching). An earlier
version drove the actor's `/v1/chat/completions` endpoint directly with a
concurrent burst and killed the scheduler twice on training-state servers
while the same burst survived on clean ones, so the direct path is abandoned:
only the path the training loop itself exercises every step is trusted here.
The router takes rendered text, so the driver renders each item's chat messages
with the tokenizer from the same checkpoint directory the server loaded --
same template file, same string -- and records the original messages payload
unchanged for hashing.

This module produces the same on-disk contract that `run_cell` produces --
`plan.json`, `state.json`, and per cell `contract.json`, `responses.jsonl`,
`generation-complete.json` -- so the existing scoring path picks the cells up
unchanged. What differs is provenance, not output:

* the request payload names the model the server *actually* serves, discovered
  from `/v1/models`, rather than the hardcoded `eval-gemma` a freshly launched
  eval server would report;
* `contract.json["server"]` records that the cells came from the training
  process, together with the step, the checkpoint identity, and the served
  model id.

`run_cell` accepts the served model recorded in the contract (falling back to
`eval-gemma` for legacy cells), so `score-spool` treats these cells exactly
like any other. The server-identity re-check (`verify_server`) is what demands
`model_path` equal a checkpoint directory; it is not applicable when there is
no separate server. Provenance here comes from the step, the source commit,
and the checkpoint hash instead -- the same evidence with a different anchor.

Layout mirrors `queue_full_alternating.eval_plan`: one plan per
`<case>/eval/<run>-step<N>/`, one job per plan, cells at
`cells/<job-id>/<benchmark>/<seed>`.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

__all__ = [
    "DEFAULT_STEPS",
    "parse_steps",
    "served_model_id",
    "server_record",
    "render_prompt_texts",
    "router_sampling_params",
    "adapt_router_output",
    "run",
]

# The historical milestone convention: 40, 80, ..., 280, plus the final 312.
# Note `ec --steps 40..312` stops at 280; 312 must be listed explicitly.
DEFAULT_STEPS = (40, 80, 120, 160, 200, 240, 280, 312)


def parse_steps(text: str) -> List[int]:
    steps = sorted({int(x) for x in str(text).split(",") if x.strip()})
    if any(x <= 0 for x in steps):
        raise ValueError("Eval checkpoint steps must be positive")
    return steps


def _eval_modules():
    """Import the eval scripts without making kdflow depend on them at import time."""
    root = Path(__file__).resolve().parents[2]
    eval_dir = root / "scripts" / "evaluation"
    if not (eval_dir / "contract_eval.py").is_file():
        raise ValueError(f"Eval scripts not found at {eval_dir}")
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    import contract_eval as E
    import queue_data as D
    return E, D


def _json(base: str, suffix: str, payload: Optional[dict] = None, timeout: float = 600.0):
    """POST/GET JSON at the trainer's own rollout server, bypassing any proxy.

    `contract_eval.request_json` refuses anything that is not localhost. That
    guard protects a caller-supplied endpoint; here the endpoint is the
    trainer's own rollout server, built from the run configuration, and it
    listens on the node address rather than 127.0.0.1.
    """
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme != "http":
        raise ValueError(f"Refusing non-http rollout endpoint {base!r}")
    request = urllib.request.Request(
        base.rstrip("/") + suffix,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def served_model_id(base_url: str) -> str:
    """Return the single model id the rollout server actually serves.

    Exactly one id must be advertised; anything else means the request label
    would be ambiguous, and ambiguous provenance fails closed.
    """
    data = _json(base_url, "/v1/models", timeout=60.0)
    ids = [item["id"] for item in data.get("data", [])]
    if len(ids) != 1:
        raise ValueError(
            f"Rollout server must advertise exactly one model id, got {ids!r}"
        )
    return ids[0]


def server_record(*, served_model: str, step: int, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "kind": "in-process-rollout",
        "served_model": served_model,
        "step": step,
        "optimizer_updates": step,
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_path": checkpoint["path"],
        "source_commit": os.environ.get("MP_SOURCE_COMMIT"),
        "note": (
            "Generated by the training process against its resident policy; "
            "no separate eval server was launched."
        ),
    }


def _atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def _cell_dir(root: Path, job_id: str, benchmark: str, seed: int) -> Path:
    return root / "cells" / job_id / benchmark / str(seed)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_contract(plan_hash: str, job: Dict[str, Any], data: Dict[str, Any],
                   benchmark: str, seed: int, server: Dict[str, Any], profile: str) -> Dict[str, Any]:
    # Same shape as eval_queue.cell_contract; the server record carries the
    # in-process provenance instead of a verify_server snapshot.
    return {
        "plan_sha256": plan_hash,
        "checkpoint_sha256": job["checkpoint"]["sha256"],
        "data_sha256": data[benchmark]["sha256"],
        "benchmark": benchmark,
        "seed": seed,
        "profile": profile,
        "server": server,
    }


def render_prompt_texts(apply_chat_template: Callable[..., str],
                         items: Dict[str, Any]) -> Dict[str, str]:
    """Render each item's messages to the exact text the server would generate from.

    `apply_chat_template` is the tokenizer's bound method; taking it as a
    parameter keeps this pure and unit-testable. It must be the tokenizer from
    the checkpoint directory the serving process loaded, with the same kwargs
    the chat endpoint applies (`add_generation_prompt=True`,
    `enable_thinking=False`), otherwise the rendered text is a different
    request wearing the same hash.
    """
    texts = {}
    for item_id, item in items.items():
        text = apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(text, str) or not text:
            raise ValueError(f"empty rendered prompt for item {item_id!r}")
        texts[item_id] = text
    return texts


def router_sampling_params(E, benchmark: str, payload: dict) -> dict:
    """Sampling params for the router path, carrying the contract seed.

    Temperature/top_p/max_tokens mirror the recorded payload; the per-item
    seed travels as `sampling_seed`, the same key the training rollout uses
    for stateless request RNG.
    """
    return {
        "temperature": payload["temperature"],
        "top_p": payload["top_p"],
        "max_new_tokens": payload["max_tokens"],
        "sampling_seed": payload["seed"],
    }


def adapt_router_output(output: dict) -> dict:
    """Rewrap a router `/generate` output into the OpenAI response envelope.

    The scorer and `validate_response` read `choices[0].finish_reason` and
    `choices[0].message.content`; both come verbatim from the server, only the
    envelope is reconstructed. An unknown finish reason fails loudly instead
    of being coerced, because the truncation accounting depends on it.
    """
    meta = output.get("meta_info") or {}
    reason = meta.get("finish_reason")
    if isinstance(reason, dict):
        reason = reason.get("type")
    if reason not in ("stop", "length"):
        raise ValueError(f"Unexpected router finish reason: {reason!r}")
    text = output.get("text")
    if not isinstance(text, str):
        raise ValueError("Router response missing text")
    return {"choices": [{"finish_reason": reason, "message": {"content": text}}]}


def _generate_cell(*, E, cell: Path, items: Dict[str, Any], benchmark: str, seed: int,
                   served_model: str, contract: Dict[str, Any],
                   render_texts: Callable[[Dict[str, Any]], Dict[str, str]],
                   generate_texts: Callable[[List[str], List[dict]], List[dict]]) -> int:
    """Generate one cell, resuming a valid marker and never repairing a live journal."""
    cell.mkdir(parents=True, exist_ok=True)
    manifest = cell / "contract.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != contract:
            raise ValueError(f"cell resume contract mismatch at {cell}")
    else:
        manifest.write_text(json.dumps(contract, indent=2) + "\n")

    responses_path = cell / "responses.jsonl"
    marker = cell / "generation-complete.json"
    if marker.exists():
        value = json.loads(marker.read_text())
        if (
            value["contract"] != contract
            or value["count"] != len(items)
            or value["responses_sha256"] != _sha256_file(responses_path)
        ):
            raise ValueError(f"completed generation spool mismatch at {cell}")
        return len(items)

    if responses_path.exists():
        # An interrupted spool without a marker is evidence, not garbage: refuse
        # to silently extend it, exactly like the queue path refuses repairs.
        raise ValueError(
            f"interrupted generation spool without a completion marker at {cell}; "
            "inspect it before retrying"
        )

    # The recorded payload is unchanged from the direct-chat version: same
    # messages, same seed rule, same hash. Only the transport differs.
    ordered = list(items.values())
    payload_of = {}
    for item in ordered:
        payload_of[item["id"]] = E.generation_payload(served_model, item, benchmark, seed)
    texts = render_texts(items)
    params = [router_sampling_params(E, benchmark, payload_of[item["id"]]) for item in ordered]
    prompt_texts = [texts[item["id"]] for item in ordered]
    outputs = generate_texts(prompt_texts, params)
    if len(outputs) != len(ordered):
        raise RuntimeError(
            f"eval cell {benchmark}/{seed} returned {len(outputs)} of {len(ordered)} outputs"
        )
    with responses_path.open("w") as handle:
        for item, output in zip(ordered, outputs):
            response = adapt_router_output(output)
            E.validate_response(response)
            row = {
                "id": item["id"],
                "seed": seed,
                "request_sha256": E.digest(E.encoded(payload_of[item["id"]])),
                "response": response,
            }
            handle.write(json.dumps(row) + "\n")
            handle.flush()

    rows = [json.loads(line) for line in responses_path.read_text().splitlines() if line.strip()]
    if len(rows) != len(items):
        raise RuntimeError(f"eval cell {benchmark}/{seed} produced {len(rows)} of {len(items)} responses")
    for row in rows:
        expected = payload_of[row["id"]]
        if row["request_sha256"] != E.digest(E.encoded(expected)):
            raise ValueError("response request mismatch; refusing to mark generation complete")

    _atomic_json(marker, {
        "contract": contract,
        "count": len(items),
        "responses_sha256": _sha256_file(responses_path),
    })
    return len(items)


def run(trainer, *, step_dir: str) -> Dict[str, Any]:
    """Generate the 3-seed eval cells for the current optimizer step.

    Called from the trainer right after the checkpoint save, while the rollout
    server still holds exactly this step's weights. Blocking is intentional:
    the generations must belong to this step, not a later one.
    """
    E, D = _eval_modules()
    args = trainer.args.train
    step = int(trainer.completed_optimizer_updates)

    case_dir = Path(args.eval_case_dir)
    prepared_dir = Path(args.eval_prepared_dir)
    if not case_dir.is_dir():
        raise ValueError(f"eval_case_dir is not a directory: {case_dir}")
    benchmarks = ["gsm8k", "math500", "mbpp", "live-code-bench-v6"]
    seeds = [42, 43, 44]
    data: Dict[str, Any] = {}
    for benchmark in benchmarks:
        path = prepared_dir / f"{benchmark}.json"
        if not path.is_file():
            raise ValueError(f"prepared eval data missing: {path}")
        dataset = json.loads(path.read_text())
        if dataset.get("profile") != D.PROFILE:
            raise ValueError(f"prepared data profile mismatch at {path}")
        if len(dataset.get("items", [])) != E.COUNTS[benchmark]:
            raise ValueError(f"prepared data count mismatch at {path}")
        data[benchmark] = {
            "path": str(path.resolve()),
            "sha256": E.file_hash(path),
            "count": len(dataset["items"]),
            "source": dataset.get("source"),
        }

    checkpoint = E.checkpoint_identity(step_dir)
    # save_path is <run>/checkpoint in the standard layout, so the run name is
    # its parent; otherwise fall back to the directory itself.
    save_root = Path(args.save_path).resolve()
    run_name = save_root.parent.name if save_root.name == "checkpoint" else save_root.name
    job_id = f"{run_name}-step{step}"
    mode = str(getattr(trainer.args.kd, "mp_opd_mode", "unknown"))
    job = {"id": job_id, "mode": mode, "step": step, "tier": 0, "checkpoint": checkpoint}

    plan = {
        "schema": "eval-queue-v1",
        "profile": D.PROFILE,
        "seeds": seeds,
        "data": data,
        "jobs": [job],
        # Protocol text mirrors queue_data.prepare; the hashed `source` below
        # is what the queue actually verifies.
        "protocol": {
            "context_length": 8192, "temperature": 0.6, "top_p": 0.95, "n": 1, "caps": E.CAPS,
            "math": "author helpers, not math-verify",
            "mbpp": "author assertion helper 10 seconds/problem",
            "lcb": "pinned official tester; public+private; functional-specific prompt",
            "code_execution": "explicit internal subprocess, NOT namespace/container isolation",
            "limits": "2 GiB address space, 120s CPU/problem, <=120s wall/problem",
            "scope": "3 evaluation seeds, exploratory checkpoint selection, not exact paper replication",
        },
        "source": D.script_hashes(),
        "hours": 168,
        "admit_hours": 168,
    }

    root = case_dir / "eval" / f"{run_name}-step{step}"
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        if existing != plan:
            raise ValueError(f"eval plan drift at {plan_path}")
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    plan_hash = E.file_hash(plan_path)

    items_by_bench = {
        benchmark: {item["id"]: item for item in json.loads(Path(data[benchmark]["path"]).read_text())["items"]}
        for benchmark in benchmarks
    }

    import ray

    base_url = ray.get(trainer.rollout_group.actors[0].get_server_url.remote())
    model = served_model_id(base_url)
    server = server_record(served_model=model, step=step, checkpoint=checkpoint)
    server["generation_path"] = "rollout-router-generate"

    # The tokenizer must come from the checkpoint directory the serving
    # process loaded: same template file, same rendered string. Anything else
    # is a different request wearing the same hash.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        trainer.rollout_group.model_path, trust_remote_code=True
    )
    render_texts = lambda items: render_prompt_texts(tokenizer.apply_chat_template, items)

    def generate_texts(prompt_texts: List[str], params: List[dict]) -> List[dict]:
        outputs = trainer.rollout_group.generate(prompt_texts, params)
        if len(outputs) != len(prompt_texts):
            raise RuntimeError(
                f"router returned {len(outputs)} outputs for {len(prompt_texts)} prompts"
            )
        return outputs

    woke = False
    if bool(getattr(args, "enable_sleep", False)):
        trainer.rollout_group.wakeup(tags=["weights"])
        woke = True
    try:
        counts: Dict[str, int] = {}
        for benchmark in benchmarks:
            for seed in seeds:
                contract = _cell_contract(plan_hash, job, data, benchmark, seed, server, D.PROFILE)
                counts[f"{benchmark}/{seed}"] = _generate_cell(
                    E=E, cell=_cell_dir(root, job_id, benchmark, seed),
                    items=items_by_bench[benchmark], benchmark=benchmark, seed=seed,
                    served_model=model, contract=contract,
                    render_texts=render_texts, generate_texts=generate_texts,
                )
    finally:
        if woke:
            trainer.rollout_group.sleep(tags=["weights"])

    # state.json mirrors queue_full_alternating.eval_plan so score-spool runs
    # unchanged; qualification is measured here, once, on this machine.
    # _eval_modules() already put scripts/evaluation on sys.path, so both
    # imports below resolve to the same files whose hashes went into the plan.
    state_path = root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("plan_sha256") != plan_hash:
            raise ValueError(f"eval state drift at {state_path}")
    else:
        from eval_queue import preflight as _preflight

        _atomic_json(state_path, {
            "plan_sha256": plan_hash,
            "started": time.time(),
            "deadline": time.time() + 168 * 3600,
            "admit_until": time.time() + 168 * 3600,
            "jobs": {},
            "durations": [],
            "qualification": _preflight(str(getattr(args, "eval_score_python", "/usr/bin/python3.12"))),
        })

    return {"step": step, "root": str(root), "served_model": model, "cells": counts}
