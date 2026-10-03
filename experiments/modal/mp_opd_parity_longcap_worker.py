"""Long-cap Gemma-2 SGLang/HF log-probability parity probe (single GPU).

Context: `BUG_parity_logprob_4096cap.md`. The MP-OPD guard compares SGLang
output-token logprobs (decode path) against the trainer's eager recomputation.
A 900-token probe on this same pinned image showed the residual disagreement is
small (mean ~0.019, max ~0.36, zero tokens above 0.5). The production failure
happens only on the sample that runs to `generate_max_len=4096`.

This worker therefore **forces** the cap (`ignore_eos=True`, `max_new_tokens`
configurable) and measures, per engine configuration, on the *same* trajectory:

  decode          engine output-token logprobs (temperature-scaled by SGLang)
  prefill_raw     engine input-token logprobs of the response, rescored from the
                  full sequence in one prefill (SGLang keeps these raw)
  hf_eager_raw    HF eager logits, log_softmax at T=1
  hf_eager_T      HF eager logits, log_softmax at T=rollout temperature
  hf_sdpa_raw/T   same with the sdpa attention implementation

The production guard metric is `decode vs hf_eager_T`. `prefill_raw vs
hf_eager_raw` isolates the prefill attention kernel; the difference between the
two comparisons separates the decode/KV-cache path from the prefill path.

Diagnostics that matter for the root cause:

  * per-band (256-token) statistics over the response;
  * the exact positions above 0.5 plus their consecutive-difference histogram
    (the bug note reports a stride-8 pattern, which must be confirmed or killed);
  * error stratified by HF entropy and by the top-2 logit margin, because a
    numerically perturbed logit only shows up in the logprob where the next-token
    distribution is flat;
  * full arrays are written so no second GPU run is needed for re-analysis.

No company prompt, dataset or parity capture is involved: the prompts are
written here and the model is the pinned public base checkpoint.
"""

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
DEFAULT_MAX_NEW_TOKENS = 4096
BAND = 256
ENTROPY_BINS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9)
MARGIN_BINS = (0.0, 0.05, 0.25, 1.0, 4.0, 1e9)

# Long-answer prompts: the point is to reach the cap with diverse (non-degenerate)
# text so the next-token distribution stays flat in the middle of the response.
PROMPTS = (
    "Write a long, detailed technical explanation of how a modern GPU executes a "
    "matrix multiplication, covering tiling, shared memory, register blocking, "
    "occupancy, warp scheduling, tensor cores and memory bandwidth. Be exhaustive "
    "and keep going for as long as you can.",
    "Write a long, detailed textbook chapter section about the water cycle, "
    "covering evaporation, transpiration, condensation, precipitation, runoff, "
    "groundwater, and the role of oceans and ice. Include many specific numbers "
    "and keep writing for as long as you can.",
)

# Topics used only to pad a batch up to the production rollout width
# (rollout_batch_size=64). The engine's decode-attention KV-split schedule depends
# on the whole batch, so batch composition is a real variable, not decoration.
BATCH_TOPICS = (
    "the history of the Roman aqueducts", "photosynthesis in C4 plants",
    "the mathematics of Fourier series", "TCP congestion control algorithms",
    "plate tectonics and subduction zones", "the chemistry of nitrogen fixation",
    "compiler register allocation", "Bayesian posterior inference",
    "the geology of the Grand Canyon", "immune system antibody diversity",
    "public-key cryptography and RSA", "ocean thermohaline circulation",
    "the structure of the human nephron", "sorting algorithm lower bounds",
    "atmospheric radiative transfer", "the economics of central banking",
    "CRISPR-Cas9 repair pathways", "graph colouring and the four-colour theorem",
    "semiconductor photolithography", "the physiology of bird migration",
    "linear programming duality", "volcanic eruption prediction",
    "the chemistry of Portland cement", "distributed consensus with Raft",
    "epigenetic gene regulation", "the physics of superconductivity",
    "hydraulic engineering of canals", "the history of the printing press",
    "neutrino oscillation experiments", "enzyme kinetics and inhibition",
    "the design of modern CPUs", "soil formation and pedogenesis",
    "the mathematics of Markov chains", "antibiotic resistance mechanisms",
    "the geology of mid-ocean ridges", "quantum error correction codes",
    "the biology of coral reefs", "reinforcement learning value estimation",
    "the chemistry of batteries", "wind turbine aerodynamics",
    "the history of the Silk Road", "protein folding thermodynamics",
    "cryptographic hash function design", "the physiology of the heart",
    "glacier dynamics and ice sheets", "the theory of NP-completeness",
    "radio telescope interferometry", "the chemistry of photosynthesis",
    "municipal water treatment", "the mathematics of option pricing",
    "neural crest cell development", "the geology of karst landscapes",
    "optical fibre communication", "the history of antibiotics",
    "turbulence modelling in CFD", "the biology of extremophiles",
    "lithium-ion cell degradation", "the mathematics of wavelets",
    "seismic tomography methods", "the chemistry of catalysis",
    "swarm robotics coordination", "the history of cartography",
)


def prompt_set(n: int, style: str = "coherent", pad_tokens: int = 0,
               tokenizer: Any = None) -> list[str]:
    """The fixed prompts, optionally lengthened to a target token count.

    Production's failing row had a 239-token prompt, so its engine sequence was
    239 + 3856 = 4095 tokens. This probe's own prompts are ~50 tokens, which leaves
    prompt length as the last uncontrolled contract difference; `pad_tokens` closes
    it without changing the question the prompt asks.
    """
    out = _prompt_set_raw(n, style)
    if pad_tokens <= 0 or tokenizer is None:
        return out
    filler = ("The following context is administrative filler and carries no "
              "instruction: this paragraph exists only to lengthen the prompt so "
              "that the engine's sequence length matches the production case. ")
    padded = []
    for prompt in out:
        text = prompt
        for _ in range(200):
            if len(tokenizer(text, return_tensors=None)["input_ids"]) >= pad_tokens:
                break
            text = filler + text
        padded.append(text)
    return padded


def _prompt_set_raw(n: int, style: str = "coherent") -> list[str]:
    """The fixed prompts first, then topic-templated long-answer prompts."""
    if style == "flat":
        out = list(FLAT_PROMPTS)
        while n > 0 and len(out) < n:
            out.append(FLAT_PROMPTS[len(out) % len(FLAT_PROMPTS)])
        return out[:n] if n > 0 else out
    if style == "repeat":
        out = list(REPEAT_PROMPTS)
        while n > 0 and len(out) < n:
            out.append(REPEAT_PROMPTS[len(out) % len(REPEAT_PROMPTS)])
        return out[:n] if n > 0 else out
    out = list(PROMPTS)
    for topic in BATCH_TOPICS:
        if len(out) >= n:
            break
        out.append(
            f"Write a very long, detailed, structured explanation of {topic}. "
            "Cover background, mechanisms, key quantities, worked examples, common "
            "misconceptions and open questions. Keep writing for as long as you can."
        )
    return out[:n] if n > 0 else out


# Prompts that force a *flat* next-token distribution for thousands of tokens.
# This is the regime the guard actually fails in: a logprob error is proportional
# to the logit error times the flatness of the distribution, so a coherent answer
# (entropy < 4 nats, as measured) cannot exercise it while a free-choice list can.
FLAT_PROMPTS = (
    "List as many distinct random English nouns as you can, one per line, separated "
    "by newlines. Choose each word freely and unpredictably, never repeat a word, "
    "never stop, and keep going for as long as you can.",
    "Generate a long list of unrelated random English words, exactly one word per "
    "line. Pick each word independently at random, do not organise them by topic, "
    "do not repeat, and keep going for as long as you can.",
)

# The production failing sample was a *degenerate* capped generation: the guard's
# 150 outliers sat at stride 8 inside one contiguous band. A short-period
# repetition repeats the one numerically hard position of the cycle over and over,
# which is what turns a handful of ~1.5-nat kernel disagreements into a p99 above
# the guard's threshold. This style forces that trajectory on purpose.
REPEAT_PROMPTS = (
    "Repeat exactly these eight words in this exact order, separated by single "
    "spaces, and nothing else: alpha beta gamma delta epsilon zeta eta theta. "
    "When you reach theta, start again from alpha. Never stop and never add any "
    "other text.",
    "Output only this eight-word sequence over and over with single spaces between "
    "the words: one two three four five six seven eight. After eight, continue with "
    "one again. Keep repeating forever without adding anything else.",
)


# The production serving contract, read off the failing run's engine log
# (BUG_parity_logprob_4096cap.md section 2.5).
PRODUCTION_KWARGS: dict[str, Any] = {
    "attention_backend": "triton",
    "disable_piecewise_cuda_graph": False,
    "disable_cuda_graph": True,
    "chunked_prefill_size": 16384,
    "max_prefill_tokens": 16384,
    "context_length": 8192,
    "mem_fraction_static": 0.25,
    "skip_server_warmup": True,
}

CONFIGS: dict[str, dict[str, Any]] = {
    "prod": {},
    # The bug note's test #1 named `triton_attention_reduce_in_fp32`, which is a dead
    # flag in 0.5.11. The REAL knob for the Triton decode kernel's split-KV reduction
    # is `triton_attention_num_kv_splits` (default 8), together with
    # `triton_attention_split_tile_size`. These arms vary the actual reduction
    # structure instead of a flag that nothing reads: splits=1 removes the split-KV
    # reduction entirely, and 16/32 make it deeper.
    "prod_splits1": {"triton_attention_num_kv_splits": 1},
    "prod_splits2": {"triton_attention_num_kv_splits": 2},
    "prod_splits16": {"triton_attention_num_kv_splits": 16},
    "prod_splits32": {"triton_attention_num_kv_splits": 32},
    "prod_reduce_fp32": {"triton_attention_reduce_in_fp32": True},
    "prod_piecewise_off": {"disable_piecewise_cuda_graph": True},
    "prod_flashinfer": {"attention_backend": "flashinfer"},
    "prod_no_radix": {"disable_radix_cache": True},
    # The overlap scheduler is the engine's only remaining concurrency mechanism:
    # it prepares the next batch while the current forward runs, which means
    # per-step buffers (logits, probs, sampling info) are shared across steps. A
    # stale buffer would corrupt the returned logprob for a stretch of steps. This
    # arm runs the production-matched contract with it turned off.
    "prod_no_overlap": {"disable_overlap_schedule": True},
    # None is ServerArgs' own default: lets the Gemma2 handler pick the backend
    # (trtllm_mha on Blackwell) instead of the project's explicit Triton choice.
    # NOTE: this arm made the GPU raise XID 31 (MMU fault) during piecewise CUDA
    # graph capture on 2026-10-03; treat it as unsafe until re-checked.
    "prod_default_backend": {"attention_backend": None},
    # The failing run's log says disable_cuda_graph=True, but nothing in the project
    # or in SGLang 0.5.11's Gemma2/Triton path forces it. If the reading was wrong,
    # production replayed a captured decode graph -- a mechanism that can produce a
    # structured, position-localised logit error. Keep the alternative explicit.
    "prod_cudagraph_on": {"disable_cuda_graph": False},
    # rollout_actor.init() always passes these two; they change the allocator.
    "prod_memory_saver": {"enable_memory_saver": True, "enable_weights_cpu_backup": True},
    # A larger split-tile makes the deterministic KV-split schedule token-exact;
    # without deterministic inference it only affects the Triton split count.
    "prod_split_tile_256": {"triton_attention_split_tile_size": 256},
    # Same flags as `prod`, run second: its reference-trajectory rescore is an
    # engine-vs-engine determinism check on byte-identical tokens.
    "prod_repeat": {},
    # Production's RolloutArguments default is rollout_mem_fraction_static=0.6, not
    # the 0.25 the September harness used. The pool size changes the KV block table,
    # hence the gather order inside the decode attention kernel.
    "prod_mem06": {"mem_fraction_static": 0.6},
    # The production failure is on the *decode* path (KV cache). Comparing decode
    # logprobs against HF is confounded by sampling, because the engine picks its own
    # tokens. Forcing chunked prefill makes the rescore of one fixed sequence walk
    # the KV cache deterministically, so this isolates the KV-cache attention path
    # against the same HF oracle -- the one engine path not yet measured.
    "prod_chunk512": {"chunked_prefill_size": 512, "max_prefill_tokens": 512},
    "prod_chunk1024": {"chunked_prefill_size": 1024, "max_prefill_tokens": 1024},
    # SGLANG_RETURN_ORIGINAL_LOGPROB makes SGLang return log_softmax(logits) instead
    # of log(softmax(logits)) with a bf16 probability in between. Comparing this arm
    # against the raw HF reference separates "the engine's logits are wrong" from
    # "the engine reports the logprob at low precision".
    "prod_original_logprob": {},
    # The production rollout loop does sleep() (release_memory_occupation) before
    # training and wakeup() (resume_memory_occupation) before the next rollout, so
    # every rollout after the first runs on a re-allocated KV cache. A stale or
    # partly re-initialised pool would corrupt attention silently over a contiguous
    # range of positions -- intermittent, and localised, which is exactly the shape
    # the bug note reports. Nothing tested so far exercises this cycle.
    "prod_sleepwake": {"enable_memory_saver": True, "enable_weights_cpu_backup": True},
    "prod_sleepwake_kvonly": {"enable_memory_saver": True,
                              "enable_weights_cpu_backup": True},
}

# Optional action performed on a freshly created engine, before it is used.
CONFIG_PRELUDE: dict[str, str] = {
    "prod_sleepwake": "sleep_wake_all",
    "prod_sleepwake_kvonly": "sleep_wake_kv",
}

# Environment overrides applied only while one config's engine is alive. The SGLang
# server runs as a child process, so it inherits these.
CONFIG_ENV: dict[str, dict[str, str]] = {
    "prod_original_logprob": {"SGLANG_RETURN_ORIGINAL_LOGPROB": "1"},
}

SERVER_FIELDS = (
    "attention_backend",
    "sampling_backend",
    "enable_deterministic_inference",
    "disable_cuda_graph",
    "disable_piecewise_cuda_graph",
    "triton_attention_reduce_in_fp32",
    "disable_radix_cache",
    "chunked_prefill_size",
    "max_prefill_tokens",
    "context_length",
    "cuda_graph_max_bs",
    "cuda_graph_bs",
    "piecewise_cuda_graph_max_tokens",
    "mem_fraction_static",
    "skip_server_warmup",
    "dtype",
    "tp_size",
)


def _values(entries: list[list[Any]]) -> list[float]:
    return [float(item[0]) for item in entries]


def _ids(entries: list[list[Any]]) -> list[int]:
    return [int(item[1]) for item in entries]


def _token_ids(value: Any) -> list[int]:
    """Accept whatever `apply_chat_template(tokenize=True)` returns.

    Transformers 5.6 returns a BatchEncoding/dict rather than a bare list, and
    an older/newer signature may return a nested single-row list.
    """
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


def _server_args(engine: Any) -> dict[str, Any]:
    args = getattr(engine, "server_args", None)
    out: dict[str, Any] = {}
    for name in SERVER_FIELDS:
        if args is not None and hasattr(args, name):
            value = getattr(args, name)
            out[name] = value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
        else:
            out[name] = "<absent>"
    return out


def _delta(left: list[float], right: list[float]) -> list[float]:
    assert len(left) == len(right), (len(left), len(right))
    return [abs(a - b) for a, b in zip(left, right)]


def _bins(values: list[float], edges: tuple[float, ...]) -> list[dict[str, Any]]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows.append({"lo": lo, "hi": hi, "n": 0, "mean": None, "max": None})
    return rows


def _stats(delta: list[float], positions: list[int] | None = None) -> dict[str, Any]:
    if not delta:
        return {"tokens": 0}
    ordered = sorted(delta)
    n = len(ordered)

    def quantile(q: float) -> float:
        idx = min(n - 1, max(0, int(math.ceil(q * n)) - 1))
        return ordered[idx]

    worst = max(range(n), key=lambda i: delta[i])
    above = [i for i, v in enumerate(delta) if v > 0.5]
    row: dict[str, Any] = {
        "tokens": n,
        "mean": sum(delta) / n,
        "p50": quantile(0.50),
        "p99": quantile(0.99),
        "p999": quantile(0.999),
        "max": delta[worst],
        "worst_index": worst,
        "above_0p1": sum(1 for v in delta if v > 0.1),
        "above_0p5": len(above),
        "above_1p0": sum(1 for v in delta if v > 1.0),
        "above_0p5_fraction": len(above) / n,
    }
    if positions is not None:
        row["worst_position"] = positions[worst]
    if above:
        shown = above[:200]
        row["above_0p5_index_head"] = shown
        row["above_0p5_position_head"] = [positions[i] for i in shown] if positions else None
        if len(above) > 1:
            gaps = [b - a for a, b in zip(shown, shown[1:])]
            hist: dict[str, int] = {}
            for g in gaps:
                hist[str(g)] = hist.get(str(g), 0) + 1
            row["above_0p5_gap_histogram"] = dict(sorted(hist.items(), key=lambda kv: -kv[1])[:12])
            row["above_0p5_gap_min"] = min(gaps)
            row["above_0p5_gap_max"] = max(gaps)
        row["above_0p5_span"] = [above[0], above[-1]]
    return row


def _stratify(delta: list[float], driver: list[float], edges: tuple[float, ...]) -> list[dict[str, Any]]:
    rows = _bins(delta, edges)
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = [i for i, v in enumerate(driver) if lo <= v < hi]
        row = next(r for r in rows if r["lo"] == lo and r["hi"] == hi)
        row["n"] = len(idx)
        if idx:
            values = [delta[i] for i in idx]
            row["mean"] = sum(values) / len(values)
            row["max"] = max(values)
            row["above_0p5"] = sum(1 for v in values if v > 0.5)
    return rows


def _bands(delta: list[float], band: int) -> list[list[Any]]:
    out = []
    for lo in range(0, len(delta), band):
        seg = delta[lo:lo + band]
        out.append([lo, lo + len(seg), len(seg), sum(seg) / len(seg), max(seg)])
    return out


def _hf_scores(model_path: str, records: list[dict[str, Any]], attn: str,
               temperature: float) -> dict[str, Any]:
    """Single full-sequence forward per record; returns response-position scores.

    Chunked over the sequence so the vocab-sized float32 softmax never materialises
    for all 4096 positions at once.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation=attn,
    ).to("cuda").eval()
    per_record: list[dict[str, Any]] = []
    with torch.inference_mode():
        for record in records:
            prompt_ids = record["prompt_ids"]
            output_ids = record["output_ids"]
            all_ids = torch.tensor([prompt_ids + output_ids], device="cuda")
            logits = model(input_ids=all_ids, use_cache=False).logits[0]
            start = len(prompt_ids) - 1
            stop = start + len(output_ids)
            labels = torch.tensor(output_ids, device="cuda")
            raw: list[float] = []
            scaled: list[float] = []
            entropy_T: list[float] = []
            margin_T: list[float] = []
            p_sampled_T: list[float] = []
            argmax: list[int] = []
            for lo in range(start, stop, 512):
                hi = min(lo + 512, stop)
                block = logits[lo:hi].float()
                target = labels[lo - start:hi - start]
                # Raw model distribution (T=1): this is what SGLang returns for
                # input/prefill logprobs.
                logp_raw = torch.log_softmax(block, dim=-1)
                raw.extend(logp_raw.gather(-1, target[:, None]).squeeze(-1).tolist())
                argmax.extend(torch.argmax(block, dim=-1).tolist())
                # Rollout distribution (T=temperature): this is what SGLang returns
                # for output/decode logprobs, and therefore what the guard compares.
                logp_T = torch.log_softmax(block / temperature, dim=-1)
                scaled.extend(logp_T.gather(-1, target[:, None]).squeeze(-1).tolist())
                probs_T = logp_T.exp()
                entropy_T.extend((-(probs_T * logp_T).sum(-1)).tolist())
                top2 = torch.topk(logp_T, 2, dim=-1).values
                margin_T.extend((top2[:, 0] - top2[:, 1]).tolist())
                p_sampled_T.extend(probs_T.gather(-1, target[:, None]).squeeze(-1).tolist())
                del block, logp_raw, logp_T, probs_T, top2
            del logits
            per_record.append({
                "raw": raw,
                "temperature": scaled,
                "entropy_nats": entropy_T,
                "top2_margin_nats": margin_T,
                "p_sampled": p_sampled_T,
                "argmax_ids": argmax,
                "sampled_is_argmax": [int(a == b) for a, b in zip(argmax, output_ids)],
            })
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"attn": attn, "records": per_record}


def _hf_padded_check(model_path: str, records: list[dict[str, Any]], attn: str,
                     temperature: float, max_rows: int = 6) -> dict[str, Any]:
    """Settle the bug note's unresolved fork: does batch padding move the long row?

    `BUG_parity_logprob_4096cap.md` section 2.4 reports a 24x difference between
    the replay tool's `--layout batch` and `--layout unpadded`, then flags it as
    unresolved because a hand-written per-row strip test showed no difference.

    This reproduces the tool's batch layout exactly (pad to the widest row, 2D
    attention mask, position_ids = cumsum(mask) - 1 with pads forced to 1) and
    compares the *longest* row's response logprobs against the same row computed
    alone. If the two agree, padding cannot be the cause and the note's 2x2 table
    is a tool artifact; if they disagree, the trainer's padded forward -- not the
    serving engine -- is what the guard is measuring.
    """
    import torch
    from transformers import AutoModelForCausalLM

    if len(records) < 2:
        return {"status": "skipped", "reason": "need at least two rows to pad"}
    longest = max(range(len(records)), key=lambda i: len(records[i]["prompt_ids"])
                  + len(records[i]["output_ids"]))
    others = [i for i in range(len(records)) if i != longest][: max_rows - 1]
    rows = [longest] + others
    width = max(len(records[i]["prompt_ids"]) + len(records[i]["output_ids"]) for i in rows)
    pad_id = 0
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    loss = torch.zeros((len(rows), width), dtype=torch.bool)
    for r, i in enumerate(rows):
        seq = records[i]["prompt_ids"] + records[i]["output_ids"]
        n = len(seq)
        ids[r, :n] = torch.tensor(seq, dtype=torch.long)
        mask[r, :n] = 1
        start = len(records[i]["prompt_ids"]) - 1
        loss[r, start:start + len(records[i]["output_ids"])] = True
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 1)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation=attn,
    ).to("cuda").eval()
    result: dict[str, Any] = {"status": "completed", "attn": attn,
                              "rows": rows, "longest_row": longest, "width": width}
    with torch.inference_mode():
        logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda(),
                       position_ids=positions.cuda(), use_cache=False).logits
        row = 0  # the longest row is first
        # Match _hf_scores' convention exactly: the response token at absolute
        # position p is predicted by logits[p - 1], and the response occupies
        # [start, start + len(output)). Slicing by the loss mask instead would put
        # this list one position out of step with the unpadded reference and
        # manufacture a large, periodic difference on repetitive text.
        out_len = len(records[longest]["output_ids"])
        prompt_len = len(records[longest]["prompt_ids"])
        labels_kept = ids[row][prompt_len:prompt_len + out_len].cuda()
        selected = logits[row][prompt_len - 1:prompt_len - 1 + out_len].float()
        raw: list[float] = []
        scaled: list[float] = []
        for lo in range(0, out_len, 512):
            hi = min(lo + 512, out_len)
            block = selected[lo:hi]
            logp_raw = torch.log_softmax(block, dim=-1)
            raw.extend(logp_raw.gather(-1, labels_kept[lo:hi, None]).squeeze(-1).tolist())
            logp_T = torch.log_softmax(block / temperature, dim=-1)
            scaled.extend(logp_T.gather(-1, labels_kept[lo:hi, None]).squeeze(-1).tolist())
            del block, logp_raw, logp_T
        del logits, selected
    result["padded_raw"] = raw
    result["padded_temperature"] = scaled
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _engine_kwargs(model_path: str, extra: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    kwargs: dict[str, Any] = {
        "model_path": model_path,
        "trust_remote_code": False,
        "tp_size": 1,
        "log_level": "warning",
    }
    kwargs.update(PRODUCTION_KWARGS)
    kwargs.update(extra)
    try:
        from sglang.srt.server_args import ServerArgs

        valid = set(ServerArgs.__dataclass_fields__)
        unknown = sorted(k for k in kwargs if k not in valid)
    except Exception as exc:  # pragma: no cover - introspection is best effort
        unknown = ["<introspection failed: %s>" % type(exc).__name__]
    return kwargs, unknown


def run(model_path: str, output_path: Path, configs: list[str], max_new_tokens: int,
        prompt_limit: int, force_cap: int = 0, prompt_style: str = "coherent",
        prompt_pad: int = 0) -> dict[str, Any]:
    import sglang
    import torch
    import transformers
    from transformers import AutoTokenizer

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    prompts = prompt_set(prompt_limit, prompt_style, prompt_pad, tokenizer)
    prompt_ids = [
        _token_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True))
        for prompt in prompts
    ]
    environment = {
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "sglang": getattr(sglang, "__version__", "unknown"),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": max_new_tokens,
        "prompt_style": prompt_style,
        "prompt_tokens": [len(ids) for ids in prompt_ids],
    }
    print("LONGCAP_ENV_JSON=" + json.dumps(environment, sort_keys=True), flush=True)

    results: dict[str, Any] = {}
    hf_cache: dict[str, Any] = {}
    reference_trajectory: list[dict[str, Any]] | None = None
    reference_prefill_raw: list[list[float]] | None = None

    def write_partial() -> None:
        """Persist after every config so a crash cannot lose earlier evidence."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({
            "status": "partial",
            "environment": environment,
            "configs": results,
            "reference_trajectory_owner": next(
                (k for k, v in results.items() if v.get("owns_reference_trajectory")), None),
            "elapsed_seconds": time.monotonic() - started,
        }, indent=2, sort_keys=True) + "\n")

    for name in configs:
        if name not in CONFIGS:
            raise ValueError("unknown config " + name)
        extra = CONFIGS[name]
        kwargs, unknown = _engine_kwargs(model_path, extra)
        print("LONGCAP_CONFIG_START=" + json.dumps({"config": name, "extra": extra,
                                                    "unknown_server_args": unknown},
                                                   sort_keys=True), flush=True)
        entry: dict[str, Any] = {"extra": extra, "unknown_server_args": unknown,
                                 "requested_kwargs": {k: v for k, v in kwargs.items()
                                                      if k not in ("model_path",)}}
        if unknown:
            # Fail closed and cheap: an unknown ServerArgs field means this config
            # would silently degenerate into the baseline, which is worse than no
            # data. Do not construct an engine for it.
            entry["status"] = "failed"
            entry["error_type"] = "UnknownServerArgs"
            entry["error"] = ("requested flags are not SGLang 0.5.11 ServerArgs fields: "
                              + ",".join(unknown))
            results[name] = entry
            write_partial()
            print("LONGCAP_CONFIG_DONE=" + json.dumps(
                {"config": name, "status": "failed", "error": entry["error"]},
                sort_keys=True), flush=True)
            continue
        engine = None
        env_overrides = CONFIG_ENV.get(name, {})
        env_previous = {k: os.environ.get(k) for k in env_overrides}
        try:
            for key, value in env_overrides.items():
                os.environ[key] = value
            entry["env_overrides"] = env_overrides
            engine = sglang.Engine(**kwargs)
            entry["server_args"] = _server_args(engine)
            prelude = CONFIG_PRELUDE.get(name)
            if prelude:
                # Mirror the production loop: release the memory occupation, then
                # resume it, and only then run the rollout that gets compared.
                tags = None if prelude == "sleep_wake_all" else ["kv_cache"]
                engine.release_memory_occupation(tags=tags)
                engine.resume_memory_occupation(tags=tags)
                entry["prelude"] = {"action": prelude, "tags": tags}
            # Production ran a 64-request rollout batch in which most requests
            # stopped at EOS early and one ran to the 4096 cap, so the decode batch
            # shrank from 64 to 1 and the capped sequence finished alone. A uniform
            # ignore_eos batch keeps bs constant and does not exercise that. The
            # last `force_cap` prompts are therefore pinned to the cap; the rest
            # stop normally.
            params = []
            for index in range(len(prompt_ids)):
                pinned = force_cap > 0 and index >= len(prompt_ids) - force_cap
                params.append({
                    "temperature": TEMPERATURE, "top_p": TOP_P,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": bool(pinned) or force_cap <= 0,
                })
            entry["sampling_plan"] = {
                "force_cap": force_cap,
                "pinned_requests": [i for i, p in enumerate(params) if p["ignore_eos"]],
                "eos_requests": [i for i, p in enumerate(params) if not p["ignore_eos"]],
            }
            generated = engine.generate(
                input_ids=prompt_ids,
                sampling_params=params,
                return_logprob=True,
                logprob_start_len=-1,
            )
            generated = generated if isinstance(generated, list) else [generated]
            records = []
            for index, output in enumerate(generated):
                response_ids = [int(v) for v in output["output_ids"]]
                entries = output["meta_info"]["output_token_logprobs"]
                assert _ids(entries) == response_ids, "decode logprob IDs do not match sampled IDs"
                records.append({
                    "sample": index,
                    "prompt_ids": prompt_ids[index],
                    "output_ids": response_ids,
                    "decode_logprobs": _values(entries),
                    "finish_reason": output["meta_info"].get("finish_reason"),
                    "completion_tokens": output["meta_info"].get("completion_tokens"),
                })
            entry["samples"] = [
                {"sample": r["sample"], "prompt_len": len(r["prompt_ids"]),
                 "output_len": len(r["output_ids"]), "finish_reason": r["finish_reason"],
                 "completion_tokens": r["completion_tokens"]}
                for r in records
            ]
            # Keep the raw arrays: the alignment test (is the engine's logprob for a
            # neighbouring position?) can only be run offline if these survive.
            entry["prompt_ids"] = [r["prompt_ids"] for r in records]
            entry["output_ids"] = [r["output_ids"] for r in records]
            entry["decode_logprobs"] = [r["decode_logprobs"] for r in records]

            # Engine prefill rescoring of the exact same full sequences.
            full_ids = [r["prompt_ids"] + r["output_ids"] for r in records]
            prefill: dict[str, list[list[float]]] = {}
            for scale_name, temperature in (("raw", 1.0), ("temperature", TEMPERATURE)):
                rescored = engine.generate(
                    input_ids=full_ids,
                    sampling_params={"temperature": temperature, "top_p": TOP_P,
                                     "max_new_tokens": 0},
                    return_logprob=True,
                    logprob_start_len=0,
                )
                rescored = rescored if isinstance(rescored, list) else [rescored]
                rows = []
                for record, output in zip(records, rescored):
                    entries = output["meta_info"]["input_token_logprobs"]
                    ids = _ids(entries)
                    assert ids == record["prompt_ids"] + record["output_ids"], \
                        "prefill rescore IDs do not match the trajectory"
                    rows.append(_values(entries[len(record["prompt_ids"]):]))
                prefill[scale_name] = rows
            entry["prefill_raw"] = prefill["raw"]
            entry["prefill_temperature"] = prefill["temperature"]

            if reference_trajectory is None:
                reference_trajectory = [
                    {"prompt_ids": r["prompt_ids"], "output_ids": r["output_ids"]}
                    for r in records
                ]
                reference_prefill_raw = prefill["raw"]
                entry["owns_reference_trajectory"] = True
            else:
                # Token-exact cross-config test: rescore the *same* token sequence
                # under this config's kernels and compare with the owner's prefill
                # logprobs. This is the only comparison in the probe where the
                # tokens are identical across configs, so it isolates the flag.
                ref_full = [r["prompt_ids"] + r["output_ids"] for r in reference_trajectory]
                ref_out = engine.generate(
                    input_ids=ref_full,
                    sampling_params={"temperature": 1.0, "top_p": TOP_P, "max_new_tokens": 0},
                    return_logprob=True,
                    logprob_start_len=0,
                )
                ref_out = ref_out if isinstance(ref_out, list) else [ref_out]
                ref_rows = []
                for ref_record, output in zip(reference_trajectory, ref_out):
                    entries = output["meta_info"]["input_token_logprobs"]
                    ids = _ids(entries)
                    assert ids == ref_record["prompt_ids"] + ref_record["output_ids"], \
                        "reference prefill rescore IDs do not match the reference trajectory"
                    ref_rows.append(_values(entries[len(ref_record["prompt_ids"]):]))
                entry["reference_prefill_raw"] = ref_rows
                entry["reference_prefill_raw_vs_owner"] = _stats(_delta(
                    [v for row in ref_rows for v in row],
                    [v for row in reference_prefill_raw for v in row]))

            if name not in hf_cache:
                hf_cache[name] = {}
            # sdpa is a second HF kernel only used as a cross-check; paying for a
            # second 5 GB model load on every config would be waste.
            attn_list = ("eager", "sdpa") if name == configs[0] else ("eager",)
            for attn in attn_list:
                hf_cache[name][attn] = _hf_scores(model_path, records, attn, TEMPERATURE)

            if name == configs[0] and len(records) > 1:
                padded = _hf_padded_check(model_path, records, "eager", TEMPERATURE)
                entry["hf_padded_check"] = {
                    k: v for k, v in padded.items()
                    if k not in ("padded_raw", "padded_temperature")
                }
                if padded.get("status") == "completed":
                    long_row = padded["longest_row"]
                    base = hf_cache[name]["eager"]["records"][long_row]
                    entry["hf_padded_vs_unpadded_raw"] = _stats(
                        _delta(padded["padded_raw"], base["raw"]))
                    entry["hf_padded_vs_unpadded_temperature"] = _stats(
                        _delta(padded["padded_temperature"], base["temperature"]))

            comparisons: dict[str, Any] = {}
            # Engine-internal KV-vs-prefill check. Only meaningful when the config
            # sets SGLANG_RETURN_ORIGINAL_LOGPROB=1, because that makes the *decode*
            # logprob raw, matching the prefill logprob semantics. With the env
            # unset the decode value is temperature-scaled and this row is expected
            # to be large (~0.1); read it together with entry["env_overrides"].
            comparisons["decode_vs_own_prefill_raw"] = _stats(_delta(
                [v for r in records for v in r["decode_logprobs"]],
                [v for row in prefill["raw"] for v in row]))
            for attn in attn_list:
                hf = hf_cache[name][attn]["records"]
                decode = [v for r in records for v in r["decode_logprobs"]]
                hf_T = [v for r in hf for v in r["temperature"]]
                hf_raw = [v for r in hf for v in r["raw"]]
                prefill_raw = [v for r in prefill["raw"] for v in r]
                prefill_T = [v for r in prefill["temperature"] for v in r]
                comparisons[f"decode_vs_hf_{attn}_temperature"] = _stats(_delta(decode, hf_T))
                comparisons[f"decode_vs_hf_{attn}_raw"] = _stats(_delta(decode, hf_raw))
                comparisons[f"prefill_raw_vs_hf_{attn}_raw"] = _stats(_delta(prefill_raw, hf_raw))
                comparisons[f"prefill_temperature_vs_hf_{attn}_temperature"] = _stats(
                    _delta(prefill_T, hf_T))
                comparisons[f"prefill_raw_vs_prefill_temperature"] = _stats(
                    _delta(prefill_raw, prefill_T))
            hf_e = hf_cache[name]["eager"]["records"]
            if "sdpa" in hf_cache[name]:
                hf_s = hf_cache[name]["sdpa"]["records"]
                comparisons["hf_eager_vs_sdpa_raw"] = _stats(_delta(
                    [v for r in hf_e for v in r["raw"]], [v for r in hf_s for v in r["raw"]]))
            if entry.get("hf_padded_vs_unpadded_raw"):
                comparisons["hf_padded_vs_unpadded_raw"] = entry["hf_padded_vs_unpadded_raw"]
                comparisons["hf_padded_vs_unpadded_temperature"] = \
                    entry["hf_padded_vs_unpadded_temperature"]

            # The production guard metric, in detail.
            decode_rows = [r["decode_logprobs"] for r in records]
            hf_rows = [r["temperature"] for r in hf_e]
            delta_rows = [_delta(d, h) for d, h in zip(decode_rows, hf_rows)]
            flat_delta = [v for row in delta_rows for v in row]
            flat_positions = [i for row in delta_rows for i in range(len(row))]
            driver_entropy = [v for r in hf_e for v in r["entropy_nats"]]
            driver_margin = [v for r in hf_e for v in r["top2_margin_nats"]]
            driver_p = [v for r in hf_e for v in r["p_sampled"]]
            driver_argmax = [v for r in hf_e for v in r["sampled_is_argmax"]]
            detail = {
                "guard_metric_decode_vs_hf_eager_temperature": _stats(flat_delta, flat_positions),
                "per_band": _bands(flat_delta, BAND),
                "per_sample_band": [_bands(row, BAND) for row in delta_rows],
                "error_by_hf_entropy": _stratify(flat_delta, driver_entropy, ENTROPY_BINS),
                "error_by_hf_top2_margin": _stratify(flat_delta, driver_margin, MARGIN_BINS),
                "sampled_is_argmax_fraction": sum(driver_argmax) / len(driver_argmax),
                "mean_hf_entropy": sum(driver_entropy) / len(driver_entropy),
                "mean_p_sampled": sum(driver_p) / len(driver_p),
            }
            worst = sorted(range(len(flat_delta)), key=lambda i: -flat_delta[i])[:25]
            detail["worst_positions"] = [
                {"flat_index": i,
                 "delta": flat_delta[i],
                 "hf_entropy_nats": driver_entropy[i],
                 "hf_top2_margin_nats": driver_margin[i],
                 "hf_p_sampled": driver_p[i],
                 "sampled_is_argmax": driver_argmax[i]}
                for i in worst
            ]
            entry["comparisons"] = comparisons
            entry["detail"] = detail
            entry["hf_eager"] = hf_e
            entry["hf_sdpa"] = hf_cache[name].get("sdpa", {}).get("records")
            entry["status"] = "completed"
        except BaseException as exc:  # keep the other configs running
            import traceback

            entry["status"] = "failed"
            entry["error_type"] = type(exc).__name__
            entry["error"] = str(exc)[:2000]
            entry["traceback_tail"] = traceback.format_exc().splitlines()[-25:]
        finally:
            if engine is not None:
                try:
                    engine.shutdown()
                except Exception:
                    pass
            gc.collect()
            torch.cuda.empty_cache()
            for key, previous in env_previous.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous
        results[name] = entry
        write_partial()
        print("LONGCAP_CONFIG_DONE=" + json.dumps({
            "config": name, "status": entry["status"],
            "error": entry.get("error"),
            "server_args": entry.get("server_args"),
            "guard_metric": entry.get("detail", {}).get(
                "guard_metric_decode_vs_hf_eager_temperature"),
        }, sort_keys=True), flush=True)

    payload = {
        "status": "completed",
        "environment": environment,
        "configs": results,
        "elapsed_seconds": time.monotonic() - started,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    compact = {
        "status": "completed",
        "environment": environment,
        "elapsed_seconds": payload["elapsed_seconds"],
        "configs": {
            name: {
                "status": entry["status"],
                "error": entry.get("error"),
                "server_args": entry.get("server_args"),
                "guard_metric": entry.get("detail", {}).get(
                    "guard_metric_decode_vs_hf_eager_temperature"),
                "prefill_raw_vs_hf_eager_raw": entry.get("comparisons", {}).get(
                    "prefill_raw_vs_hf_eager_raw"),
                "reference_prefill_raw_vs_owner": entry.get("reference_prefill_raw_vs_owner"),
                "per_band": entry.get("detail", {}).get("per_band"),
                "error_by_hf_entropy": entry.get("detail", {}).get("error_by_hf_entropy"),
                "mean_hf_entropy": entry.get("detail", {}).get("mean_hf_entropy"),
            }
            for name, entry in results.items()
        },
    }
    print("LONGCAP_SUMMARY_JSON=" + json.dumps(compact, sort_keys=True), flush=True)
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--prompt-limit", type=int, default=0)
    parser.add_argument("--force-cap", type=int, default=0,
                        help="pin only the last N requests to max_new_tokens; 0 pins all")
    parser.add_argument("--prompt-style", choices=("coherent", "flat", "repeat"),
                        default="coherent")
    parser.add_argument("--prompt-pad", type=int, default=0,
                        help="pad each prompt with filler up to this many tokens")
    args = parser.parse_args()
    configs = [v.strip() for v in args.configs.split(",") if v.strip()]
    if not configs:
        raise ValueError("at least one config is required")
    run(args.model_path, args.output, configs, args.max_new_tokens, args.prompt_limit,
        args.force_cap, args.prompt_style, args.prompt_pad)


if __name__ == "__main__":
    main()
