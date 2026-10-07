"""
Monkey patch for SGLang scheduler's process_batch_result_prefill method.
This allows using numpy() instead of tolist() for hidden_states, which is much faster.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
from pathlib import Path

import torch

from kdflow.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import (
        EmbeddingBatchResult,
        GenerationBatchResult,
        ScheduleBatch,
        Scheduler,
    )

logger = init_logger(__name__)

# Flag to prevent multiple patch applications
_PATCH_APPLIED = False

# Upstream stream_output_generation, kept so the audit wrapper never forks it.
_ORIG_STREAM_OUTPUT_GENERATION = None

# Cap on audit lines per process: enough to see a pattern, not enough to flood.
_AUDIT_MAX_LINES = 20
_AUDIT_LINES = 0

# Separate cap for the unconditional probe below. The disagreement audit can only
# speak about requests where output_token_logprobs_val is non-empty, so "no audit
# lines" is ambiguous between "clean" and "nothing was ever inspected". This
# counter prints structure for the first N requests regardless, which is what
# separates those two readings.
_ALWAYS_MAX_LINES = 12
_ALWAYS_LINES = 0


def process_batch_result_prefill_patched(
    self: "Scheduler",
    batch: "ScheduleBatch",
    result: Union["GenerationBatchResult", "EmbeddingBatchResult"],
):
    """
    Patched version of process_batch_result_prefill.
    Key change: Use .numpy() instead of .tolist() for hidden_states (much faster).
    """
    from sglang.srt.environ import envs
    from sglang.srt.managers.io_struct import AbortReq
    # from sglang.srt.managers.schedule_batch import RequestStage
    from sglang.srt.mem_cache.common import release_kv_cache
    # from sglang.srt.tracing.trace import trace_slice

    skip_stream_req = None

    if self.is_generation:
        if result.copy_done is not None:
            result.copy_done.synchronize()

        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
        )

        # Move next_token_ids and logprobs to cpu
        next_token_ids = next_token_ids.tolist()
        if batch.return_logprob:
            if logits_output.next_token_logprobs is not None:
                logits_output.next_token_logprobs = (
                    logits_output.next_token_logprobs.tolist()
                )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = tuple(
                    logits_output.input_token_logprobs.tolist()
                )

        hidden_state_offset = 0

        # Check finish conditions
        logprob_pt = 0

        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            if req.finished() or req.is_retracted:
                # decode req in mixed batch or retracted req
                continue

            if req.is_chunked <= 0:
                if hasattr(req.time_stats, "prefill_finished_ts"):
                    if req.time_stats.prefill_finished_ts == 0.0:
                        req.time_stats.prefill_finished_ts = time.time()
                else:
                    req.time_stats.set_prefill_finished_time()

                # req output_ids are set here
                req.output_ids.append(next_token_id)
                req.check_finished()

                if req.finished():
                    self.maybe_collect_routed_experts(req)
                    release_kv_cache(req, self.tree_cache)
                    req.time_stats.completion_time = time.perf_counter()
                elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                    # This updates radix so others can match
                    self.tree_cache.cache_unfinished_req(req)

                self.maybe_collect_customized_info(i, req, logits_output)

                if batch.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]

                    num_input_logprobs = self._calculate_num_input_logprobs(
                        req, extend_input_len, extend_logprob_start_len
                    )

                    if req.return_logprob:
                        self.add_logprob_return_values(
                            i,
                            req,
                            logprob_pt,
                            next_token_ids,
                            num_input_logprobs,
                            logits_output,
                        )
                    # logprob_pt is a shared offset: this prefill advance decides where the
                    # (unpatched) decode path writes output_token_logprobs. If the advance
                    # disagrees with the number of slots SGLang actually fills, decode writes
                    # at wrong offsets and the leftover slots keep SGLang's 0.0 default, which
                    # the parity path then reads as a log-probability of 0 (p=1). Gated by env
                    # so production runs stay quiet.
                    if os.environ.get("MP_LOGPROB_ACCOUNTING", "0") == "1":
                        print(
                            "[logprob-accounting] rid=%s pt_before=%d start_len=%d input_len=%d "
                            "num_input=%d pt_after=%d chunked=%d finished=%d"
                            % (
                                getattr(req, "rid", "?"),
                                logprob_pt - num_input_logprobs,
                                extend_logprob_start_len,
                                extend_input_len,
                                num_input_logprobs,
                                logprob_pt,
                                getattr(req, "is_chunked", -1),
                                req.finished(),
                            ),
                            flush=True,
                        )
                    logprob_pt += num_input_logprobs

                # === KEY CHANGE: Use .numpy() instead of .tolist() ===
                if (
                    req.return_hidden_states
                    and logits_output.hidden_states is not None
                ):
                    req.hidden_states.append(
                        logits_output.hidden_states[
                            hidden_state_offset : (
                                hidden_state_offset := hidden_state_offset
                                + len(req.origin_input_ids)
                            )
                        ]
                        .half()
                        .cpu()
                        .numpy()
                    )

                if req.grammar is not None:
                    # FIXME: this try-except block is for handling unexpected xgrammar issue.
                    try:
                        req.grammar.accept_token(next_token_id)
                    except ValueError as e:
                        # Grammar accept_token can raise ValueError if the token is not in the grammar.
                        # This can happen if the grammar is not set correctly or the token is invalid.
                        logger.error(
                            f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                        )
                        self.abort_request(AbortReq(rid=req.rid))
                    req.grammar.finished = req.finished()

                # trace_slice(
                #     RequestStage.PREFILL_FORWARD,
                #     req.rid,
                #     auto_next_anon=not req.finished(),
                #     thread_finish_flag=req.finished(),
                # )

            else:
                # being chunked reqs' prefill is not finished
                req.is_chunked -= 1
                # There is only at most one request being currently chunked.
                # Because this request does not finish prefill,
                # we don't want to stream the request currently being chunked.
                skip_stream_req = req

                # Incrementally update input logprobs.
                if batch.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        # Update input logprobs.
                        num_input_logprobs = self._calculate_num_input_logprobs(
                            req, extend_input_len, extend_logprob_start_len
                        )
                        if req.return_logprob:
                            self.add_input_logprob_return_values(
                                i,
                                req,
                                logits_output,
                                logprob_pt,
                                num_input_logprobs,
                                last_prefill_chunk=False,
                            )
                        logprob_pt += num_input_logprobs

                # trace_slice(
                #     RequestStage.PREFILL_CHUNKED_FORWARD,
                #     req.rid,
                #     auto_next_anon=True,
                # )

    else:  # embedding or reward model
        if result.copy_done is not None:
            result.copy_done.synchronize()

        is_sparse = envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set()

        embeddings = result.embeddings

        if is_sparse:
            batch_ids, token_ids = embeddings.indices()
            values = embeddings.values()

            embeddings = [{} for _ in range(embeddings.size(0))]
            for i in range(batch_ids.shape[0]):
                embeddings[batch_ids[i].item()][token_ids[i].item()] = values[
                    i
                ].item()
        else:
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.tolist()
            else:
                embeddings = [tensor.tolist() for tensor in embeddings]

        # Check finish conditions
        for i, req in enumerate(batch.reqs):
            if req.is_retracted:
                continue

            req.embedding = embeddings[i]
            if req.is_chunked <= 0:
                # Dummy output token for embedding models
                req.output_ids.append(0)
                req.check_finished()

                if req.finished():
                    release_kv_cache(req, self.tree_cache)
                else:
                    self.tree_cache.cache_unfinished_req(req)
            else:
                # being chunked reqs' prefill is not finished
                req.is_chunked -= 1

            # trace_slice(
            #     RequestStage.PREFILL_FORWARD,
            #     req.rid,
            #     auto_next_anon=not req.finished(),
            #     thread_finish_flag=req.finished(),
            # )

    self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)

    if self.current_scheduler_metrics_enabled:
        can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
        self.log_prefill_stats(
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )


def _audit_emit(line: str) -> None:
    """Emit one audit line to stdout *and* to a file on shared storage.

    stdout from an engine scheduler subprocess has repeatedly failed to reach any
    log we can grep (attempt log, ray session log), so the file is the source of
    truth; the print stays for whoever is tailing the console.  Capped in size so
    a pathological run cannot fill the mount.
    """
    print(line, flush=True)
    # Deliberately not an MP_* variable: run_command strips those, so an MP_ name
    # would never reach the engine process. Defaults to a shared-storage sink.
    path = os.environ.get("SIMCT_LOGPROB_AUDIT_FILE") or os.environ.get(
        "MP_LOGPROB_AUDIT_FILE"
    ) or "/workspace/storage-shared/nlp/tungks/_simct_logprob_audit.log"
    try:
        p = Path(path)
        if p.exists() and p.stat().st_size > 5 * 1024 * 1024:
            return
        with p.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception as exc:  # never let diagnostics kill the engine
        print("[logprob-audit] write failed: %r" % (exc,), flush=True)


def _audit_stream_output_logprobs(
    reqs,
    return_logprob: bool,
    skip_req=None,
) -> None:
    """Report, per request, whether token count and collected log-probs disagree.

    ``stream_output_generation`` cuts the outgoing slice with an independent
    counter (``logprob_end = max(len(output_ids_), 1)``) from the one that fills
    ``req.output_token_logprobs_val``.  If those two ever disagree, the client
    reads a log-probability that belongs to a different position.

    Always on, but it prints only when it actually finds a disagreement or a
    literal ``0.0``, and at most ``_AUDIT_MAX_LINES`` lines per process.  It is
    deliberately not gated by an env var: the engine runs as a separate Ray
    actor, so an ``MP_*`` flag never reaches this process.
    """
    global _AUDIT_LINES, _ALWAYS_LINES
    for req in reqs:
        if req is skip_req or not getattr(req, "return_logprob", False):
            continue
        vals = getattr(req, "output_token_logprobs_val", None) or []
        if _ALWAYS_LINES < _ALWAYS_MAX_LINES:
            _ALWAYS_LINES += 1
            zeros = [i for i, v in enumerate(vals) if v == 0.0]
            span = (
                "none"
                if not zeros
                else "%d..%d(n=%d,run=%s)"
                % (zeros[0], zeros[-1], len(zeros), zeros[-1] - zeros[0] + 1 == len(zeros))
            )
            _audit_emit(
                "[logprob-always] pid=%d rid=%s vals=%d n_zero=%d span=%s head=%s tail=%s"
                % (
                    os.getpid(),
                    getattr(req, "rid", "?"),
                    len(vals),
                    len(zeros),
                    span,
                    ["%.4g" % v for v in vals[:4]],
                    ["%.4g" % v for v in vals[-4:]],
                )
            )
        if not vals:
            continue
        ids = len(req.output_ids_through_stop)
        first_zero = next((i for i, v in enumerate(vals) if v == 0.0), -1)
        if (ids != len(vals) or first_zero >= 0) and _AUDIT_LINES < _AUDIT_MAX_LINES:
            _AUDIT_LINES += 1
            _audit_emit(
                "[logprob-audit] pid=%d rid=%s ids=%d vals=%d mismatch=%d first_zero=%d"
                % (
                    os.getpid(),
                    getattr(req, "rid", "?"),
                    ids,
                    len(vals),
                    ids - len(vals),
                    first_zero,
                )
            )


def stream_output_generation_patched(
    self: "Scheduler",
    reqs: List["Req"],
    return_logprob: bool,
    skip_req: Optional["Req"] = None,
    is_idle_batch: bool = False,
):
    """Wrapper that audits log-prob alignment before delegating to SGLang."""
    if return_logprob:
        _audit_stream_output_logprobs(reqs, return_logprob, skip_req)
    return _ORIG_STREAM_OUTPUT_GENERATION(
        self, reqs, return_logprob, skip_req, is_idle_batch
    )


def apply_patch():
    """
    Apply the monkey patch to SGLang's SchedulerOutputProcessorMixin.
    
    This function is idempotent - calling it multiple times is safe.
    Returns True if patch was applied (or already applied), False otherwise.
    """
    global _PATCH_APPLIED, _ORIG_STREAM_OUTPUT_GENERATION
    
    if _PATCH_APPLIED:
        return True
    
    try:
        from sglang.srt.managers.scheduler_output_processor_mixin import (
            SchedulerOutputProcessorMixin,
        )
        
        # Check if already patched (by another mechanism like sitecustomize)
        current_method = getattr(SchedulerOutputProcessorMixin, 'process_batch_result_prefill', None)
        if current_method is not None and getattr(current_method, '_kdflow_patched', False):
            _PATCH_APPLIED = True
            print(f"[monkey_patch] Patch already applied, PID={os.getpid()}", flush=True)
            return True
        
        # Mark the patched function
        process_batch_result_prefill_patched._kdflow_patched = True
        
        # Apply patch
        SchedulerOutputProcessorMixin.process_batch_result_prefill = process_batch_result_prefill_patched

        # Audit wrapper: keep the upstream method and probe it instead of forking it.
        _ORIG_STREAM_OUTPUT_GENERATION = (
            SchedulerOutputProcessorMixin.stream_output_generation
        )
        stream_output_generation_patched._kdflow_patched = True
        SchedulerOutputProcessorMixin.stream_output_generation = (
            stream_output_generation_patched
        )
        print(
            f"[monkey_patch] SUCCESS: prefill + stream_output_generation patched! PID={os.getpid()}",
            flush=True,
        )
        
        _PATCH_APPLIED = True
        return True
        
    except ImportError as e:
        print(f"[monkey_patch] Cannot import SGLang module: {e}", flush=True)
        return False
    except Exception as e:
        print(f"[monkey_patch] Error applying patch: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


def is_patch_applied():
    """Check if the patch has been applied in this process."""
    return _PATCH_APPLIED
