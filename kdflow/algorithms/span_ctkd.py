import torch
import torch.nn.functional as F

from kdflow.loss import build_loss_fn
from kdflow.algorithms import register_algorithm
from kdflow.loss.cross_entropy import compute_cross_entropy
from kdflow.utils.logging_utils import init_logger

logger = init_logger(__name__)


class _MemoryEfficientReverseKL(torch.autograd.Function):
    """Exact row-wise reverse KL without retaining full FP32 intermediates.

    SimCT's shared vocabulary can be almost as large as the original
    vocabulary.  The regular RKL implementation retains log-probability and
    probability tensors for backward, which makes a long response exceed a
    24-GiB device even at micro-batch size one.  This implementation stores
    only the input logits and recomputes small row chunks in backward.
    """

    @staticmethod
    def forward(ctx, student_logits, teacher_logits, temperature, row_chunk_size):
        temperature = float(temperature)
        row_chunk_size = max(int(row_chunk_size), 1)
        values = []
        for start in range(0, student_logits.shape[0], row_chunk_size):
            end = min(start + row_chunk_size, student_logits.shape[0])
            student_log_probs = torch.log_softmax(
                student_logits[start:end].float() / temperature, dim=-1
            )
            teacher_log_probs = torch.log_softmax(
                teacher_logits[start:end].float() / temperature, dim=-1
            )
            student_probs = student_log_probs.exp()
            values.append(
                (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
            )
        ctx.save_for_backward(student_logits, teacher_logits)
        ctx.temperature = temperature
        ctx.row_chunk_size = row_chunk_size
        return torch.cat(values, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        student_logits, teacher_logits = ctx.saved_tensors
        temperature = ctx.temperature
        row_chunk_size = ctx.row_chunk_size
        student_grad = torch.empty_like(student_logits)

        for start in range(0, student_logits.shape[0], row_chunk_size):
            end = min(start + row_chunk_size, student_logits.shape[0])
            student_log_probs = torch.log_softmax(
                student_logits[start:end].float() / temperature, dim=-1
            )
            teacher_log_probs = torch.log_softmax(
                teacher_logits[start:end].float() / temperature, dim=-1
            )
            student_probs = student_log_probs.exp()
            log_ratio = student_log_probs - teacher_log_probs
            row_kl = (student_probs * log_ratio).sum(dim=-1, keepdim=True)
            chunk_grad = student_probs * (log_ratio - row_kl)
            chunk_grad.mul_(grad_output[start:end].float().unsqueeze(-1) / temperature)
            student_grad[start:end] = chunk_grad.to(student_logits.dtype)

        return student_grad, None, None, None


def memory_efficient_reverse_kl(
    student_logits, teacher_logits, temperature=1.0, row_chunk_size=128
):
    """Return the unreduced exact RKL for 2-D virtual-vocabulary logits."""
    if student_logits.ndim != 2 or teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "memory_efficient_reverse_kl expects equal 2-D student/teacher logits"
        )
    return _MemoryEfficientReverseKL.apply(
        student_logits, teacher_logits.detach(), temperature, row_chunk_size
    )


@register_algorithm("span_ctkd")
class SpanCrossTokenizerKD:
    """Span-based Cross-Tokenizer Knowledge Distillation.

    Improvement over SimpleCrossTokenizerKD: instead of discarding positions where
    student and teacher tokens do not align 1:1, we group misaligned tokens into
    "spans" and compute their logits as the arithmetic mean of the constituent
    token logits (which corresponds to geometric mean after softmax).

    These spans are appended to the shared (overlap) vocabulary to form a
    "virtual common vocabulary". RKL alignment is then performed on this
    extended vocabulary for every aligned position (both 1:1 tokens and spans).

    For span segments, the overlap logits are taken from the **first** token
    position only (not averaged across positions), because the span is treated
    as a single virtual token occupying one position in the sequence.
    """

    def __init__(
        self,
        strategy,
        student_model,
        teacher_lm_head,
        student_tokenizer,
        teacher_tokenizer,
        **kwargs,
    ):
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher_lm_head = teacher_lm_head
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.student_overlap_token_ids, self.teacher_overlap_token_ids = self._find_overlap_tokens()
        self.loss_fn = build_loss_fn(self.args.kd.kd_loss_fn, self.args)
        self._debug_count = 0  # Counter for debug logging
        self._span_debug_count = 0  # Counter for span logits debug logging

    # ------------------------------------------------------------------
    # Overlap vocabulary (copied from simple_ctkd)
    # ------------------------------------------------------------------
    def _find_overlap_tokens(self):
        """Find the overlap tokens between student and teacher tokenizer."""
        student_vocab = {k.replace("Ġ", "▁"): v for k, v in self.student_tokenizer.get_vocab().items()}
        teacher_vocab = {k.replace("Ġ", "▁"): v for k, v in self.teacher_tokenizer.get_vocab().items()}
        overlap_tokens = set(student_vocab.keys()) & set(teacher_vocab.keys())
        student_ids = [student_vocab[token] for token in overlap_tokens]
        teacher_ids = [teacher_vocab[token] for token in overlap_tokens]
        stu_eos, tea_eos = self.student_tokenizer.eos_token_id, self.teacher_tokenizer.eos_token_id
        if stu_eos not in student_ids or tea_eos not in teacher_ids:
            student_ids.append(stu_eos)
            teacher_ids.append(tea_eos)
        device = self.teacher_lm_head.weight.device
        logger.info(f"[SpanCTKD] Num of overlap_tokens between student & teacher: {len(student_ids)}")
        return (
            torch.tensor(student_ids, dtype=torch.long, device=device),
            torch.tensor(teacher_ids, dtype=torch.long, device=device),
        )

    # ------------------------------------------------------------------
    # Sequence alignment with span identification
    # (adapted from ALM._compute_chunk_alignment)
    # ------------------------------------------------------------------
    def _align_sequences_with_spans(self, tea_label_ids, stu_label_ids):
        """Align teacher and student label tokens and identify spans.

        Operates on **label ids** (i.e. input_ids shifted left by one position,
        already filtered by loss_mask) so that alignment is consistent with the
        next-token-prediction semantics of the logits.

        Uses ``tokenizer.decode([tid])`` per token (O(n)) to correctly handle
        all tokenizer conventions including newlines.

        Args:
            tea_label_ids: 1-D list of teacher label token ids (loss_mask positions).
            stu_label_ids: 1-D list of student label token ids (loss_mask positions).

        Returns:
            segments: list of (tea_start, tea_end, stu_start, stu_end) where
                indices are positions within the label id sequences.  Each tuple
                represents an aligned segment: teacher tokens ``[tea_start, tea_end)``
                and student tokens ``[stu_start, stu_end)`` encode the same text.
                When both ranges have length 1 it is a normal 1:1 alignment;
                otherwise it is a span.
            tea_label_ids_list: the teacher label ids as a Python list.
            stu_label_ids_list: the student label ids as a Python list.
        """
        if len(tea_label_ids) == 0 or len(stu_label_ids) == 0:
            return [], [], []

        tea_ids_list = tea_label_ids if isinstance(tea_label_ids, list) else tea_label_ids.cpu().tolist()
        stu_ids_list = stu_label_ids if isinstance(stu_label_ids, list) else stu_label_ids.cpu().tolist()

        # Convert token ids to text strings for alignment.
        # Use tokenizer.decode([tid]) per token to correctly handle all
        # tokenizer conventions (e.g. Qwen uses Ċ for \n internally, which
        # would NOT be normalized by simple strip of ▁/Ġ prefixes).
        tea_token_texts = [self.teacher_tokenizer.decode([tid]) for tid in tea_ids_list]
        stu_token_texts = [self.student_tokenizer.decode([tid]) for tid in stu_ids_list]

        # Greedy cumulative-text alignment (same algorithm as simple_ctkd)
        tea_eos = self.teacher_tokenizer.eos_token
        stu_eos = self.student_tokenizer.eos_token

        i, j = 0, 0
        boundaries = []
        history_tea = ""
        history_stu = ""

        while i < len(tea_token_texts) and j < len(stu_token_texts):
            is_eos_match = (tea_token_texts[i] == tea_eos and stu_token_texts[j] == stu_eos)
            if history_tea == history_stu and (
                tea_token_texts[i] == stu_token_texts[j] or is_eos_match
            ):
                boundaries.append((i, j))
                history_tea += tea_token_texts[i]
                history_stu += stu_token_texts[j]
                i += 1
                j += 1
            elif len(history_tea) > len(history_stu):
                history_stu += stu_token_texts[j]
                j += 1
            elif len(history_tea) < len(history_stu):
                history_tea += tea_token_texts[i]
                i += 1
            else:
                history_tea += tea_token_texts[i]
                history_stu += stu_token_texts[j]
                i += 1
                j += 1

        if len(boundaries) == 0:
            if self._debug_count < 3:
                self._debug_count += 1
                logger.warning(
                    f"[SpanCTKD DEBUG] No boundaries found! "
                    f"tea_token_texts({len(tea_token_texts)}): {tea_token_texts[:10]}... "
                    f"stu_token_texts({len(stu_token_texts)}): {stu_token_texts[:10]}..."
                )
            return [], tea_ids_list, stu_ids_list

        if self._debug_count < 3:
            self._debug_count += 1
            logger.info(
                f"[SpanCTKD DEBUG] Found {len(boundaries)} boundaries! "
                f"tea_texts({len(tea_token_texts)}): {tea_token_texts[:5]}... "
                f"stu_texts({len(stu_token_texts)}): {stu_token_texts[:5]}..."
            )

        # Convert boundaries to aligned segments (local indices within response)
        segments = []
        for idx in range(len(boundaries)):
            if idx == 0:
                local_tea_start, local_stu_start = 0, 0
            else:
                local_tea_start = boundaries[idx - 1][0] + 1
                local_stu_start = boundaries[idx - 1][1] + 1
            local_tea_end = boundaries[idx][0] + 1
            local_stu_end = boundaries[idx][1] + 1
            segments.append((local_tea_start, local_tea_end, local_stu_start, local_stu_end))

        return segments, tea_ids_list, stu_ids_list

    # ------------------------------------------------------------------
    # Build virtual common vocabulary logits for a single sample
    # ------------------------------------------------------------------
    def _build_virtual_vocab_logits(
        self,
        segments,
        stu_logits_aligned,
        tea_logits_aligned,
        stu_label_ids_list,
        tea_label_ids_list,
    ):
        """Build student and teacher logit matrices on the virtual common vocabulary.

        For each aligned segment we construct a logit vector of size
        ``num_overlap + num_total_spans`` where:
        - the first ``num_overlap`` dims come from the base overlap vocabulary,
        - the remaining ``num_total_spans`` dims correspond to span logits.

        For a 1:1 aligned position the overlap-vocab logits are taken directly
        from the single token; for a span the overlap-vocab logits are taken
        from the **first** token position only (since the span is treated as a
        single virtual token — other overlap tokens compete with it at the
        first position).  The span-logit dimensions are filled with ``-1e9``
        (≈ 0 after softmax) except for the span that the current segment
        belongs to, which gets the mean of each constituent token's logit at
        its own token id (i.e. the model's average confidence in generating
        this span).

        Args:
            segments: list of (tea_start, tea_end, stu_start, stu_end) —
                      indices into the label id sequences.
            stu_logits_aligned: student logits at loss_mask positions
                                [num_stu_loss_tokens, vocab_s].
            tea_logits_aligned: teacher logits at loss_mask positions
                                [num_tea_loss_tokens, vocab_t].
            stu_label_ids_list: Python list of student label token ids.
            tea_label_ids_list: Python list of teacher label token ids.

        Returns:
            stu_virtual_logits: [num_segments, num_overlap + num_spans]
            tea_virtual_logits: [num_segments, num_overlap + num_spans]
        """
        if getattr(self.args.kd, "span_score_mode", "raw_logit") == "mean_logprob":
            # Eq.7: normalize each autoregressive position BEFORE averaging.
            # Shared-token candidates must use log probabilities as well.
            stu_logits_aligned = torch.log_softmax(stu_logits_aligned.float(), dim=-1)
            tea_logits_aligned = torch.log_softmax(tea_logits_aligned.float(), dim=-1)
        num_overlap = self.student_overlap_token_ids.shape[0]
        device = stu_logits_aligned.device

        # Identify which segments are spans (multi-token on either side)
        span_indices = []  # indices into `segments` that are spans
        for seg_idx, (ts, te, ss, se) in enumerate(segments):
            if (te - ts) > 1 or (se - ss) > 1:
                span_indices.append(seg_idx)
        num_spans = len(span_indices)
        virtual_dim = num_overlap + num_spans

        # Gather overlap-vocabulary logits for every segment in one operation.
        # This is mathematically identical to the former per-segment loop, but
        # avoids launching one GPU indexing kernel per segment and vocabulary.
        stu_first_positions = torch.tensor(
            [ss for _ts, _te, ss, _se in segments], device=device, dtype=torch.long
        )
        tea_first_positions = torch.tensor(
            [ts for ts, _te, _ss, _se in segments], device=device, dtype=torch.long
        )
        stu_overlap = stu_logits_aligned[
            stu_first_positions.unsqueeze(1), self.student_overlap_token_ids.unsqueeze(0)
        ]
        tea_overlap = tea_logits_aligned[
            tea_first_positions.unsqueeze(1), self.teacher_overlap_token_ids.unsqueeze(0)
        ]

        if num_spans == 0:
            return stu_overlap, tea_overlap

        stu_span_values = []
        tea_span_values = []
        for seg_idx in span_indices:
            ts, te, ss, se = segments[seg_idx]
            stu_span_token_ids = torch.tensor(
                stu_label_ids_list[ss:se], device=device, dtype=torch.long
            )
            tea_span_token_ids = torch.tensor(
                tea_label_ids_list[ts:te], device=device, dtype=torch.long
            )
            stu_self_logits = stu_logits_aligned[
                torch.arange(ss, se, device=device), stu_span_token_ids
            ]
            tea_self_logits = tea_logits_aligned[
                torch.arange(ts, te, device=device), tea_span_token_ids
            ]
            stu_span_values.append(stu_self_logits.mean())
            tea_span_values.append(tea_self_logits.mean())

            if self._span_debug_count < 1:
                self._span_debug_count += 1
                logger.info(
                    f"\n[SpanCTKD SPAN DEBUG] === Example span logits (seg_idx={seg_idx}) ===\n"
                    f"  Student span token ids: {stu_span_token_ids.detach().cpu().tolist()}\n"
                    f"  Student self_logits (per token): {stu_self_logits.detach().cpu().tolist()}\n"
                    f"  Student span_dim value (mean):   {stu_self_logits.mean().item():.4f}\n"
                    f"  Teacher span token ids: {tea_span_token_ids.detach().cpu().tolist()}\n"
                    f"  Teacher self_logits (per token): {tea_self_logits.detach().cpu().tolist()}\n"
                    f"  Teacher span_dim value (mean):   {tea_self_logits.mean().item():.4f}\n"
                    f"  ---\n"
                    f"  Student overlap logits (first pos) stats: mean={stu_overlap[seg_idx].mean().item():.4f}, "
                    f"max={stu_overlap[seg_idx].max().item():.4f}, min={stu_overlap[seg_idx].min().item():.4f}, "
                    f"std={stu_overlap[seg_idx].std().item():.4f}\n"
                    f"  Teacher overlap logits (first pos) stats: mean={tea_overlap[seg_idx].mean().item():.4f}, "
                    f"max={tea_overlap[seg_idx].max().item():.4f}, min={tea_overlap[seg_idx].min().item():.4f}, "
                    f"std={tea_overlap[seg_idx].std().item():.4f}\n"
                    f"  num_overlap={num_overlap}, num_spans={num_spans}, virtual_dim={virtual_dim}"
                )

        stu_span_dims = torch.full(
            (len(segments), num_spans), -1e9, device=device, dtype=stu_overlap.dtype
        )
        tea_span_dims = torch.full(
            (len(segments), num_spans), -1e9, device=device, dtype=tea_overlap.dtype
        )
        span_rows = torch.tensor(span_indices, device=device, dtype=torch.long)
        span_columns = torch.arange(num_spans, device=device)
        stu_span_dims[span_rows, span_columns] = torch.stack(stu_span_values)
        tea_span_dims[span_rows, span_columns] = torch.stack(tea_span_values)
        return (
            torch.cat([stu_overlap, stu_span_dims], dim=1),
            torch.cat([tea_overlap, tea_span_dims], dim=1),
        )

    # ------------------------------------------------------------------
    # Re-normalization multiplier G(h) safeguard
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_segment_teacher_Z(self, segments, tea_logits_aligned, tea_label_ids_list):
        """Teacher mass ``Z_T(h)`` captured by the SimCT candidate space, per segment.

        ``Z_T(h)`` is computed on the **full-vocabulary** teacher softmax so that
        it measures the original teacher probability mass that the candidate
        space (shared tokens + the span realized at this position) represents:

        - shared-token mass at the first teacher position, plus
        - for a span, the geometric-mean probability of the span's teacher
          tokens (matching the span logit used in the virtual distribution).

        The re-normalization multiplier is then ``G(h) = 1 / Z_T(h)``.  A large
        ``G(h)`` means the candidate space holds little teacher mass, so
        re-normalizing over it would amplify a low-mass tail.

        Returns:
            Z: [num_segments] float tensor in (0, 1].
        """
        device = tea_logits_aligned.device
        if not segments:
            return torch.empty(0, device=device, dtype=torch.float32)

        # One full-vocabulary normalization per teacher-token row.  The former
        # reference implementation launched softmax/log_softmax separately for
        # every segment and every span token.  Using log-sum-exp identities is
        # the same probability calculation while batching the GPU work:
        #   sum_{v in overlap} softmax(x)_v
        # = exp(logsumexp(x_overlap) - logsumexp(x_full)).
        tea_logits_float = tea_logits_aligned.float()
        full_log_norm = torch.logsumexp(tea_logits_float, dim=-1)
        first_positions = torch.tensor(
            [ts for ts, _te, _ss, _se in segments], device=device, dtype=torch.long
        )
        first_logits = tea_logits_float[first_positions]
        overlap_log_mass = torch.logsumexp(
            first_logits[:, self.teacher_overlap_token_ids], dim=-1
        )
        Z = torch.exp(overlap_log_mass - full_log_norm[first_positions])

        for seg_idx, (ts, te, ss, se) in enumerate(segments):
            if (te - ts) <= 1 and (se - ss) <= 1:
                continue
            span_positions = torch.arange(ts, te, device=device)
            span_ids = torch.tensor(
                tea_label_ids_list[ts:te], device=device, dtype=torch.long
            )
            span_logps = (
                tea_logits_float[span_positions, span_ids]
                - full_log_norm[span_positions]
            )
            Z[seg_idx] = (Z[seg_idx] + torch.exp(span_logps.mean())).clamp(max=1.0)
        return Z.clamp(max=1.0)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(self, micro_batch):
        """Perform one training step with span-based cross-tokenizer KD loss.

        Flow:
        1. Student forward pass → full-sequence logits.
        2. Teacher logits from hidden states (loss_mask positions only).
        3. Per-sample: align sequences, identify spans, build virtual vocab
           logits, compute RKL loss.
        """
        student_input_ids = micro_batch["stu_input_ids"]
        student_attn_mask = micro_batch["stu_attn_mask"]
        student_loss_mask = micro_batch["stu_loss_mask"].bool()
        teacher_input_ids = micro_batch["tea_input_ids"]
        teacher_attn_mask = micro_batch["tea_attn_mask"]
        teacher_loss_mask = micro_batch["tea_loss_mask"].bool()
        teacher_hiddens = micro_batch.get("teacher_hiddens", None)
        avg_token_num = micro_batch["avg_micro_batch_token_num"]

        assert teacher_hiddens is not None, "micro_batch must contain `teacher_hiddens` for KD"

        mm_kwargs = {k[3:]: v for k, v in micro_batch.items() if k.startswith("mm_")}

        # Student forward
        output = self.student(
            student_input_ids,
            attention_mask=student_attn_mask,
            allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,
            **mm_kwargs,
        )
        student_logits = output["logits"]

        # Extract label ids (next-token) at loss_mask positions — consistent
        # with simple_ctkd which aligns on labels, not inputs.
        student_label_ids = student_input_ids.roll(shifts=-1, dims=1)
        teacher_label_ids = teacher_input_ids.roll(shifts=-1, dims=1)

        # Flatten student logits at loss_mask positions (same as simple_ctkd)
        student_logits_flat = student_logits[student_loss_mask]  # [total_stu_loss_tokens, vocab_s]

        # Free full student logits early — only loss_mask positions are needed
        del student_logits, output

        # NOTE: `teacher_hiddens` from teacher_actor is ALREADY 2D and filtered
        # by loss_mask — shape [total_tea_loss_tokens, hidden_size]. See
        # teacher_actor.py where hidden states are gathered only at loss_mask
        # positions and np.concatenate-d across the micro-batch. Do NOT re-index
        # with teacher_loss_mask (that would be shape [B, T] and mismatch).
        teacher_hiddens = teacher_hiddens.to(self.teacher_lm_head.weight)
        teacher_logits_flat = self.teacher_lm_head(teacher_hiddens)  # [total_tea_loss_tokens, vocab_t]
        del teacher_hiddens  # free after lm_head computation

        # Process each sample in the batch
        batch_size = student_input_ids.shape[0]
        total_loss = torch.tensor(0.0, device=student_logits_flat.device, requires_grad=True)
        total_aligned_tokens = 0
        total_response_tokens = 0
        total_kept_segments = 0
        total_masked_segments = 0
        total_segments = 0
        total_span_segments = 0
        total_z_sum = 0.0
        total_z_count = 0
        min_teacher_candidate_mass = 1.0
        max_teacher_candidate_mass = 0.0

        # Offset trackers for flattened logits (both are 2D across batch)
        tea_logits_offset = 0
        stu_logits_offset = 0

        for b in range(batch_size):
            stu_mask = student_loss_mask[b]
            tea_mask = teacher_loss_mask[b]

            stu_num_loss_tokens = stu_mask.sum().item()
            tea_num_loss_tokens = tea_mask.sum().item()

            # Student logits for this sample (already flattened by loss_mask)
            stu_logits_b = student_logits_flat[stu_logits_offset:stu_logits_offset + stu_num_loss_tokens]
            stu_logits_offset += stu_num_loss_tokens

            # Teacher logits for this sample (already flattened by loss_mask)
            tea_logits_b = teacher_logits_flat[tea_logits_offset:tea_logits_offset + tea_num_loss_tokens]
            tea_logits_offset += tea_num_loss_tokens

            # Extract label ids at loss_mask positions
            stu_label_ids_b = student_label_ids[b][stu_mask].cpu().tolist()
            tea_label_ids_b = teacher_label_ids[b][tea_mask].cpu().tolist()

            # Align on label ids (next-token) and identify spans
            segments, tea_ids_list, stu_ids_list = self._align_sequences_with_spans(
                tea_label_ids_b, stu_label_ids_b
            )

            if len(segments) == 0:
                total_response_tokens += max(stu_num_loss_tokens, tea_num_loss_tokens)
                continue

            total_segments += len(segments)
            total_span_segments += sum(
                1 for ts, te, ss, se in segments
                if (te - ts) > 1 or (se - ss) > 1
            )

            # Build virtual vocab logits and compute loss.
            # stu_logits_b and tea_logits_b are already [num_loss_tokens, vocab]
            # and segments index directly into them.
            stu_virtual, tea_virtual = self._build_virtual_vocab_logits(
                segments, stu_logits_b, tea_logits_b,
                stu_ids_list, tea_ids_list,
            )

            assert stu_virtual.shape == tea_virtual.shape, \
                f"Virtual logit shape mismatch: student {stu_virtual.shape} vs teacher {tea_virtual.shape}"

            # Compute RKL loss on virtual common vocabulary
            if self.args.kd.kd_loss_fn == "rkl":
                sample_loss = memory_efficient_reverse_kl(
                    stu_virtual,
                    tea_virtual,
                    temperature=self.args.kd.kd_temperature,
                )
            else:
                sample_loss = self.loss_fn(
                    stu_virtual,
                    tea_virtual.detach(),
                    reduction="none",
                )

            # Re-normalization safeguard: drop positions whose multiplier
            # G(h) = 1 / Z_T(h) exceeds the threshold (teacher captured mass
            # Z_T(h) < 1/threshold). These low-mass positions would otherwise
            # let re-normalization amplify a weak/noisy teacher tail, so they
            # contribute no distillation gradient.
            gh_threshold = getattr(self.args.kd, "span_gh_mask_threshold", 0.0)
            if gh_threshold and gh_threshold > 0:
                Z = self._compute_segment_teacher_Z(segments, tea_logits_b, tea_ids_list)
                total_z_sum += float(Z.sum().item())
                total_z_count += int(Z.numel())
                min_teacher_candidate_mass = min(
                    min_teacher_candidate_mass, float(Z.min().item())
                )
                max_teacher_candidate_mass = max(
                    max_teacher_candidate_mass, float(Z.max().item())
                )
                keep = (Z >= (1.0 / gh_threshold)).to(sample_loss.dtype)  # G(h) <= threshold
                total_kept_segments += int(keep.sum().item())
                total_masked_segments += int((keep.numel() - keep.sum()).item())
                sample_loss = sample_loss * keep

            total_loss = total_loss + sample_loss.sum()

            # Count aligned tokens (sum of all segment lengths on student side)
            for ts, te, ss, se in segments:
                total_aligned_tokens += max(te - ts, se - ss)
            total_response_tokens += max(stu_num_loss_tokens, tea_num_loss_tokens)

        # Normalize
        kd_loss = total_loss / avg_token_num

        align_ratio = torch.tensor(
            total_aligned_tokens / max(total_response_tokens, 1),
            device=student_logits_flat.device,
        )

        loss_info = {"loss": kd_loss, "kd_loss": kd_loss, "align_ratio": align_ratio}

        gh_masked_total = total_kept_segments + total_masked_segments
        loss_info["gh_mask_ratio"] = torch.tensor(
            total_masked_segments / max(gh_masked_total, 1),
            device=student_logits_flat.device,
        )
        # Exact, algorithm-native counters.  StudentActor uses these to expose
        # meaningful token throughput for SimCT (rather than the previous zero
        # fallback that was only populated by X-Token).
        metric_device = student_logits_flat.device
        loss_info["valid_student_tokens"] = torch.tensor(
            float(student_loss_mask.sum().item()), device=metric_device
        )
        loss_info["valid_teacher_tokens"] = torch.tensor(
            float(teacher_loss_mask.sum().item()), device=metric_device
        )
        loss_info["aligned_token_count"] = torch.tensor(
            float(total_aligned_tokens), device=metric_device
        )
        loss_info["aligned_segment_count"] = torch.tensor(
            float(total_segments), device=metric_device
        )
        loss_info["span_segment_count"] = torch.tensor(
            float(total_span_segments), device=metric_device
        )
        loss_info["span_segment_ratio"] = torch.tensor(
            total_span_segments / max(total_segments, 1), device=metric_device
        )
        loss_info["gh_kept_segment_count"] = torch.tensor(
            float(total_kept_segments), device=metric_device
        )
        loss_info["gh_masked_segment_count"] = torch.tensor(
            float(total_masked_segments), device=metric_device
        )
        if total_z_count:
            loss_info["teacher_candidate_mass_mean"] = torch.tensor(
                total_z_sum / total_z_count, device=metric_device
            )
            loss_info["teacher_candidate_mass_min"] = torch.tensor(
                min_teacher_candidate_mass, device=metric_device
            )
            loss_info["teacher_candidate_mass_max"] = torch.tensor(
                max_teacher_candidate_mass, device=metric_device
            )

        if self.args.kd.kd_ratio < 1:
            ce_label_ids = student_label_ids[student_loss_mask]
            ce_loss = compute_cross_entropy(student_logits_flat, ce_label_ids, reduction="sum") / avg_token_num
            loss = (1 - self.args.kd.kd_ratio) * ce_loss + self.args.kd.kd_ratio * kd_loss
            loss_info["loss"] = loss
            loss_info["ce_loss"] = ce_loss

        return loss_info
