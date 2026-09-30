"""Memory-lean, higher-order-safe selected-token log-probabilities.

Why this exists: the MP-OPD credit path needs log p(y_i) for the sampled tokens
only, but the naive path materialises the full [tokens, vocab] logits, an fp32
copy of them and a full log_softmax before gathering one value per row. At vocab
256k and 4096 tokens that is ~2 GiB (bf16 logits) plus ~2x4 GiB of fp32
transients for a single long sample, and it scales with microbatch size.

Two levels are provided:
  1. masked_selected_logprobs        - keep [N, V] logits but never build the fp32
     copy or the full log_softmax: log-sum-exp runs over vocabulary chunks.
  2. chunked_head_selected_logprobs  - never build [N, V] at all: the LM head is
     applied one vocabulary chunk at a time.

Both use ordinary differentiable ops (max/exp/sum/log/matmul) so double backward
keeps working - the property the full-meta energy path depends on, and the reason
a custom fused kernel cannot be dropped in blindly.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _apply_softcap(logits: torch.Tensor, softcap: Optional[float]) -> torch.Tensor:
    """Gemma2-style final logit soft capping, applied before normalisation."""
    if softcap is None:
        return logits
    return torch.tanh(logits / softcap) * softcap


def accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    """Accumulate in fp32 for half precision, but keep fp64 for gradcheck."""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _logsumexp_chunked(logits: torch.Tensor, vocab_chunk: int) -> torch.Tensor:
    """Row-wise logsumexp over the vocabulary, chunked along the vocab axis."""
    if logits.ndim != 2:
        raise ValueError("logits must be [rows, vocab]")
    rows, vocab = logits.shape
    if vocab_chunk < 1:
        raise ValueError("vocab_chunk must be >= 1")
    acc = accumulation_dtype(logits.dtype)
    running_max = torch.full((rows,), float("-inf"), dtype=acc, device=logits.device)
    running_sum = torch.zeros((rows,), dtype=acc, device=logits.device)
    for start in range(0, vocab, vocab_chunk):
        chunk = logits[:, start:start + vocab_chunk].to(acc)
        chunk_max = chunk.max(dim=-1).values
        new_max = torch.maximum(running_max, chunk_max)
        scaled = running_sum * torch.exp(running_max - new_max)
        running_sum = scaled + torch.exp(chunk - new_max.unsqueeze(-1)).sum(dim=-1)
        running_max = new_max
    return running_max + torch.log(running_sum)


def masked_selected_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    vocab_chunk: int = 16384,
    softcap: Optional[float] = None,
) -> torch.Tensor:
    """log p(label) per row without materialising fp32 logits or log_softmax."""
    if logits.ndim != 2:
        raise ValueError("logits must be [rows, vocab]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must be [rows]")
    logits = _apply_softcap(logits, softcap)
    log_z = _logsumexp_chunked(logits, vocab_chunk)
    target = logits.gather(1, labels.long().unsqueeze(1)).squeeze(1).to(log_z.dtype)
    return target - log_z


def chunked_head_selected_logprobs(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    vocab_chunk: int = 16384,
    softcap: Optional[float] = None,
) -> torch.Tensor:
    """Same values as applying the LM head then masked_selected_logprobs."""
    if hidden.ndim != 2:
        raise ValueError("hidden must be [rows, hidden_dim]")
    if weight.ndim != 2 or weight.shape[1] != hidden.shape[1]:
        raise ValueError("weight must be [vocab, hidden_dim]")
    if labels.ndim != 1 or labels.shape[0] != hidden.shape[0]:
        raise ValueError("labels must be [rows]")
    vocab = weight.shape[0]
    rows = hidden.shape[0]
    acc = accumulation_dtype(hidden.dtype)
    running_max = torch.full((rows,), float("-inf"), dtype=acc, device=hidden.device)
    running_sum = torch.zeros((rows,), dtype=acc, device=hidden.device)
    target = torch.zeros((rows,), dtype=acc, device=hidden.device)
    labels = labels.long()
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= vocab):
        raise ValueError('labels must be within [0, vocab)')
    for start in range(0, vocab, vocab_chunk):
        stop = min(start + vocab_chunk, vocab)
        bias_chunk = None if bias is None else bias[start:stop]
        chunk = F.linear(hidden, weight[start:stop], bias_chunk)
        chunk = _apply_softcap(chunk, softcap).to(acc)
        chunk_max = chunk.max(dim=-1).values
        new_max = torch.maximum(running_max, chunk_max)
        scaled = running_sum * torch.exp(running_max - new_max)
        running_sum = scaled + torch.exp(chunk - new_max.unsqueeze(-1)).sum(dim=-1)
        running_max = new_max
        in_chunk = (labels >= start) & (labels < stop)
        if bool(in_chunk.any()):
            # gather runs for every row, so out-of-chunk rows (label not in this
            # chunk at all) must be clamped into range instead of indexing past
            # the end of the chunk.
            index = (labels - start).clamp(0, stop - start - 1).unsqueeze(1)
            picked = chunk.gather(1, index).squeeze(1)
            target = torch.where(in_chunk, picked, target)
    log_z = running_max + torch.log(running_sum)
    return target - log_z
