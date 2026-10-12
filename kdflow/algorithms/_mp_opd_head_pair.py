"""Full-response output-edge head geometry, including cross-chunk terms.

Two coefficient-weighted head updates are reduced in vocabulary tiles. No
global atom Gram or persistent [vocab, hidden] update is allocated. This is the
LM-head output contribution, not a full-Transformer (or tied-input) gradient.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ._mp_opd_grass_span import _exact_fp32, _log_partition_from_selected


@dataclass(frozen=True)
class HeadPairGeometry:
    first_norm2: float
    second_norm2: float
    dot: float


def checked_geometry(first_norm2, second_norm2, dot) -> HeadPairGeometry:
    values = tuple(float(v) for v in (first_norm2, second_norm2, dot))
    if not all(math.isfinite(v) for v in values):
        raise FloatingPointError("head pair geometry must be finite")
    a, b, d = values
    if a < 0 or b < 0:
        raise ValueError("head pair squared norms must be nonnegative")
    bound = math.sqrt(a) * math.sqrt(b)
    if abs(d) > bound * (1 + 1e-6) + 1e-300:
        raise ValueError("head pair geometry violates Cauchy-Schwarz")
    return HeadPairGeometry(a, b, max(-bound, min(d, bound)))


@torch.no_grad()
def head_pair_geometry(logits, hidden, labels, ranges, first, second, *,
                       selected_log_prob, softcap=None, head_bias=False,
                       vocab_chunk=4096) -> HeadPairGeometry:
    """Read <sum first_i G_i, sum second_i G_i> in exact fp32 head geometry.

    Each atom loss is a TOKEN SUM. Thus an atom's coefficient is repeated on
    every token, with no division by atom length. Bias and final softcap use
    precisely the same output-edge derivative as the training loss.
    """
    ranges = tuple(ranges)
    if (not ranges or ranges[0][0] != 0 or
            any(b <= a for a, b in ranges) or
            any(ranges[i][1] != ranges[i+1][0] for i in range(len(ranges)-1))):
        raise ValueError("head pair ranges must tile a nonempty covered prefix")
    covered = ranges[-1][1]
    if logits.ndim != 2 or hidden.ndim != 2 or labels.ndim != 1:
        raise ValueError("head pair expects token-by-vocab/hidden and flat labels")
    if min(logits.shape[0], hidden.shape[0], labels.numel(), selected_log_prob.numel()) < covered:
        raise ValueError("head pair input is shorter than covered prefix")
    if first.ndim != 1 or second.shape != first.shape or first.numel() != len(ranges):
        raise ValueError("head pair coefficients must match atom count")
    if labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("head pair labels must be integer token IDs")
    if any(t.device != logits.device for t in (hidden, labels, first, second, selected_log_prob)):
        raise ValueError("head pair inputs must share one device")
    vocab, width = logits.shape[1], hidden.shape[1]
    if vocab < 1 or width < 1 or int(vocab_chunk) < 1:
        raise ValueError("head pair dimensions and vocabulary tile must be positive")
    if softcap is not None and not (math.isfinite(float(softcap)) and float(softcap) > 0):
        raise ValueError("head pair softcap must be finite and positive")
    logits, hidden, labels = logits[:covered], hidden[:covered], labels[:covered]
    selected_log_prob = selected_log_prob[:covered]
    if not all(bool(torch.isfinite(t).all()) for t in (logits, hidden, first, second, selected_log_prob)):
        raise FloatingPointError("non-finite head pair input")
    if bool(((labels < 0) | (labels >= vocab)).any()):
        raise ValueError("head pair label outside vocabulary")
    tile = min(int(vocab_chunk), vocab)
    # Conservative transient bound: logits/probabilities/Jacobian tiles, two
    # head tiles including fp64 reduction temporaries, and weighted hidden states.
    need = 16 * covered * tile + 48 * tile * width + 16 * covered * width
    if logits.is_cuda and need > torch.cuda.mem_get_info(logits.device)[0] * .5:
        raise ValueError("head pair vocabulary tiles exceed memory guard")
    lengths = torch.tensor([b-a for a, b in ranges], device=logits.device)
    qa = first.detach().float().repeat_interleave(lengths)
    qb = second.detach().float().repeat_interleave(lengths)
    ha = hidden.detach().float() * qa[:, None]
    hb = hidden.detach().float() * qb[:, None]
    logz = _log_partition_from_selected(logits, labels, selected_log_prob).float()
    a2 = b2 = dot = 0.0
    # Force exact fp32 irrespective of outer autocast; a bf16 matmul would make
    # this guard measure a different update from the stated fp32 head proxy.
    with _exact_fp32(), torch.autocast(device_type=logits.device.type, enabled=False):
        for low in range(0, vocab, tile):
            high = min(low+tile, vocab)
            z = logits[:, low:high].detach().float()
            delta = torch.exp(z - logz[:, None])
            jac = None
            if softcap is not None:
                jac = (1 - (z / float(softcap)).clamp(-1, 1).square()).clamp_min(0)
                delta *= jac
            rows = ((labels >= low) & (labels < high)).nonzero().flatten()
            cols = labels[rows] - low
            delta[rows, cols] -= 1.0 if jac is None else jac[rows, cols]
            ga, gb = (delta.T @ ha).double(), (delta.T @ hb).double()
            a2 += float(ga.square().sum())
            b2 += float(gb.square().sum())
            dot += float((ga*gb).sum())
            if head_bias:
                ba, bb = (qa @ delta).double(), (qb @ delta).double()
                a2 += float(ba.square().sum())
                b2 += float(bb.square().sum())
                dot += float((ba*bb).sum())
    return checked_geometry(a2, b2, dot)
