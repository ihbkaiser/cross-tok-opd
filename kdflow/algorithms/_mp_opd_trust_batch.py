"""Exact full-batch TRUST head updates, streamed without retaining forward graphs."""
from __future__ import annotations

import torch

from ._mp_opd_grass_span import _exact_fp32, _log_partition_from_selected
from ._mp_opd_trust import trust_calibrate_scalars


class TrustHeadAccumulator:
    """Sum strict/mismatch LM-head gradients, including cross-response terms.

    Storage is two [vocab, hidden] fp32 tensors, independent of batch token count.
    Vocabulary tiles bound transients; no [batch tokens, vocab] concatenation and
    no atom-by-atom global Gram are constructed. Loss normalization is one common
    factor for the entire accumulation window and cancels from lambda.
    """

    def __init__(self, *, softcap=None, head_bias=False, vocab_chunk=4096):
        self.softcap = softcap
        self.head_bias = head_bias
        self.vocab_chunk = int(vocab_chunk)
        if self.vocab_chunk < 1:
            raise ValueError("TRUST vocabulary tile must be positive")
        self.gs = self.gm = self.bs = self.bm = None
        self.responses = []

    @torch.no_grad()
    def add_response(self, logits, hidden, labels, ranges, rate, is_strict,
                     selected_log_prob):
        covered = ranges[-1][1]
        logits, hidden, labels = logits[:covered], hidden[:covered], labels[:covered]
        lengths = torch.tensor([b-a for a, b in ranges], device=logits.device)
        if sum(b-a for a, b in ranges) != covered or ranges[0][0] != 0:
            raise ValueError("TRUST ranges must tile the covered token prefix")
        if any(ranges[i][1] != ranges[i+1][0] for i in range(len(ranges)-1)):
            raise ValueError("TRUST ranges must have no gaps")
        if not all(bool(torch.isfinite(t).all()) for t in (logits, hidden, rate)):
            raise FloatingPointError("non-finite full-batch TRUST input")
        vocab, width = int(logits.shape[1]), int(hidden.shape[1])
        if self.gs is None:
            need = 2 * vocab * width * 4 + (2 * vocab * 4 if self.head_bias else 0)
            if logits.is_cuda:
                free, _ = torch.cuda.mem_get_info(logits.device)
                if need > free * 0.5:
                    raise ValueError("full-batch TRUST head accumulators exceed memory guard")
            self.gs = torch.zeros(vocab, width, dtype=torch.float32, device=logits.device)
            self.gm = torch.zeros_like(self.gs)
            if self.head_bias:
                self.bs = torch.zeros(vocab, dtype=torch.float32, device=logits.device)
                self.bm = torch.zeros_like(self.bs)
        if self.gs.shape != (vocab, width):
            raise ValueError("student head shape changed within TRUST batch")
        strict = is_strict.to(device=logits.device)
        q = rate.detach().float().repeat_interleave(lengths)
        token_strict = strict.repeat_interleave(lengths)
        qs = q * token_strict
        qm = q * ~token_strict
        hs = hidden.detach().float() * qs.unsqueeze(1)
        hm = hidden.detach().float() * qm.unsqueeze(1)
        logz = _log_partition_from_selected(logits, labels, selected_log_prob[:covered]).float()
        with _exact_fp32():
            for low in range(0, vocab, self.vocab_chunk):
                high = min(low + self.vocab_chunk, vocab)
                z = logits[:, low:high].detach().float()
                delta = torch.exp(z - logz.unsqueeze(1))
                jacobian = None
                if self.softcap is not None:
                    jacobian = (1.0 - (z / float(self.softcap)).clamp(-1, 1).square()).clamp_min(0)
                    delta *= jacobian
                rows = ((labels >= low) & (labels < high)).nonzero().flatten()
                cols = labels[rows] - low
                delta[rows, cols] -= 1.0 if jacobian is None else jacobian[rows, cols]
                self.gs[low:high].add_(delta.T @ hs)
                self.gm[low:high].add_(delta.T @ hm)
                if self.head_bias:
                    self.bs[low:high].add_(qs @ delta)
                    self.bm[low:high].add_(qm @ delta)
        self.responses.append({
            "strict_atoms": int(strict.sum()), "mismatch_atoms": int((~strict).sum()),
            "strict_tokens": int(lengths[strict].sum()),
            "mismatch_tokens": int(lengths[~strict].sum()),
        })

    @torch.no_grad()
    def finalize(self, *, eps_g=1e-12, distributed=False):
        if distributed:
            import torch.distributed as dist
            # Every rank uses the same pinned student head shape. Empty ranks
            # allocate matching buffers after exchanging shapes, before collectives.
            shapes = [None] * dist.get_world_size()
            dist.all_gather_object(shapes, None if self.gs is None else tuple(self.gs.shape))
            shape = next((s for s in shapes if s is not None), None)
            if any(s is not None and s != shape for s in shapes):
                raise ValueError("TRUST ranks have inconsistent student head shapes")
            if shape is not None and self.gs is None:
                dev = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
                self.gs = torch.zeros(shape, dtype=torch.float32, device=dev)
                self.gm = torch.zeros_like(self.gs)
                if self.head_bias:
                    self.bs = torch.zeros(shape[0], dtype=torch.float32, device=dev)
                    self.bm = torch.zeros_like(self.bs)
            if shape is not None:
                for tensor in (self.gs, self.gm, self.bs, self.bm):
                    if tensor is not None:
                        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            records = [None] * dist.get_world_size()
            dist.all_gather_object(records, self.responses)
            self.responses = [r for group in records for r in group]
        gs2 = gm2 = dot = 0.0
        if self.gs is not None:
            for low in range(0, self.gs.shape[0], self.vocab_chunk):
                s = self.gs[low:low+self.vocab_chunk].double()
                m = self.gm[low:low+self.vocab_chunk].double()
                gs2 += float(s.square().sum())
                gm2 += float(m.square().sum())
                dot += float((s*m).sum())
            if self.head_bias:
                gs2 += float(self.bs.double().square().sum())
                gm2 += float(self.bm.double().square().sum())
                dot += float((self.bs.double()*self.bm.double()).sum())
        n_s = sum(r["strict_atoms"] for r in self.responses)
        n_m = sum(r["mismatch_atoms"] for r in self.responses)
        result = trust_calibrate_scalars(gs2, gm2, dot, n_s, n_m, eps_g=eps_g)
        self.gs = self.gm = self.bs = self.bm = None
        return result
