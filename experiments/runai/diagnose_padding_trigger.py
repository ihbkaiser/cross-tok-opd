"""Find which property of the capture's own arrays makes a padded forward differ.

Confirmed on `failure-zusi05xj`: a padded-batch eager forward reproduces the trainer's
stored logprobs BIT-EXACTLY (max 0.0) while a per-row forward reproduces the engine's
(max 0.33); the two differ by 1.7597 max on 150 positions. A synthetic sweep that copied
the capture's geometry (lengths 506/224/4096/644, longest at index 2, pad id 0, train
mode, position_ids = cumsum-1 with pads forced to 1) reports max|delta logit| = 0.00000
for every variant. So the trigger is a property of the capture's DATA, not its shape.

This ablates the real arrays one input at a time and reports the content-independent
statistic `max |delta logit|` between the padded forward and the per-row forward:

  base            ids + attention_mask exactly as stored, position_ids = cumsum-1
  base_nopos      same, but no position_ids passed
  pad_zero        padded region of every row replaced by token id 0
  attn_prefix     attention_mask rebuilt as a plain prefix-ones mask from its own sum
  attn_all_ones   attention_mask replaced by all ones
  long_only_pad   single-row batch of the long row, padded to the stored width
  long_only       the long row alone at its own length (no padding at all)

If `attn_prefix` or `pad_zero` collapses the delta, that names the trigger.

Read-only and structure-only: it prints mask shape statistics and deltas, never prompt
text or token ids.

  CUDA_VISIBLE_DEVICES=0 bash $SRC/experiments/runai/python-b200-host.sh \
    /tmp/diagnose_padding_trigger.py "$CAPDIR" --out /tmp/pad_trigger.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def max_logit_delta(a, b, chunk: int = 256) -> dict[str, Any]:
    """max |a - b| over the vocabulary, per position, in chunks."""
    n = int(min(a.shape[0], b.shape[0]))
    worst, worst_pos = 0.0, -1
    means: list[float] = []
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        d = (a[lo:hi].float() - b[lo:hi].float()).abs()
        means.append(float(d.mean()))
        m = float(d.max())
        if m > worst:
            worst, worst_pos = m, lo + int(d.max(dim=1).values.argmax())
        del d
    return {"positions": n, "max_abs_dlogit": worst, "worst_position": worst_pos,
            "mean_abs_dlogit": sum(means) / len(means) if means else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    ap.add_argument("--row", type=int, default=-1, help="row to compare; -1 = longest")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    meta = json.loads((args.capture / "metadata.json").read_text())
    batch = torch.load(args.capture / "batch.pt", map_location="cpu", weights_only=True)
    ids = batch["stu_input_ids"].long()
    attn = batch["stu_attn_mask"].long()
    lengths = [int(v) for v in attn.sum(-1).tolist()]
    width = int(ids.shape[1])
    r = args.row if args.row >= 0 else max(range(len(lengths)), key=lambda i: lengths[i])

    report: dict[str, Any] = {
        "capture": str(args.capture), "rows": int(ids.shape[0]), "width": width,
        "lengths": lengths, "row": r,
        "model_training": meta.get("model_training"),
        "attention_backend": meta.get("attention_backend"),
    }
    print("TRIGGER_META=" + json.dumps(
        {k: report[k] for k in ("rows", "width", "lengths", "row", "model_training")},
        sort_keys=True), flush=True)

    # Structural description of the masks only: counts and whether the ones form a
    # plain prefix. No token ids and no text leave the machine.
    struct = []
    for i in range(ids.shape[0]):
        a = attn[i]
        ones = int(a.sum())
        first_zero = int((a == 0).nonzero()[0]) if bool((a == 0).any()) else -1
        struct.append({"row": i, "ones": ones, "first_zero": first_zero,
                       "is_prefix_ones": bool(int(a[:ones].sum()) == ones)})
    report["mask_structure"] = struct
    print("TRIGGER_MASK_STRUCTURE=" + json.dumps(struct), flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.capture / "student", dtype=getattr(torch, args.dtype),
        attn_implementation="eager", local_files_only=True,
        trust_remote_code=False).cuda()
    model.train(bool(meta.get("model_training")))

    n_r = int(lengths[r])
    lone_ids = ids[r:r + 1, :n_r].clone()

    def positions_for(a):
        p = a.long().cumsum(-1) - 1
        p.masked_fill_(a == 0, 1)
        return p

    pos_stored = positions_for(attn)

    # ---------------------------------------------------------------------
    # Per-row delta for the stored configuration, for EVERY row.
    #
    # Discriminator between two mechanisms: if rows AFTER the first are affected
    # (row 1 sees row 0, row 3 sees rows 0-2) then the trainer's forward couples
    # rows -- a concatenation/packing-style defect. If only the LONGEST row is
    # affected it is a length/width effect. The two have different fixes.
    # ---------------------------------------------------------------------
    with torch.inference_mode():
        padded_all = model(input_ids=ids.cuda(), attention_mask=attn.cuda(),
                           position_ids=pos_stored.cuda(), use_cache=False).logits
    per_row: dict[str, Any] = {}
    for i in range(ids.shape[0]):
        n_i = int(lengths[i])
        with torch.inference_mode():
            ref_i = model(input_ids=ids[i:i + 1, :n_i].cuda(), use_cache=False).logits[0]
        per_row[str(i)] = max_logit_delta(padded_all[i, :n_i], ref_i)
        del ref_i
        torch.cuda.empty_cache()
        print("TRIGGER_PER_ROW=" + json.dumps(
            {"row": i, "length": n_i, **per_row[str(i)]}, sort_keys=True), flush=True)
    del padded_all
    torch.cuda.empty_cache()
    report["per_row_stored_config"] = per_row

    # Structure of the padded region only: is it a single constant pad value? No
    # token ids are printed, just counts.
    pad_region = []
    for i in range(ids.shape[0]):
        L = int(lengths[i])
        if L >= width:
            pad_region.append({"row": i, "padded_tokens": 0})
            continue
        seg = ids[i, L:]
        uniq = int(torch.unique(seg).numel())
        pad_region.append({"row": i, "padded_tokens": int(seg.numel()),
                           "distinct_pad_values": uniq, "all_equal": bool(uniq == 1)})
    report["pad_region_structure"] = pad_region
    print("TRIGGER_PAD_REGION=" + json.dumps(pad_region), flush=True)

    prefix = torch.zeros_like(attn)
    for i, L in enumerate(lengths):
        prefix[i, :L] = 1
    pad_ids = torch.zeros((1, width), dtype=torch.long)
    pad_attn = torch.zeros((1, width), dtype=torch.long)
    pad_ids[0, :n_r] = ids[r, :n_r]
    pad_attn[0, :n_r] = 1

    n_rows = int(ids.shape[0])
    # One reference per row: that row forwarded alone, no mask and no position_ids.
    # This is the semantics SGLang serves, and it is what the fix must reproduce.
    refs: list[Any] = []
    for i in range(n_rows):
        n_i = int(lengths[i])
        with torch.inference_mode():
            refs.append(model(input_ids=ids[i:i + 1, :n_i].cuda(),
                              use_cache=False).logits[0])
        torch.cuda.empty_cache()

    plans: list[tuple[str, Any, Any, Any, list[int]]] = [
        ("base", ids, attn, pos_stored, list(range(n_rows))),
        ("base_nopos", ids, attn, None, list(range(n_rows))),
        ("pad_zero", torch.where(attn.bool(), ids, torch.zeros_like(ids)), attn,
         pos_stored, list(range(n_rows))),
        ("attn_prefix", ids, prefix, None, list(range(n_rows))),
        ("attn_all_ones", ids, torch.ones_like(attn), None, list(range(n_rows))),
        ("long_only_pad", pad_ids, pad_attn, None, [r]),
        ("long_only", lone_ids, torch.ones_like(lone_ids), None, [r]),
    ]
    variants: dict[str, dict[str, Any]] = {}
    per_variant_row: dict[str, dict[str, Any]] = {}
    for name, x, a, p, row_map in plans:
        with torch.inference_mode():
            out = model(input_ids=x.cuda(),
                        attention_mask=(a.cuda() if a is not None else None),
                        position_ids=(p.cuda() if p is not None else None),
                        use_cache=False).logits
        out3 = out if out.dim() == 3 else out.unsqueeze(0)
        del out
        torch.cuda.empty_cache()
        per_variant_row[name] = {}
        for j, capture_row in enumerate(row_map):
            n_i = int(lengths[capture_row])
            d = max_logit_delta(out3[j, :n_i], refs[capture_row])
            per_variant_row[name][str(capture_row)] = d
            print("TRIGGER_VARIANT_ROW=" + json.dumps(
                {"variant": name, "row": capture_row, "length": n_i,
                 "max_abs_dlogit": d["max_abs_dlogit"],
                 "mean_abs_dlogit": round(d["mean_abs_dlogit"], 6)}, sort_keys=True),
                flush=True)
        del out3
        torch.cuda.empty_cache()
        # Headline = the longest row, which is the one the parity guard trips on.
        variants[name] = per_variant_row[name][str(r)]
        print("TRIGGER_VARIANT=" + json.dumps({"variant": name, **variants[name]},
                                             sort_keys=True), flush=True)
    del refs
    torch.cuda.empty_cache()
    report["per_variant_row"] = per_variant_row
    # The right criterion is NOT "every row goes to zero". The short rows carry a
    # separate batched-vs-single-row numerical difference that has nothing to do with
    # position_ids and does not move the guard (their logprob deltas stay small
    # because their distributions are peaked). What matters is that the row the guard
    # actually trips on -- the longest one -- collapses to zero, and that no row gets
    # worse.
    fixed = {k: round(v["max_abs_dlogit"], 6)
             for k, v in per_variant_row.get("base_nopos", {}).items()}
    broken = {k: round(v["max_abs_dlogit"], 6)
              for k, v in per_variant_row.get("base", {}).items()}
    longest_fixed = fixed.get(str(r), None) == 0.0
    none_worse = all(fixed.get(k, 0.0) <= v for k, v in broken.items())
    report["fix_check"] = {
        "base_max_per_row": broken, "base_nopos_max_per_row": fixed,
        "longest_row_is_bit_identical_to_per_row": longest_fixed,
        "no_row_gets_worse": none_worse,
        "verdict": "fix_confirmed" if (longest_fixed and none_worse) else "investigate",
    }
    print("TRIGGER_FIX_CHECK=" + json.dumps(report["fix_check"], sort_keys=True), flush=True)

    report["variants"] = variants
    ranked = sorted((v["max_abs_dlogit"], k) for k, v in variants.items())
    report["smallest_delta_variant"] = ranked[0][1] if ranked else None
    print("TRIGGER_RANKED=" + json.dumps(
        {"smallest": ranked[0][1] if ranked else None,
         "deltas": {k: round(v["max_abs_dlogit"], 6) for k, v in variants.items()}},
        sort_keys=True), flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
        print("TRIGGER_OUT=" + str(args.out), flush=True)
    print("TRIGGER_DONE=1", flush=True)


if __name__ == "__main__":
    main()
