"""Decisive diagnostics for an MP-OPD parity capture, on the capture itself.

`BUG_parity_logprob_4096cap.md` leaves one open question and one contradiction:

  * why the divergence appears only on the sample that ran to the 4096 cap, in a
    contiguous middle band with 150 outliers at stride 8;
  * section 2.4 reports a 24x difference between the replay tool's `--layout batch`
    and `--layout unpadded`, while the author's hand-written per-row strip test
    showed no difference.

A synthetic B200 probe could not reproduce the anomaly with any single-variable
change (see LONGCAP_PARITY_PROBE.md), so the remaining evidence is the capture.
This script answers, from the capture alone and with one command:

  1. Does a *fresh* eager recomputation reproduce the stored trainer logprobs?
     (trainer side is self-consistent)
  2. Does it reproduce the stored SGLang behaviour logprobs?  (engine side)
  3. Is the padded-batch forward different from the per-row forward on these very
     rows?  -> settles section 2.4 on the real data
  4. Is the mismatch a temperature-semantics bug, an off-by-k alignment bug, or
     flatness-amplified numerical noise?
  5. What is the token structure at the outlier positions (stride, period)?

Nothing is uploaded and no prompt text is printed; only token ids, positions and
aggregate statistics leave the machine.

Run on the node that owns the capture (read-only, one GPU):

  SH=/workspace/storage-shared/nlp/tungks
  SRC=$SH/simct-b200-portable-a1aec0a
  CAPDIR=$SH/borrow8-8MgodXcM/parity-captures-a1aec0a/failure-zusi05xj
  export PYTHONPATH=$SRC/experiments/modal/vendor:$SRC:$SRC/experiments/runai
  CUDA_VISIBLE_DEVICES=0 bash $SRC/experiments/runai/python-b200-host.sh \
    $SRC/experiments/runai/diagnose_parity_capture.py "$CAPDIR" \
    --out /tmp/parity_diag.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))]


def stats(delta: list[float], positions: list[int] | None = None) -> dict[str, Any]:
    if not delta:
        return {"tokens": 0}
    worst = max(range(len(delta)), key=lambda i: delta[i])
    above = [i for i, v in enumerate(delta) if v > 0.5]
    row: dict[str, Any] = {
        "tokens": len(delta),
        "mean": sum(delta) / len(delta),
        "p50": quantile(delta, 0.5),
        "p99": quantile(delta, 0.99),
        "p999": quantile(delta, 0.999),
        "max": delta[worst],
        "worst_index": worst,
        "above_0p1": sum(1 for v in delta if v > 0.1),
        "above_0p5": len(above),
        "above_1p0": sum(1 for v in delta if v > 1.0),
    }
    if positions is not None:
        row["worst_position"] = positions[worst]
        if above:
            shown = above[:300]
            row["above_0p5_index_head"] = shown
            row["above_0p5_position_head"] = [positions[i] for i in shown]
            row["above_0p5_span"] = [above[0], above[-1]]
            if len(above) > 1:
                hist: dict[str, int] = {}
                for a, b in zip(shown, shown[1:]):
                    hist[str(b - a)] = hist.get(str(b - a), 0) + 1
                row["above_0p5_gap_histogram"] = dict(
                    sorted(hist.items(), key=lambda kv: -kv[1])[:12])
                row["above_0p5_span_positions"] = [
                    positions[above[0]], positions[above[-1]]]
    return row


def pearson(x: list[float], y: list[float]) -> float:
    n = min(len(x), len(y))
    if n < 3:
        return float("nan")
    x, y = x[:n], y[:n]
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx and dy else float("nan")


def bands(delta: list[float], band: int) -> list[list[Any]]:
    out = []
    for lo in range(0, len(delta), band):
        seg = delta[lo:lo + band]
        out.append([lo, lo + len(seg), len(seg), sum(seg) / len(seg), max(seg)])
    return out


def stratify(delta: list[float], driver: list[float], edges: tuple[float, ...]) -> list[dict]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = [i for i, v in enumerate(driver) if lo <= v < hi]
        values = [delta[i] for i in idx]
        rows.append({"lo": lo, "hi": hi, "n": len(idx),
                     "mean": (sum(values) / len(values)) if values else None,
                     "max": max(values) if values else None,
                     "above_0p5": sum(1 for v in values if v > 0.5)})
    return rows


def repeat_fraction(ids: list[int], lag: int) -> float:
    if len(ids) <= lag:
        return float("nan")
    return sum(1 for i in range(len(ids) - lag) if ids[i] == ids[i + lag]) / (len(ids) - lag)


def stored_pair_analysis(ids, mask, lengths, behavior_all, trainer_flat, band: int,
                         max_shift: int) -> dict[str, Any]:
    """Structure of the delta that actually tripped the guard, with NO model.

    `batch.pt` carries both sides of the comparison: `stu_behavior_log_probs` (what
    SGLang returned) and `actual_log_probs` (what the trainer computed, masked-flat
    in row-major order). Their difference IS the guard's delta. That means the
    structural questions -- which row, where, contiguous or strided, what period --
    are answerable from the capture alone, with no checkpoint, no GPU and no HF.

    This matters because the capture holds a 6.4 GB student checkpoint and the bug
    note flags a cleanup policy for it: if it is gone, a model-dependent diagnostic
    produces nothing at all, while this path still reports the primary evidence.
    """
    tv: list[float] = []          # stored trainer logprobs, aligned order
    bv: list[float] = []          # stored behaviour logprobs, aligned order
    positions: list[int] = []
    rows: list[int] = []
    cursor = 0
    nan_per_row: dict[str, int] = {}
    for r in range(ids.shape[0]):
        n = int(lengths[r])
        for p in mask[r, :n].nonzero().squeeze(-1).tolist():
            if cursor >= len(trainer_flat):
                break
            t = float(trainer_flat[cursor])
            b = float(behavior_all[r, p])
            cursor += 1
            if not math.isfinite(b):
                # NaN marks the synthetic terminal event on a capped row; it is
                # intentional and carries no behaviour probability. Count it so the
                # number can be checked against the capture's own per_sample stats.
                nan_per_row[str(r)] = nan_per_row.get(str(r), 0) + 1
                continue
            tv.append(t)
            bv.append(b)
            positions.append(p)
            rows.append(r)
    delta = [abs(a - b) for a, b in zip(tv, bv)]
    out: dict[str, Any] = {
        "note": "stored behaviour vs stored trainer logprobs; this is the guard's delta",
        "compared_tokens": len(delta),
        "nan_sentinel_tokens": sum(nan_per_row.values()),
        "nan_sentinel_per_row": nan_per_row,
        "guard_stored_pair": stats(delta, positions),
        "per_row": {},
        "per_band": bands(delta, band),
        "shift_test_stored_pair": {},
        "periodicity": {},
        "outlier_token_id_histogram_top": {},
    }
    for r in range(ids.shape[0]):
        sel = [i for i, rr in enumerate(rows) if rr == r]
        if sel:
            out["per_row"][str(r)] = stats([delta[i] for i in sel],
                                           [positions[i] for i in sel])
    # Alignment test between the two STORED arrays: compare trainer[i] against
    # behaviour[i + k]. If the engine's values were shifted relative to the
    # trainer's, a non-zero shift would beat shift 0 here. (Comparing the deltas to
    # each other, as a first draft did, is meaningless.)
    for k in range(-max_shift, max_shift + 1):
        pairs = [abs(tv[i] - bv[i + k]) for i in range(len(tv)) if 0 <= i + k < len(bv)]
        out["shift_test_stored_pair"][str(k)] = stats(pairs)
    for r in range(ids.shape[0]):
        n = int(lengths[r])
        row_ids = [int(v) for v in ids[r, :n].tolist()]
        out["periodicity"][str(r)] = {
            str(lag): round(repeat_fraction(row_ids, lag), 4)
            for lag in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32)}
    outliers = [i for i, v in enumerate(delta) if v > 0.5]
    if outliers:
        tids = [int(ids[rows[i], positions[i]].item()) for i in outliers]
        hist: dict[str, int] = {}
        for t in tids:
            hist[str(t)] = hist.get(str(t), 0) + 1
        out["outlier_token_id_histogram_top"] = dict(
            sorted(hist.items(), key=lambda kv: -kv[1])[:10])
    return out


def score_row(model, row_ids: list[int], temperature: float) -> dict[str, list[float]]:
    """Per-position logprobs for positions 1..L-1 of one unpadded row."""
    import torch

    t = torch.tensor([row_ids], device="cuda")
    logits = model(input_ids=t, use_cache=False).logits[0]
    labels = torch.tensor(row_ids[1:], device="cuda")
    raw: list[float] = []
    scaled: list[float] = []
    entropy: list[float] = []
    margin: list[float] = []
    p_sampled: list[float] = []
    argmax: list[int] = []
    for lo in range(0, logits.shape[0] - 1, 512):
        hi = min(lo + 512, logits.shape[0] - 1)
        block = logits[lo:hi].float()
        target = labels[lo:hi]
        lp_raw = torch.log_softmax(block, dim=-1)
        raw.extend(lp_raw.gather(-1, target[:, None]).squeeze(-1).tolist())
        argmax.extend(torch.argmax(block, dim=-1).tolist())
        lp_t = torch.log_softmax(block / temperature, dim=-1)
        scaled.extend(lp_t.gather(-1, target[:, None]).squeeze(-1).tolist())
        probs = lp_t.exp()
        entropy.extend((-(probs * lp_t).sum(-1)).tolist())
        top2 = torch.topk(lp_t, 2, dim=-1).values
        margin.extend((top2[:, 0] - top2[:, 1]).tolist())
        p_sampled.extend(probs.gather(-1, target[:, None]).squeeze(-1).tolist())
        del block, lp_raw, lp_t, probs, top2
    del logits, t
    torch.cuda.empty_cache()
    return {"raw": raw, "temperature": scaled, "entropy": entropy, "margin": margin,
            "p_sampled": p_sampled, "argmax": argmax}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    ap.add_argument("--band", type=int, default=256)
    ap.add_argument("--max-shift", type=int, default=3)
    ap.add_argument("--engine", action="store_true",
                    help="also rescore the capture's exact tokens with SGLang under the "
                         "production flags, to separate 'the stored values were corrupted' "
                         "from 'the engine is wrong today'")
    ap.add_argument("--no-model", action="store_true",
                    help="skip every checkpoint-dependent check and report only the "
                         "stored-pair structure (works from batch.pt alone)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    meta = json.loads((args.capture / "metadata.json").read_text())
    batch = torch.load(args.capture / "batch.pt", map_location="cpu", weights_only=True)
    ids = batch["stu_input_ids"]
    attn = batch["stu_attn_mask"]
    mask = batch["stu_loss_mask"].bool()
    behavior_all = batch["stu_behavior_log_probs"]
    temperature = float(batch["temperature"])
    report: dict[str, Any] = {
        "capture": str(args.capture),
        "dtype": args.dtype,
        "temperature": temperature,
        "attention_backend": meta.get("attention_backend"),
        "model_training": meta.get("model_training"),
        "source_commit": meta.get("source_commit"),
        "torch_version": meta.get("torch_version"),
        "per_sample_metadata": meta.get("per_sample"),
        "rows": int(ids.shape[0]),
        "width": int(ids.shape[1]),
    }
    print("PARITY_DIAG_META=" + json.dumps(
        {k: report[k] for k in ("capture", "rows", "width", "temperature",
                                "attention_backend", "source_commit")}, sort_keys=True),
        flush=True)

    lengths = attn.sum(-1).tolist()

    # ---------------------------------------------------------------------
    # Model-free structural analysis, ALWAYS run first.
    #
    # `batch.pt` holds both sides of the guard's comparison, so the structure of
    # the failure is available with no checkpoint, no GPU and no HF. The capture
    # carries a 6.4 GB student checkpoint and the bug note flags a cleanup policy
    # for it; if it is gone, a model-dependent diagnostic reports nothing at all,
    # while this path still reports the primary evidence.
    # ---------------------------------------------------------------------
    trainer_flat = batch["actual_log_probs"].tolist()
    stored = stored_pair_analysis(ids, mask, lengths, behavior_all, trainer_flat,
                                  args.band, args.max_shift)
    report["stored_pair"] = stored
    print("\n== STORED-PAIR structure (no model needed) ==")
    row = stored["guard_stored_pair"]
    print(f"  guard delta from the capture itself: n={row['tokens']} "
          f"mean={row['mean']:.6f} p99={row['p99']:.4f} max={row['max']:.4f} "
          f">0.5={row['above_0p5']}")
    print(f"  NaN terminal sentinels skipped: {stored['nan_sentinel_tokens']} "
          f"per row {json.dumps(stored['nan_sentinel_per_row'])}"
          "   (1 per capped row is expected)")
    if row.get("above_0p5_position_head"):
        pos = row["above_0p5_position_head"]
        print(f"  first outlier positions: {pos[:24]}")
        # stats() already computes the gap histogram over the first 300 outliers.
        print(f"  gap histogram          : "
              f"{json.dumps(row.get('above_0p5_gap_histogram') or {})}")
        mod: dict[str, int] = {}
        for p in pos:
            mod[str(p % 8)] = mod.get(str(p % 8), 0) + 1
        print(f"  mod-8 histogram        : {json.dumps(mod)}")
        print(f"  span                   : {row.get('above_0p5_span_positions')}")
    for r, s in stored["per_row"].items():
        print(f"  row {r}: n={s['tokens']} mean={s['mean']:.6f} p99={s['p99']:.4f} "
              f"max={s['max']:.4f} >0.5={s['above_0p5']}")
    print("  per band (mean/max): " + json.dumps(
        [[b[0], b[1], round(b[3], 5), round(b[4], 4)] for b in stored["per_band"]]))
    print("  shift test (stored pair): " + json.dumps(
        {k: [round(v["mean"], 6), round(v["max"], 4)]
         for k, v in stored["shift_test_stored_pair"].items()}))
    print("  periodicity: " + json.dumps(stored["periodicity"]))
    if stored["outlier_token_id_histogram_top"]:
        print("  outlier token-id histogram (top): "
              + json.dumps(stored["outlier_token_id_histogram_top"]))

    have_model = not args.no_model and (args.capture / "student").is_dir()
    if not have_model:
        reason = ("--no-model was requested" if args.no_model
                  else f"no student checkpoint at {args.capture / 'student'}")
        report["model_skipped"] = reason
        print(f"\nPARITY_DIAG_MODEL=skipped ({reason})")
        print("  The HF cross-checks (temperature detector, entropy/margin, padding,"
              " engine) need the checkpoint and were not run.")
        if args.out:
            args.out.write_text(json.dumps(report, indent=2, sort_keys=True,
                                           default=str) + "\n")
            print("PARITY_DIAG_OUT=" + str(args.out), flush=True)
        print("PARITY_DIAG_DONE=1", flush=True)
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.capture / "student", dtype=getattr(torch, args.dtype),
        attn_implementation="eager", local_files_only=True,
        trust_remote_code=False).cuda()
    model.train(bool(meta.get("model_training")))

    # ---- per-row fresh recomputation (unpadded) -------------------------------
    per_row: list[dict[str, Any]] = []
    for r in range(ids.shape[0]):
        n = int(lengths[r])
        row_ids = [int(v) for v in ids[r, :n].tolist()]
        per_row.append(score_row(model, row_ids, temperature))
    report["per_row_tokens"] = [len(p["raw"]) for p in per_row]

    # The capture stores behaviour and trainer logprobs over loss-masked positions
    # in row-major order; align the fresh scores the same way.
    loss_offsets: list[tuple[int, int]] = []
    flat_behavior: list[float] = []
    flat_trainer: list[float] = []
    flat_fresh_T: list[float] = []
    flat_fresh_raw: list[float] = []
    flat_entropy: list[float] = []
    flat_margin: list[float] = []
    flat_p: list[float] = []
    flat_positions: list[int] = []
    flat_row: list[int] = []
    cursor = 0
    for r in range(ids.shape[0]):
        n = int(lengths[r])
        row_mask = mask[r, :n]
        idx = row_mask.nonzero().squeeze(-1).tolist()
        # `stu_loss_mask` marks *predictor* positions: the guard pairs
        # logits[j] with stu_behavior_log_probs[j], whose label is stu_ids[j + 1].
        # score_row uses the same convention (raw[j] = logprob of row_ids[j + 1]),
        # so the index is j itself -- shifting by one here would manufacture a large,
        # periodic difference on repetitive text.
        for p in idx:
            j = p
            if j < 0 or j >= len(per_row[r]["raw"]):
                continue
            flat_behavior.append(float(behavior_all[r, p]))
            flat_trainer.append(float(batch["actual_log_probs"][cursor]))
            flat_fresh_T.append(per_row[r]["temperature"][j])
            flat_fresh_raw.append(per_row[r]["raw"][j])
            flat_entropy.append(per_row[r]["entropy"][j])
            flat_margin.append(per_row[r]["margin"][j])
            flat_p.append(per_row[r]["p_sampled"][j])
            flat_positions.append(p)
            flat_row.append(r)
            cursor += 1
        loss_offsets.append((r, len(idx)))
    report["loss_masked_tokens"] = cursor
    report["capture_actual_log_probs_length"] = int(batch["actual_log_probs"].numel())
    if cursor != int(batch["actual_log_probs"].numel()):
        report["alignment_warning"] = (
            "fresh masked-token count differs from the stored actual_log_probs length")

    real = [i for i, v in enumerate(flat_behavior) if math.isfinite(v)]
    report["real_tokens"] = len(real)
    report["nan_behavior_tokens"] = len(flat_behavior) - len(real)

    def sub(values: list[float]) -> list[float]:
        return [values[i] for i in real]

    behavior = sub(flat_behavior)
    trainer = sub(flat_trainer)
    fresh_T = sub(flat_fresh_T)
    fresh_raw = sub(flat_fresh_raw)
    entropy = sub(flat_entropy)
    margin = sub(flat_margin)
    p_sampled = sub(flat_p)
    positions = [flat_positions[i] for i in real]
    rows = [flat_row[i] for i in real]

    d_T = [abs(a - b) for a, b in zip(fresh_T, behavior)]

    # ---------------------------------------------------------------------
    # Padded-batch recomputation, replicating `replay_parity.py --layout batch`.
    #
    # On the real capture the two recomputations disagree about which stored array
    # they reproduce, so this has to be measured rather than argued: a per-row
    # forward matches `stu_behavior_log_probs` while a padded-batch forward is what
    # the trainer's own path used. Running BOTH in one process decides which side
    # is anomalous, and whether padding is the mechanism at all.
    # ---------------------------------------------------------------------
    with torch.inference_mode():
        pos_p = attn.long().cumsum(-1) - 1
        pos_p.masked_fill_(attn == 0, 1)
        logits_p = model(input_ids=ids.cuda(), attention_mask=attn.cuda(),
                         position_ids=pos_p.cuda(), use_cache=False).logits
        padded_T_all: list[float] = []
        padded_raw_all: list[float] = []
        for r in range(ids.shape[0]):
            n = int(lengths[r])
            for p in mask[r, :n].nonzero().squeeze(-1).tolist():
                j = p
                if j < 0 or j >= n - 1:
                    padded_T_all.append(float("nan"))
                    padded_raw_all.append(float("nan"))
                    continue
                block = logits_p[r, j].float()
                label = int(ids[r, j + 1].item())
                padded_T_all.append(float(torch.log_softmax(block / temperature, -1)[label]))
                padded_raw_all.append(float(torch.log_softmax(block, -1)[label]))
        del logits_p
        torch.cuda.empty_cache()
    padded_T = sub(padded_T_all)
    padded_raw = sub(padded_raw_all)
    report["padded_vs_row_fresh_T"] = stats([abs(a - b) for a, b in zip(padded_T, fresh_T)])
    report["padded_fresh_T_vs_behavior"] = stats([abs(a - b) for a, b in zip(padded_T, behavior)])
    report["padded_fresh_T_vs_stored_trainer"] = stats(
        [abs(a - b) for a, b in zip(padded_T, trainer)])
    report["padded_fresh_raw_vs_stored_trainer"] = stats(
        [abs(a - b) for a, b in zip(padded_raw, trainer)])
    report["row_fresh_raw_vs_stored_trainer"] = stats(
        [abs(a - b) for a, b in zip(fresh_raw, trainer)])
    report["row_fresh_raw_vs_behavior"] = stats(
        [abs(a - b) for a, b in zip(fresh_raw, behavior)])
    print("\n== which stored array does each recomputation reproduce? ==")
    # Tolerant lookup: the guard comparisons are built further down, and an earlier
    # build of this block indexed them directly and crashed before printing the
    # engine cross-check (the padded numbers had already printed, so the root cause
    # was still visible -- but a crash must not truncate the report).
    for key, alias in (("padded_vs_row_fresh_T", None),
                       ("padded_fresh_T_vs_behavior", None),
                       ("padded_fresh_T_vs_stored_trainer", None),
                       ("row_fresh_T_vs_behavior", "guard_fresh_T_vs_behavior"),
                       ("row_fresh_T_vs_stored_trainer",
                        "trainer_fresh_T_vs_stored_trainer")):
        r_ = report.get(key) or (report.get(alias) if alias else None)
        if not r_:
            print(f"  {key:36s} (not available yet in this build)")
            continue
        report.setdefault(key, r_)
        print(f"  {key:36s} mean={r_['mean']:.6f} p99={r_['p99']:.4f} "
              f"max={r_['max']:.4f} >0.5={r_['above_0p5']}")
    print("  temperature probe on the TRAINER side (raw reference):")
    for key in ("row_fresh_raw_vs_stored_trainer", "padded_fresh_raw_vs_stored_trainer"):
        r_ = report.get(key)
        if not r_:
            continue
        print(f"    {key:34s} mean={r_['mean']:.6f} p99={r_['p99']:.4f} "
              f"max={r_['max']:.4f} >0.5={r_['above_0p5']}")
    padded_ref = report.get("padded_fresh_T_vs_stored_trainer")
    row_ref = report.get("row_fresh_T_vs_stored_trainer") or report.get(
        "trainer_fresh_T_vs_stored_trainer")
    if padded_ref and row_ref:
        close = {"padded": padded_ref["max"], "row": row_ref["max"]}
        report["which_recomputation_matches_the_trainer_array"] = (
            "padded_batch" if close["padded"] < close["row"] / 3 else
            "per_row" if close["row"] < close["padded"] / 3 else "neither_clearly")
        print(f"  => the trainer's stored array is reproduced by: "
              f"{report['which_recomputation_matches_the_trainer_array']}"
              f"  (padded max={close['padded']:.4f}, row max={close['row']:.4f})")
    d_raw = [abs(a - b) for a, b in zip(fresh_raw, behavior)]
    d_trainer = [abs(a - b) for a, b in zip(fresh_T, trainer)]
    d_trainer_raw = [abs(a - b) for a, b in zip(fresh_raw, trainer)]

    print("\n== the guard's own comparison, recomputed fresh ==")
    report["guard_fresh_T_vs_behavior"] = stats(d_T, positions)
    report["guard_fresh_raw_vs_behavior"] = stats(d_raw, positions)
    report["trainer_fresh_T_vs_stored_trainer"] = stats(d_trainer, positions)
    report["trainer_fresh_raw_vs_stored_trainer"] = stats(d_trainer_raw, positions)
    report["stored_trainer_vs_behavior_T"] = stats(
        [abs(a - b) for a, b in zip(trainer, behavior)], positions)
    for key in ("guard_fresh_T_vs_behavior", "guard_fresh_raw_vs_behavior",
                "trainer_fresh_T_vs_stored_trainer", "stored_trainer_vs_behavior_T"):
        row = report[key]
        print(f"  {key:38s} mean={row['mean']:.6f} p99={row['p99']:.4f} "
              f"max={row['max']:.4f} >0.5={row['above_0p5']}")

    print("\n== per row (fresh T vs behaviour) ==")
    report["per_row"] = {}
    for r in range(ids.shape[0]):
        sel = [i for i, rr in enumerate(rows) if rr == r]
        if not sel:
            continue
        row_stats = stats([d_T[i] for i in sel], [positions[i] for i in sel])
        report["per_row"][str(r)] = row_stats
        print(f"  row {r} (len={int(lengths[r])}, masked={len(sel)}): "
              f"mean={row_stats['mean']:.6f} p99={row_stats['p99']:.4f} "
              f"max={row_stats['max']:.4f} >0.5={row_stats['above_0p5']}")

    print("\n== structure of the outliers ==")
    guard = report["guard_fresh_T_vs_behavior"]
    if guard.get("above_0p5_gap_histogram"):
        print("  gap histogram : " + json.dumps(guard["above_0p5_gap_histogram"]))
        head = guard["above_0p5_position_head"]
        print(f"  first positions: {head[:24]}")
        mod: dict[str, int] = {}
        for p in head:
            mod[str(p % 8)] = mod.get(str(p % 8), 0) + 1
        print("  mod-8 histogram: " + json.dumps(mod))
        print(f"  span           : {guard.get('above_0p5_span_positions')}")
    report["per_band"] = bands(d_T, args.band)
    print("  per band (mean/max): " + json.dumps(
        [[b[0], b[1], round(b[3], 5), round(b[4], 4)] for b in report["per_band"]]))

    print("\n== discriminators ==")
    report["error_by_entropy"] = stratify(d_T, entropy, (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9))
    report["error_by_margin"] = stratify(d_T, margin, (0.0, 0.05, 0.25, 1.0, 4.0, 1e9))
    report["corr_delta_1_minus_p"] = pearson(d_T, [1 - v for v in p_sampled])
    report["corr_delta_entropy"] = pearson(d_T, entropy)
    report["corr_delta_abs_logprob"] = pearson(d_T, [abs(v) for v in fresh_T])
    report["mean_p_sampled"] = sum(p_sampled) / len(p_sampled)
    report["mean_entropy"] = sum(entropy) / len(entropy)
    print(f"  corr(|d|, 1-p_sampled) = {report['corr_delta_1_minus_p']:+.4f}")
    print(f"  corr(|d|, entropy)     = {report['corr_delta_entropy']:+.4f}")
    print(f"  corr(|d|, |logprob|)   = {report['corr_delta_abs_logprob']:+.4f}")
    print("  error by entropy: " + json.dumps(report["error_by_entropy"]))
    print("  error by margin : " + json.dumps(report["error_by_margin"]))

    outliers = [i for i, v in enumerate(d_T) if v > 0.5]
    if outliers:
        mt = sum(d_T[i] for i in outliers) / len(outliers)
        mr = sum(d_raw[i] for i in outliers) / len(outliers)
        report["outliers"] = {
            "n": len(outliers), "mean_d_T": mt, "mean_d_raw": mr,
            "raw_explains": mr < 0.25 * mt,
            "positions": [positions[i] for i in outliers][:300],
            "token_ids": None,
        }
        print(f"  at {len(outliers)} outliers: mean d_T={mt:.4f} mean d_raw={mr:.4f} "
              f"-> raw reference {'EXPLAINS' if mr < 0.25 * mt else 'does NOT explain'}")
        # Token ids at the outlier positions: is the text periodic there?
        tids = [int(ids[rows[i], positions[i]].item()) for i in outliers]
        report["outliers"]["token_ids"] = tids
        uniq: dict[str, int] = {}
        for t in tids:
            uniq[str(t)] = uniq.get(str(t), 0) + 1
        report["outliers"]["token_id_histogram_top"] = dict(
            sorted(uniq.items(), key=lambda kv: -kv[1])[:10])
        print("  outlier token-id histogram (top): "
              + json.dumps(report["outliers"]["token_id_histogram_top"]))

    # ---------------------------------------------------------------------
    # Raw-logprob (temperature) mismatch detector.
    #
    # SGLang's standard sampler divides the logits by the request temperature in
    # place and then returns log(softmax(...)). If `sampling_info.temperatures` is
    # 1.0 for a stretch of decode steps, that division is a no-op and the returned
    # logprob is the RAW one -- the September temperature bug, partially. The
    # detector below reads ~1.0 when the engine genuinely returns raw logprobs and
    # ~0 when it scales correctly. Calibrated on Modal/B200 with
    # google/gemma-2-2b-it (8190 response tokens per arm):
    #
    #   SGLANG_RETURN_ORIGINAL_LOGPROB=1  (positive control) -> 0.989
    #   production semantics, four arms    (negative control) -> 0.004 .. 0.037
    #
    # A 26x-250x separation, so a partial stretch shows up clearly.
    # ---------------------------------------------------------------------
    closer_raw = sum(1 for a, b in zip(d_raw, d_T) if a < b)
    frac_closer_raw = closer_raw / len(d_T)
    report["frac_closer_to_raw"] = frac_closer_raw
    report["frac_closer_to_raw_calibration"] = {
        "positive_control_return_original_logprob": 0.9891,
        "negative_control_production_semantics": [0.004, 0.037],
    }
    print("\n== raw-logprob (temperature) mismatch detector ==")
    print(f"  fraction closer to HF(T=1) than to HF(T=rollout) = {frac_closer_raw:.6f} "
          f"({closer_raw}/{len(d_T)})")
    print("  calibrated: production semantics 0.004-0.037 ; RETURN_ORIGINAL_LOGPROB=1 0.989")
    per_row_frac = {}
    for r in range(ids.shape[0]):
        sel = [i for i, rr in enumerate(rows) if rr == r]
        if sel:
            per_row_frac[str(r)] = sum(
                1 for i in sel if d_raw[i] < d_T[i]) / len(sel)
    report["frac_closer_to_raw_per_row"] = per_row_frac
    print("  per row: " + json.dumps({k: round(v, 4) for k, v in per_row_frac.items()}))

    print("\n== alignment test (shift the fresh reference) ==")
    report["shift_test"] = {}
    for k in range(-args.max_shift, args.max_shift + 1):
        pairs = [(fresh_T[i], behavior[i + k]) for i in range(len(fresh_T))
                 if 0 <= i + k < len(behavior)]
        s = stats([abs(a - b) for a, b in pairs])
        report["shift_test"][str(k)] = s
        print(f"  shift {k:+d}: mean={s['mean']:.6f} p99={s['p99']:.4f} max={s['max']:.4f}")

    print("\n== token-stream periodicity per row ==")
    report["periodicity"] = {}
    for r in range(ids.shape[0]):
        n = int(lengths[r])
        row_ids = [int(v) for v in ids[r, :n].tolist()]
        fracs = {str(lag): round(repeat_fraction(row_ids, lag), 4)
                 for lag in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32)}
        report["periodicity"][str(r)] = fracs
        print(f"  row {r}: {json.dumps(fracs)}")

    # ---------------------------------------------------------------------
    # Optional engine cross-check.
    #
    # The analysis above compares the *stored* behaviour logprobs against a fresh
    # eager recomputation. That cannot tell apart two very different worlds:
    #   (a) the engine's logits were wrong when the capture was written, or
    #   (b) the engine is fine and the stored values were corrupted in flight
    #       (an illegal memory access, e.g. the XID 31 seen in this image).
    # Rescoring the capture's EXACT token sequence through the engine now splits
    # them: prefill logprobs are raw on both sides, so if a fresh engine pass
    # matches HF while the stored values do not, the stored values are the anomaly.
    # Opt-in, and failures never abort the rest of the report.
    # ---------------------------------------------------------------------
    if args.engine:
        print("\n== engine cross-check on the capture's exact tokens ==")
        try:
            import sglang

            engine_kwargs = {
                "model_path": str(args.capture / "student"),
                "trust_remote_code": False, "tp_size": 1, "log_level": "warning",
                "attention_backend": "triton", "disable_piecewise_cuda_graph": False,
                "disable_cuda_graph": True, "chunked_prefill_size": 16384,
                "max_prefill_tokens": 16384, "context_length": 8192,
                "mem_fraction_static": 0.25, "skip_server_warmup": True,
            }
            engine = sglang.Engine(**engine_kwargs)
            try:
                full = [[int(v) for v in ids[r, :int(lengths[r])].tolist()]
                        for r in range(ids.shape[0])]
                out = engine.generate(
                    input_ids=full,
                    sampling_params={"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 0},
                    return_logprob=True, logprob_start_len=0)
                out = out if isinstance(out, list) else [out]
                engine_vals: list[list[Any]] = []
                for r, (rec, res) in enumerate(zip(full, out)):
                    entries = res["meta_info"]["input_token_logprobs"]
                    if [int(e[1]) for e in entries] != rec:
                        raise RuntimeError("engine rescore IDs do not match row " + str(r))
                    # SGLang reports None for entries it has no context for (at least
                    # the first token of a row); keep them as None and skip below.
                    engine_vals.append([None if e[0] is None else float(e[0])
                                        for e in entries])
                # SGLang's input_token_logprobs entry k is the logprob of ids[k]
                # given ids[:k], i.e. the value for PREDICTOR position k - 1. The
                # loss mask marks predictor positions p whose label is ids[p + 1],
                # so the matching entry is p + 1 -- not p - 1. Indexing by p - 1
                # here would manufacture a large, structured difference; this is
                # the same off-by-one class that was caught twice already.
                #
                # Align on the UNFILTERED flat arrays (flat_row / flat_positions /
                # flat_fresh_raw / flat_behavior): the NaN terminal sentinel means
                # the filtered arrays are shorter, so a running cursor over mask
                # positions would drift out of step.
                engine_raw: list[float] = []
                hf_raw: list[float] = []
                stored_scaled: list[float] = []
                skipped = 0
                for i, (r, p) in enumerate(zip(flat_row, flat_positions)):
                    if not math.isfinite(flat_behavior[i]):
                        continue
                    if p + 1 >= len(engine_vals[r]) or engine_vals[r][p + 1] is None:
                        skipped += 1
                        continue
                    engine_raw.append(float(engine_vals[r][p + 1]))
                    hf_raw.append(flat_fresh_raw[i])
                    stored_scaled.append(flat_behavior[i])
                report["engine_compared_tokens"] = len(engine_raw)
                report["engine_skipped_tokens"] = skipped
                print(f"  engine comparison covers {len(engine_raw)} tokens "
                      f"({skipped} skipped for missing engine logprobs)")
                if len(engine_raw) != len(fresh_T):
                    report["engine_alignment_warning"] = (
                        f"engine comparison has {len(engine_raw)} tokens, the filtered "
                        f"reference has {len(fresh_T)}")
                report["engine_prefill_raw_vs_hf_raw"] = stats(
                    [abs(a - b) for a, b in zip(engine_raw, hf_raw)])
                report["engine_prefill_raw_vs_stored_scaled"] = stats(
                    [abs(a - b) for a, b in zip(engine_raw, stored_scaled)])
                report["stored_scaled_vs_hf_T"] = stats(
                    [abs(a - b) for a, b in zip(stored_scaled, fresh_T)])
                for key in ("engine_prefill_raw_vs_hf_raw", "stored_scaled_vs_hf_T"):
                    row = report[key]
                    print(f"  {key:34s} n={row['tokens']} mean={row['mean']:.6f} "
                          f"p99={row['p99']:.4f} max={row['max']:.4f} "
                          f">0.5={row['above_0p5']}")
                fresh = report["engine_prefill_raw_vs_hf_raw"]
                stored = report["stored_scaled_vs_hf_T"]
                # Relative, self-calibrating rule. An absolute threshold is wrong here
                # because the engine-vs-HF spread depends on the text: on a pure
                # period-8 loop it reaches 0.79 max on a 512-token synthetic capture,
                # while on coherent text it stays near 0.2. What matters is whether the
                # STORED values are far worse than a fresh engine pass on the same
                # tokens. (An absolute rule was tried first and failed to flag the
                # injected anomaly -- the self-test caught it.)
                # Either statistic firing is enough: the outlier count is the more
                # sensitive of the two (a +1.5 injected band was 90x worse by count
                # but only 1.9x worse by max), and requiring both was tried first and
                # missed it -- the self-test caught that too.
                stored_anomalous = (
                    stored["above_0p5"] > 10 * max(fresh["above_0p5"], 1)
                    or stored["max"] > 3.0 * max(fresh["max"], 0.1)
                )
                verdict = (
                    "STORED values are anomalous vs a fresh engine pass on the SAME "
                    "tokens => the engine is fine today and the stored logprobs were "
                    "the anomaly (in-flight corruption); re-run the case to see whether "
                    "it recurs"
                    if stored_anomalous else
                    "engine and stored values are consistent => read the temperature / "
                    "alignment / flatness discriminators above"
                )
                report["engine_verdict"] = verdict
                report["engine_verdict_inputs"] = {
                    "fresh_engine_max": fresh["max"],
                    "fresh_engine_above_0p5": fresh["above_0p5"],
                    "stored_max": stored["max"],
                    "stored_above_0p5": stored["above_0p5"],
                    "stored_anomalous": stored_anomalous,
                }
                print("  VERDICT: " + verdict)
            finally:
                try:
                    engine.shutdown()
                except Exception:
                    pass
        except BaseException as exc:  # never lose the rest of the report
            report["engine_check_error"] = f"{type(exc).__name__}: {exc}"[:500]
            print("  ENGINE_CHECK_FAILED: " + report["engine_check_error"])

    if args.out:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
        print("\nPARITY_DIAG_OUT=" + str(args.out), flush=True)
    print("PARITY_DIAG_DONE=1", flush=True)


if __name__ == "__main__":
    main()
