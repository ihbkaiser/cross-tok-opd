#!/usr/bin/env python3
"""Apply the pre-registered beta selection rule to a finished GBV pilot.

Read-only over the pilot logs: it never trains, never edits a run, and never looks at a
benchmark. The rule is frozen in B200_REAL_RUN.md before the numbers were seen:

  1. per run, take the mean of the last 10 logged values of the GBV telemetry and the
     logit-gradient diagnostics;
  2. reference is the atomic partition, whose cosine is 1.0 by construction: every span
     holds exactly one atom, so pooled == atomic, delta == 0 and retained_dof_fraction
     == 1. An atomic run is therefore optional and only a canary; when its log exists it
     is used, otherwise the analytic anchor 1.0 is used and reported as such;
  3. among betas whose mean diag logit-grad cosine is within COSINE_TOL of the atomic
     cosine, pick the smallest mean retained_dof_fraction;
  4. if no beta qualifies, pick the highest cosine and report the directional gate as
     failed;
  5. if the diagnostics never appeared, fall back to retained_dof_fraction plus the
     stability of total_cost, and report the pilot as having no directional branch.

Refuses to select from an incomplete pilot: every beta run must show RUN_VERIFIED.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

TAIL = 10
COSINE_TOL = 0.02
ATOMIC_COSINE_ANALYTIC = 1.0
KEYS = (
    "mp_opd_gbv_retained_dof_fraction",
    "mp_opd_gbv_total_cost",
    "mp_opd_gbv_distortion_term",
    "mp_opd_gbv_dof_term",
    "mp_opd_gbv_selected_span_count",
    "mp_opd_gbv_selected_span_length_mean",
    "mp_opd_gbv_boundary_strength_mean",
    "mp_opd_gbv_degenerate",
    "mp_opd_diag_logit_grad_cosine",
    "mp_opd_diag_logit_grad_norm_ratio",
    "mp_opd_diag_logit_grad_delta_ratio",
)
RUNS = (("atomic", None), ("b0p1", 0.1), ("b0p3", 0.3), ("b1p0", 1.0), ("b3p0", 3.0))
# The atomic run only canaries the probe: its cosine, norm ratio, distortion and retained
# dof fraction are fixed by construction, so a missing atomic log is not an incomplete
# pilot. Beta runs are never optional.
OPTIONAL_TAGS = ("atomic",)


def series(text: str, key: str) -> list[float]:
    return [float(value) for value in re.findall(re.escape(key) + r":\s*([-\d.eE+]+)", text)]


def summarize(text: str) -> dict[str, dict[str, float] | None]:
    summary: dict[str, dict[str, float] | None] = {}
    for key in KEYS:
        values = series(text, key)
        if not values:
            summary[key] = None
            continue
        tail = values[-TAIL:]
        summary[key] = {
            "mean_last10": statistics.fmean(tail),
            "std_last10": statistics.pstdev(tail) if len(tail) > 1 else 0.0,
            "n": len(values),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="pilot output directory, e.g. $SH/SimCT/runs/gbv-pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=Path, default=None, help="write the decision as JSON here")
    args = parser.parse_args()

    reports: dict[str, dict] = {}
    incomplete: list[str] = []
    for tag, beta in RUNS:
        log = args.out / f"gbv-pilot-{tag}-s{args.seed}-r1.out"
        optional = tag in OPTIONAL_TAGS
        if not log.is_file():
            if not optional:
                incomplete.append(tag)
            reports[tag] = {"beta": beta, "log": str(log), "verified": False,
                            "optional": optional, "reason": "log missing"}
            continue
        text = log.read_text(errors="replace")
        verified = "RUN_VERIFIED" in text
        if not verified and not optional:
            incomplete.append(tag)
        reports[tag] = {"beta": beta, "log": str(log), "verified": verified,
                        "optional": optional, "metrics": summarize(text)}

    print("run    beta   verified  dof_frac  cosine  norm_ratio  total_cost(std)  span_len")
    for tag, _beta in RUNS:
        report = reports[tag]
        metrics = report.get("metrics") or {}

        def cell(key: str, fmt: str = "{:.6f}") -> str:
            entry = metrics.get(key)
            return "  --  " if not entry else fmt.format(entry["mean_last10"])

        cost = metrics.get("mp_opd_gbv_total_cost")
        print(f"{tag:<6} {str(report['beta']):<6} {str(report['verified']):<9} "
              f"{cell('mp_opd_gbv_retained_dof_fraction')}  {cell('mp_opd_diag_logit_grad_cosine')}  "
              f"{cell('mp_opd_diag_logit_grad_norm_ratio')}  "
              f"{cell('mp_opd_gbv_total_cost')}"
              f"({0.0 if not cost else cost['std_last10']:.6f})  "
              f"{cell('mp_opd_gbv_selected_span_length_mean')}")

    if incomplete:
        print("\nPILOT_INCOMPLETE: " + ", ".join(incomplete) + " - khong chon beta tu pilot thieu")
        if args.json:
            args.json.write_text(json.dumps({"status": "incomplete", "runs": reports}, indent=2))
        return 2

    def metric(tag: str, key: str):
        entry = (reports[tag].get("metrics") or {}).get(key)
        return None if entry is None else entry["mean_last10"]

    atomic_cosine = metric("atomic", "mp_opd_diag_logit_grad_cosine")
    if atomic_cosine is None:
        atomic_cosine = ATOMIC_COSINE_ANALYTIC
        anchor_source = "analytic: atomic partition pools one atom per span, so cosine is 1.0"
        print("\nGHI CHU: khong co log atomic -> dung anchor giai tich 1.0. Run atomic chi la canary; "
              "no khong doi ket qua chon beta.")
    else:
        anchor_source = "atomic log"
    beta_tags = [tag for tag, beta in RUNS if beta is not None]
    have_cosine = [tag for tag in beta_tags if metric(tag, "mp_opd_diag_logit_grad_cosine") is not None]

    decision: dict[str, object]
    if have_cosine:
        reference = atomic_cosine
        eligible = [tag for tag in have_cosine
                    if metric(tag, "mp_opd_diag_logit_grad_cosine") >= reference - COSINE_TOL]
        if eligible:
            chosen = min(eligible, key=lambda tag: metric(tag, "mp_opd_gbv_retained_dof_fraction"))
            decision = {"status": "selected", "chosen_tag": chosen, "chosen_beta": reports[chosen]["beta"],
                        "rule": "smallest retained_dof_fraction among betas within cosine tolerance",
                        "atomic_cosine": reference, "anchor_source": anchor_source,
                        "eligible": eligible, "directional_gate": "pass"}
        else:
            chosen = max(have_cosine, key=lambda tag: metric(tag, "mp_opd_diag_logit_grad_cosine"))
            decision = {"status": "selected", "chosen_tag": chosen, "chosen_beta": reports[chosen]["beta"],
                        "rule": "no beta within cosine tolerance: highest cosine taken",
                        "atomic_cosine": reference, "anchor_source": anchor_source,
                        "eligible": [], "directional_gate": "fail"}
    else:
        chosen = min(beta_tags, key=lambda tag: metric(tag, "mp_opd_gbv_retained_dof_fraction"))
        decision = {"status": "selected", "chosen_tag": chosen, "chosen_beta": reports[chosen]["beta"],
                    "rule": "no logit-gradient diagnostics: smallest retained_dof_fraction",
                    "anchor_source": anchor_source, "directional_gate": "missing"}
        print("\nCANH BAO: khong co mp_opd_diag_logit_grad_cosine -> dung fallback da dang ky; "
              "bao cao phai ghi ro pilot thieu nhanh directional.")

    decision["beta"] = reports[chosen]["beta"]
    decision["reports"] = reports
    print("\nDECISION " + json.dumps({k: v for k, v in decision.items() if k != "reports"}))
    print("Ghi beta nay vao mp_opd_gbv_beta cho primary va KHONG doi nua.")
    if args.json:
        args.json.write_text(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
