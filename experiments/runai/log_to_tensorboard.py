"""Rebuild TensorBoard event files from a run's stdout log.

A run only writes events when it was launched with TensorBoard enabled, which is
opt-in, so every run finished before that flag existed - including the campaign
runs this campaign is compared against - has its curve sitting in plain text. The
trainer prints one line per optimizer update with every scalar as ``key: value``,
so the curve can be recovered without re-running anything and without touching
the GPU.

Parsing is deliberately strict about the record boundary. ``optimizer_updates
[N/TOTAL]`` is the only thing that advances the step, and metrics are read from
the same line, so a metric can never be attached to a neighbouring step. A line
without that marker contributes nothing rather than being guessed at.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# ``key: value`` where the value is a plain number. Anchored on both ends so a
# value that is really part of a longer token is skipped instead of truncated.
_METRIC = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*):\s*(-?\d[\d.eE+-]*)\s*$")
_STEP = re.compile(r"optimizer_updates\s+\[(\d+)/(\d+)\]")


def parse_log(text: str) -> Tuple[List[Tuple[int, Dict[str, float]]], int]:
    """Return ``([(step, metrics), ...], declared_total)``.

    Only the first occurrence of a key within a step is kept: the trainer can
    repeat a metric inside one line, and a later duplicate is the same number
    rather than a second reading.
    """
    records: List[Tuple[int, Dict[str, float]]] = []
    total = 0
    for line in text.splitlines():
        step_match = _STEP.search(line)
        if step_match is None:
            continue
        step = int(step_match.group(1))
        total = int(step_match.group(2))
        metrics: Dict[str, float] = {}
        for part in line.split(","):
            metric = _METRIC.match(part)
            if metric is None:
                continue
            try:
                value = float(metric.group(2))
            except ValueError:
                continue
            metrics.setdefault(metric.group(1), value)
        if metrics:
            records.append((step, metrics))
    return records, total


def select_keys(
    records: Iterable[Tuple[int, Dict[str, float]]], wanted: Iterable[str]
) -> List[str]:
    """Keys present in every record, in the requested order first.

    TensorBoard draws one line per series, so a key that only appears on some
    steps would leave gaps. Only keys complete across the whole run are kept,
    which is what makes the resulting curve readable.
    """
    records = list(records)
    if not records:
        return []
    common = set(records[0][1])
    for _, metrics in records[1:]:
        common &= set(metrics)
    ordered = [k for k in wanted if k in common]
    ordered += sorted(common - set(ordered))
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="run .out log, or a directory containing one")
    parser.add_argument("--out", type=Path, required=True, help="TensorBoard event directory to write")
    parser.add_argument("--tag-prefix", default="", help="prefix for every series name")
    parser.add_argument(
        "--all-keys",
        action="store_true",
        help="write every complete key, not just the headline ones",
    )
    args = parser.parse_args()

    source = args.log
    if source.is_dir():
        candidates = sorted(source.glob("*.out"))
        if not candidates:
            raise SystemExit("no .out log found in %s" % source)
        source = candidates[-1]
    if not source.is_file():
        raise SystemExit("log not found: %s" % source)

    records, total = parse_log(source.read_text(encoding="utf-8", errors="replace"))
    if not records:
        raise SystemExit("no 'optimizer_updates [n/total]' record found in %s" % source)

    headline = [
        "loss",
        "grad_norm",
        "learning_rate",
        "mp_opd_dpca_ppo_kl",
        "mp_opd_dpca_clipfrac",
        "mp_opd_dpca_clipfrac_lower",
        "mp_opd_dpca_ratio_mean",
        "mp_opd_dpca_advantage_mean",
        "mp_opd_dpca_advantage_abs_max",
        "mp_opd_dpca_advantage_clamped_frac",
        "mp_opd_dpca_atoms",
        "trajectory_logprob_closer_to_raw_fraction",
        "empty_response_fraction",
        "collapse_bad_streak",
    ]
    keys = select_keys(records, headline if not args.all_keys else [])

    from torch.utils.tensorboard import SummaryWriter

    args.out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.out))
    for step, metrics in records:
        for key in keys:
            writer.add_scalar(args.tag_prefix + key, metrics[key], step)
    writer.close()

    first, last = records[0][0], records[-1][0]
    print("SOURCE=%s" % source)
    print("SERIES=%d RECORDS=%d STEP_RANGE=%d..%d DECLARED_TOTAL=%d" % (len(keys), len(records), first, last, total))
    print("OUT=%s" % args.out)
    print("KEYS=" + ",".join(keys))
    missing = total - last if total else 0
    if missing:
        print("INCOMPLETE_MISSING_STEPS=%d" % missing)


if __name__ == "__main__":
    main()
