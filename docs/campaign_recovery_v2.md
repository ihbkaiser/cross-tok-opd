# Campaign recovery and W&B v2

This workflow audits a previously hash-verified, allowlisted export; it does not run model-generated code or restart remote jobs. The current archive is campaign-results-1789373777667066716, SHA256 8c7889ad1138457157dc7e21ed10d36e53f06af55a4636c07334fac49c459efe. The archive has no model weights or response payloads, so this audit cannot reproduce generation or verify response contents.

## Commands

Run with an existing Python environment containing wandb (the auditor and dashboard use only the standard library):

```sh
python scripts/evaluation/audit_campaign_export.py --root VERIFIED_EVIDENCE --backup WANDB_BEFORE_RUNS_JSON --out ANALYSIS
python scripts/evaluation/publish_campaign_recovery.py --audit ANALYSIS/audit.json --out WANDB_V2
python scripts/evaluation/publish_campaign_recovery.py --audit ANALYSIS/audit.json --out WANDB_V2 --publish
```

The first publisher invocation is read-only on W&B and backs up touched existing runs. Authentication uses the existing WSL-accessible Windows netrc; no credentials enter the artifacts. Project and historical campaign mappings are intentionally explicit in the script, not a general importer.

## Statistical contract

- One W&B run per independent training run. Group names distinguish algorithm and pilot/full-budget variants; soft also carries frozen energy hash. Training seeds remain separate runs.
- `eval_v2/raw/<benchmark>/eval_seed42`, `eval_seed43`, `eval_seed44` describe three decoding repetitions of ONE checkpoint, never three independent training seeds.
- Per-run mean requires all three eval seeds. Across-run mean gives each complete run mean one vote. Across-training sample SD uses ddof=1, is absent for n=1, and is not a confidence interval.
- Interactive HTML supports single training runs and selectable groups of one, two or three training seeds. All three eval seeds remain visible in the per-run tables. Missing checkpoints break lines; missing values are not zeros. Point markers cover single-checkpoint runs.
- Partial checkpoints remain visible but cannot contribute a fabricated complete mean or macro. Macro is an unweighted mean across four benchmarks and requires all 12 cells.
- LCB raw and derived macro scores remain provisional while LCBfix is incomplete. Legacy matched campaign labels identify older seed43 runs whose launch logs are absent; grouping is an organizational aid, not a full configuration-equivalence certification.

## Preservation and recovery

`before/<run-id>/` stores metadata, raw chart config, full history, file listing and existing media. Existing training history is not replayed; old `eval/*` stays intact and new values use `eval_v2/*`. Changing W&B groups does not merge runs. Original groups can be restored from metadata.json using Public API group/update; old charts remain available in their original namespaces and downloaded media. Workspace layout is not rewritten by this importer.

Deterministic run IDs and source digests prevent duplicate imports. Upload receipts are written only after API readback of eval seed scalars and training loss steps. A matching digest without a receipt is reconciled, not assumed verified. Missing uploaded training steps may be appended from source (bounded to two retries), with `train/global_step` as the x-axis. Historical upload wall time is not source training time. W&B Finished describes the import process; consult `recovery_v2/training_status` for source completion.

The auditor checks metric status/count/range, checkpoint/benchmark/seed contract, journal SHA256, unique IDs/count and pass mean. It retains errors and source provenance. Unavailable original training logs and failed evaluation cells require a supplemental company export or targeted remote recovery; this workflow does not assign failure scores to infrastructure errors.

## Native schema alignment (user-requested primary view)

After verified import, run `align_campaign_wandb_schema.py --audit ANALYSIS/audit.json --out WANDB_V2 --publish`.
This adapter uses the same column order as the original publishers:

- `eval/checkpoints`: checkpoint_step, benchmark, mean, sample_std, seed42, seed43, seed44.
- `eval/dispersion_table`: step, benchmark, mean, std, variance, min, max, seed42, seed43, seed44.
- `eval/curve/*` uses `eval/checkpoint_step`; `eval/dispersion/*` uses `eval/dispersion_step`.
- `eval/charts/*_seeds` and `*_spread` use the legacy native chart format.

It skips identical scalar coordinates and rejects conflicting existing values rather than overwriting another protocol. Partial tables retain available seeds and null means. Tags use `kdflow.wandb_schema.build_wandb_tags`, including method:mp-opd and variant:atomic/fixed2/random/soft, with separate train-seed tags. Unknown old launch options are not filled from another run's defaults. The nested train.seed and experiment metadata support native filtering/grouping.

The previous workspace is preserved. Saved view `Campaign native eval groups` (`nw=juzy86xj0yj`) filters campaign_v2.primary=true and groups by W&B Group. Native bands were checked as training-run min/max around Mean, not sample SD or confidence intervals. Exact sample SD and per-step n remain in the audited group table/CSV. Old metadata/media backup remains under before/, and native-before/ records the subsequent schema-alignment snapshot. native-receipts.json and native-verification.json record scalar and downloaded-table readback. No old runs are deleted or merged.

The second saved native view `Campaign native eval seeds 42-43-44` uses `nw=aeg7gccshdq`, retains the same primary filter/grouping, and shows the 15 eval-seed scalar panels (five benchmarks including macro times three decoding seeds). Its section x-axis is explicitly `eval/dispersion_step`; W&B's automatic Step axis was found during screenshot verification and corrected. The mean view retains `eval/checkpoint_step`.

## Supplemental training export

`supplement_campaign_training.py --archive FILE --sha256 HASH --base RECOVERY_ROOT` verifies every archive member against its manifest, rejects unsafe paths and conflicting files, and matches run name, method, training seed and original checkpoint paths before enrichment. Add `--publish` to append only missing `train/*` coordinates to the existing run IDs. All numeric training fields are read back; conflicts are errors. Native eval summary/table references and existing group membership must remain unchanged. Before snapshots and receipts are stored in `extra-training/`; the previous local audit is retained as `analysis/audit-before-extra.json`.

The 2026-09-14 supplemental export (SHA256 `05325a48ab8e67c76357114ae37e3616281fdc18fdfb8cd63060f107f8175c9a`) supplies atomic43 and fixed43 completed at 312 updates, plus the earlier random43 attempt at 244 logged updates without completion evidence. The earlier attempt stays outside the primary random group. This adds training provenance, not new evaluation scores.
