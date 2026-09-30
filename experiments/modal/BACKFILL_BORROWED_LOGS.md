# Borrowed-server log backfill

`backfill_borrowed_logs.py ARCHIVE --output OUTPUT` parses only train.log archive members without extracting arbitrary paths. Add `--upload` to publish to the existing vdt-simct-tunix-reproduction W&B project using WANDB_API_KEY from the authorized environment.

Stable run IDs are derived from original directory names. Identical imports are skipped; changed snapshots refuse automatic overwrite pending an incremental-import implementation. Histories preserve logged precision, including rounded learning rates. Missing metrics are not synthesized. Use optimizer_step as chart x-axis.

W&B job state describes the import, not the original training. The summary source_training_status distinguishes completed, failed, and incomplete_snapshot. No raw logs, credentials, model files, or benchmark claims are uploaded. Imported timestamps are W&B ingestion times rather than original training wall times.

The 20260908-113841 archive contains seven runs: SFT canary 5 updates, full SFT 272, SimCT OPD 100, full OPD snapshot 140, two failed startup canaries with zero updates, and a successful OPD canary with five. Parsing checks contiguous unique optimizer steps and finite numerical values.
