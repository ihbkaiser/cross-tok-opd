# Nested 1/4/8-reference mechanics probe

Prepare4 groups of13 unique prompts: one rollout,8 select,4 eval. Select pools
are nested prefixes1/4/8, predetermined by a preparation seed, never chosen by
eval. Mean outer NLL equally weights references; each reference is token-mean.
One rollout and teacher credit computation per group, reused across counts.
Frozen adapter and model are unchanged across candidates. Weighting networks
start identically and train independently per count on earlier select groups.
Eval never trains or selects; all count outcomes are reported. Four groups
remain four independent units, not12. The ordinary summary describes the
largest count; paired-select-counts.json and results.jsonl include all counts.
Checkpoint weighting.pt is the largest-count network; weighting-select-N.pt
files retain each count. This is not a resume-complete training checkpoint.

## Company run, only after a new HF bundle pull

From the new isolated source checkout returned by the existing HF pull flow:

```bash
bash "$ABC_WORK/source/experiments/runai/transfer/04-prepare-multiref.sh"
# Explicit GPU boundary; only after the card is free and not reserved by eval:
bash "$ABC_WORK/source/experiments/runai/transfer/05-run-multiref.sh" 1
```

Prepare reuses the pinned selected.parquet or downloads its pinned HF revision
if missing, verifies SHA256, excludes all conflicting-reference prompt groups,
and writes a new exclusive file. This default source potentially appeared in
SFT: results are mechanics only, not unseen-data generalization evidence.
All52 selected prompts are disjoint across roles/groups, but no claim of
semantic deduplication or exclusion from external model pretraining is made.
For a scientific test, prepare a separate authorized train/dev reference source
and use --exclude-prompts for SFT/train/prior probe sets. Never use benchmark
test references. Keep the evaluation reference set fixed across select counts.

Runner uses exact company SFT Gemma and Qwen7B paths, FP32, maxnew64,
virtualLR.1, maxref1024,15-minute timeout. Long references fail rather than
truncate. It refuses the eval queue's GPU lock or >=1GiB occupied VRAM, so do
not stop eval to make room. A timeout preserves partial logs and returns124;
partial groups are not efficacy evidence. No automatic W&B upload.

## Validation

CPU fake-HF end-to-end tests cover legacy and1/4/8 modes, unchanged real
parameters, identical atomic eval NLL across counts, fixed eval IDs, nested
select counts, independent group accounting, overlap rejection, and analytical
gradients of the mean-reference objective. The multi-reference path completed on Modal A100-80GB (lhtu05), app
ap-uR42YdE7fVsJg3tbfZYE5h, source ee28346: four synthetic groups, base
Gemma-2-2b-it / Phi-4-mini, FP32, virtual LR 0.1. Company SFT/Qwen remains
unexecuted. See MP_OPD_OVERNIGHT_20260910.md for results and limitations.
