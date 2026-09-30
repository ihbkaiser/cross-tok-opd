# Guide campaign update

Default preparation now acquires 1,000 candidate prompts from the eight pinned
sources in `guide-sources.json`, using their original proportions (58% math,
42% code). Acquisition happens on the company machine, using streaming reads;
it does not download these datasets on the developer workstation.

Before generation, exclude full training prompts, selected SFT prompts, and
all evaluation prompts. Dedup uses normalized exact matches, containment, and
5-word shingle Jaccard >=0.8. This is a lexical audit, not proof of semantic
independence. Source shortages stop this branch rather than substitute data.

Qwen generates references on GPU1 in batches of eight. CPU finalization checks
nonempty/EOS, then the existing oracle preparer assigns disjoint rollout,
select, and eval prompts to 32 groups. These are teacher pseudo-labels, not
verified mathematical/code ground truth. Filtering can change source ratios.
Teacher file hashes, source revisions, and exclusion hashes are saved.

GPU0 confirms fixed seed43, generates its endpoint eval, then admits random
seed43 and its endpoint only when the remaining wall budget permits. GPU1
generates guide, runs atomic canary and three oracle probes, a bounded adaptive
pilot, then atomic seed43 and endpoint generation. CPU scoring remains separate.
No promise is made that every optional job fits 20 hours.

The oracle primary LR is 0.1. Promotion requires >=24 valid groups, <=8 invalid,
and positive lower bootstrap bounds plus >=60% positive groups against EACH
atomic, fixed, and skip. Weak evidence selects a random pilot, not a fatal stop.
Technical canary failure blocks that baseline's dependent training.

Remaining limitations: no full learned-partition training launcher, no exact
optimizer/RNG resume for interrupted MP training, no continued-SFT control in
this version. Weighting sensitivity is adapter-only. The guide generation
uses Transformers, not the separate SGLang eval serving pipeline.

Launch through `transfer/start-campaign.sh COMPLETED_EVAL_PLAN`. Outputs include
`guide-acquisition.json`, `guide-teacher.json`, `guide-quality.json`,
`probe-data.json`, `decision.json`, `campaign-state.json`, and `REPORT.md`.
The manager retains its original start clock on resume. Failed jobs are not
blindly restarted. The user should inspect manager and per-job logs.
