# MP-OPD overnight probe report — 2026-09-10

## Scope and current result

User-authorized envelope: $5 total, <=30 minutes GPU, use lhtu05 then
kieusontung8. Earlier phamvanvuhoan A10 run preceded that profile preference.
Company eval remains independent; no company GPU launch or W&B upload.

The numerical BF16 probe and four-group FP32 oracle probe completed. They
show that numerical dead zones can hide virtual updates, but they do not yet
establish a useful learned-partition advantage over atomic credit.

## Completed probes

1. Base Gemma A10: BF16 LR .01 nonzero gradient/update but zero logit/NLL delta.
FP32 same mechanism has NLL decrease 1.287e-5, near Taylor1.308e-5. One synthetic
sentence, same-example sensitivity check, not teacher distillation evidence.
See ORACLE_SENSITIVITY_20260910.md.

2. lhtu05 A100-80GB app ap-vaLkmIpgpeW8PN8SnRmsdS: FP32 base Gemma2-2B/Phi4-mini,
4 disjoint synthetic rollout/select/eval groups, maxnew32, rank4 last-layer
q_proj adapter, virtualLR .1. Exit0, 4 valid groups, no real model mutation.
Worker66.58sec; app verified stopped, zero tasks. Reported Modal cost $0.06181488
(GPU .05022728 + CPU .00387269 + RAM .00771491), not including the prior A10.
This model pair differs from the company SFT Gemma/Qwen pair. No unseen-data
or downstream efficacy claim is justified.

All four groups now have nonzero evaluation changes and the meta-SFT select
sensitivity control changes. Hard oracle is the best actual select loss among
atomic/fixed2/fixed4/random/oracle in all4 at this LR, but wins heldout vs
atomic in only2/4. Thus the zero-signal mechanics problem is distinguishable
from the select-to-eval generalization problem.

## Paired heldout differences

Positive mean means oracle has lower heldout reference NLL than the control.
These are natural-log loss differences, not accuracy or percentage points.
Four synthetic groups and their bootstrap intervals are exploratory only.

| Control | Mean control NLL minus oracle NLL | Fraction oracle wins |
|---|---:|---:|
| atomic | 7.62939453e-06 | 0.50 |
| fixed2 | 4.99486923e-05 | 1.00 |
| fixed4 | 7.18832016e-05 | 1.00 |
| oracle | 0 | 0.00 |
| random_oracle_lengths | 3.95774841e-05 | 1.00 |
| skip | 0.000120162964 | 1.00 |
| atomic_half_lr | 6.22272491e-05 | 1.00 |
| atomic_quarter_lr | 9.25064087e-05 | 1.00 |
| atomic_oracle_norm | 5.00679016e-06 | 0.50 |
| meta_sft | -0.000571131706 | 0.00 |
| learned_weighting | 6.91413879e-06 | 0.50 |

## Implications and proposed corrections

- The earlier all-zero company mechanics summary must not be treated as a
scientific fail of MP-OPD. First qualify actual virtual-update resolution on
that exact SFT checkpoint, using FP32 or verified nonzero forward deltas.
- Do not raise the training learning rate based on a diagnostic LR sweep.
- Credit conservation alone does not imply gradient-norm/direction preservation.
New diagnostic telemetry includes virtual-update norm and gradient/rate delta
from atomic; random different partitions may still implement the same update.
- A hard first-order DP oracle is optimal for its additive Taylor utility,
not necessarily for nonlinear finite-step improvement or a soft mixture.
- Learned weighting only starts adapting across four groups; this is not a
trained, capacity/data-exposure matched competitor. Meta-SFT directly uses
references and must remain an explicit data-exposure control.
- Next mechanics probe: convex atomic/oracle update mixtures, choose alpha on
actual select loss only; report heldout results without selecting on them.
It is a diagnostic baseline, not evidence of a novel successful algorithm.

## Reproducibility

FP32 probe source ae89acffc5ff0af5d7d61000167646964446560b; actual source/model hashes
in manifest.json. Runtime digest fixed in launcher. Data authored in runner;
all raw trajectories and results retained under
remote_artifacts/modal-oracle-fp32-20260910. Source was committed and pushed.
User company eval source remains separate; no HF bundle was republished for
these Modal-only probes.

Independent Astra xhigh audit is in progress in task
01a08768-34ff-7771-9be0-3fec4b69a5aa; its report and CPU counterexamples will be
available under D:/dev/codex/research_vdt/deliverables/MP_OPD_ASTRA_REVIEW_20260910.md.

## Finite-step mixture follow-up

Completed app ap-F4FEGjML8TzPocZq3lDJ5T on lhtu05 A10080GB, exit0,
worker75.99sec. Same4 groups, frozen model, FP32, LR .1 and10; emitted trajectory
JSONL bytes are identical across LRs. For every group alpha in {0,.25,.5,.75,1}
is selected by actual select NLL. Alpha1 (oracle) was chosen in7/8 cases; alpha.75 in one LR.1 group.
That mixture ties oracle on select NLL (5.8953714371), but its eval NLL
is higher by 5.2452e-6 (5.7566766739 versus5.7566714287); selection ties favor
the earlier candidate.
Thus this finite-step mixture-selection control did not improve the oracle.
At LR10, oracle minus atomic advantage on heldout is mean .0007603, wins3/4;
vs norm-matched atomic wins2/4, exploratory interval crosses0. Meta-SFT wins
against oracle4/4. These observations do not support an efficacy claim.
They also do not disprove soft mixtures in other curvature/noise regimes.

Atomic/fixed/skip/LR and norm-matched controls were retained. Additional
candidate forward passes have a cost; mixture selection is not compute-matched
to the original atomic baseline. VirtualLR10 is not a training recommendation.

## Billing and stop verification

Reported costs by app: {"ap-vaLkmIpgpeW8PN8SnRmsdS": 0.06181488, "ap-F4FEGjML8TzPocZq3lDJ5T": 0.06317115000000001, "ap-WuCQV7CJwbmWsEfpkt3w1S": 0.01459349}.

Total reported: $0.13957952 of authorized $5; billing snapshots retained.
Both lhtu05 apps verified stopped with zero tasks. No further GPU job planned in this overnight pass.

## Local test boundary

Targeted real-diagnostic suite: 11 passed. Broader CPU-compatible MP-OPD suite:
37 passed, 1 skipped (optional real-tokenizer fixture/import). Full directory
collection was attempted but blocked by missing local transformers in
 test_energy_and_integration.py; no dependency was installed. This is not a
claim that the complete integration suite passed. The real-model FP32 and
mixture probes did execute successfully in the existing remote runtime.


## Completed nested-reference canary (02:01, September 10)

App ap-uR42YdE7fVsJg3tbfZYE5h on lhtu05 A100-80GB completed at source
ee28346. Worker 174.85 seconds; app lifetime 190 seconds; verified stopped,
zero tasks. Cached base Gemma-2-2b-it and Phi-4-mini, FP32, virtual LR 0.1,
four synthetic groups, one shared rollout per group, nested 1/4/8 select
references and four fixed eval references. These closely related templates
are mechanics evidence, not independent natural-task generalization.

| Select references | Mean atomic minus oracle eval NLL | Positive groups |
| --- | ---: | ---: |
| 1 | 0.000014424324 | 4/4 |
| 4 | 0.000014781952 | 4/4 |
| 8 | 0.000014901161 | 4/4 |

Increasing 1 to 8 gives only 0.000000476837 mean gain; group 3 improves more
with 4 than 8. Oracle partitions change in two groups, remain identical in two.
At eight references, oracle beats norm-matched atomic in only 2/4 groups;
mean 0.000009655952 and exploratory interval crosses zero. Meta-SFT beats
oracle in all four groups, with unmatched update norms. This does not establish
a partition advantage or a reliable multi-reference improvement. Do not compare
the 4/4 rate with the previous single-reference probe as a treatment effect:
this probe changes the evaluation reference set and synthetic examples.

Raw assertions passed: unchanged real parameters, identical atomic eval NLL
across counts, fixed eval IDs. Four groups remain four units, not twelve.
CPU-compatible suite: 39 passed, 1 skipped; complete integration collection
not run locally because transformers is absent. Actual cached-model GPU path
completed all four groups. Exact company SFT/Qwen execution remains pending.

Billing snapshot: this app USD 0.15460920; refreshed total for three lhtu05
apps plus the prior A10 app USD 0.29760436. Billing can still settle.
Raw outputs and ledger are retained under remote_artifacts/modal-oracle-fp32-20260910/.

Next meaningful experiment uses the exact SFT with separate train/dev reference
prompts, stronger contextual signed weighting and norm-matched meta-SFT controls.
The new company launcher refuses occupied or eval-reserved GPUs and times out
after 15 minutes. No company evaluation was interrupted.
