# Corrected trajectory diagnostic (r8)

This atomic MP-OPD diagnostic follows the r7 generation-collapse audit. It
preserves the pinned Llama-3.2-1B/Qwen3-4B revisions, r7's exact 6,400-row input,
batch 64, seed 43, LR 5e-5, and 100-update cosine schedule. It stops after at
most 30 optimizer updates and saves every ten updates. This is implementation
validation; no task-quality or learned-partition gain is established.

`--exact_token_trajectory True` sends prompt IDs (including tokenizer-defined
special tokens) to SGLang, retains sampled response IDs, and sends the exact
teacher IDs used by alignment to its engine. The teacher prompt and response
are tokenized separately to prevent a token from crossing their boundary.
This defines a specific teacher path; it is not a marginal text probability.
The switch currently supports text-only inputs. Legacy paths remain available
with the switch disabled.

EOS credit remains masked. If a sampled path lacks EOS, a synthetic sentinel
preserves the legacy denominator; it does not represent a sampled stop event.
Raw IDs, SGLang finish reason and output log-probabilities are saved with the
number of completed updates identifying the behavior weights.

Before updating, every scored sampled token must match SGLang's log-prob ID.
The per-sample log-prob absolute error must have mean <=0.1 and maximum <=0.5.
These conservative BF16 agreement thresholds are an implementation gate,
not proof that the two backends are numerically identical. The actual LR and
final accumulated gradient norm are logged separately from legacy metrics.

The predeclared collapse gate stops before updating a batch if two consecutive
batches have median content length zero, empty fraction >=25%, or mean content
length below 10% of the first batch. A batch with no valid atomized samples
also stops before an update. Terminal checkpoints record completed updates;
the triggering rollout has the next rollout index. The training process has a
3,600-second timeout; the B200 function has a 4,200-second outer timeout.

The Modal runner is `experiments/modal/mp_opd_path_modal.py`, with an immutable
qualified registry image and a source overlay checked against a clean local
Git checkpoint. Model downloads run on remote CPU. The source data SHA-256 is
`a0d248ab7c23e09380260a337aad580fabc13fb6d013ce4a9ae20a64f7e9ee9c`.

After the profile's live billing guard passes:

```bash
modal run --profile lhtu05 --detach experiments/modal/mp_opd_path_modal.py::prepare
MP_OPD_ENABLE_B200=1 modal run --profile lhtu05 --detach experiments/modal/mp_opd_path_modal.py::main
```

`main` runs CPU regressions before allocating a B200 and refuses to overwrite
an existing training invocation. Outputs persist on `vdt-mp-opd-path-r8-20260907`;
model and input cache persist on `vdt-mp-opd-path-cache`. A failed run requires
diagnosis and a distinct retry ID; it is never silently restarted.

The workspace must have a payment method accepted by Modal for B200 access.
CPU preparation does not register GPU resources; the explicit environment
switch above registers them only for the training invocation.

## Verification and launch status (2026-09-07)

The CPU regression gate passed on Modal profile `phamvanvuhoan`:
**49 passed, 1 skipped** in 31.11 seconds. App:
`ap-ckLq6qTO18Puau6XnDv3k5`. This includes actual trainer sample construction,
mocked engine transport, shifted-label tests, zero-valid-sample early-stop
control flow, schema checks, and existing MP-OPD regressions. It does not
validate numerical SGLang/HF agreement on a GPU or training stability.

B200 registration was rejected by Modal with
`Please add a payment method to use B200 GPU functions.`
The configured `huggingface-secret` also received HTTP 403 for the pinned
`meta-llama/Llama-3.2-1B` snapshot, explicitly reporting missing gated access.
No training invocation or optimizer update was started. The runtime image
and exact input data are staged; model preparation is incomplete.

Before retrying, enable B200 billing eligibility and update the existing
Hugging Face secret to a token authorized for that model. Do not change model
revision, GPU type, profile, or objective to work around these launch blocks.

The read-only W&B probe using this profile's `wandb-secret` could not retrieve
r7: its API returned `RunNotFoundError`. This does not establish write access
or prove that the run was deleted; verify access to the configured entity and
project before training with this secret.

All preparation/probe apps are stopped. The billing snapshot at
2026-09-07 03:33:40 UTC reports $0.00780266 for this task's app descriptions
(CPU and memory, no B200 usage); reports may lag. Total profile usage is
$0.01169652 for the calendar-month interval 2026-09-01 to 2026-10-01.


## Authorized account continuation

The user confirmed $30 available credit on `lhtu05` and authorized deployment
there. The new run is `mp-opd-path-atomic-r8-lhtu05-20260907`, with output volume
`vdt-mp-opd-path-r8-lhtu05-20260907`. Scientific settings and input hashes are
unchanged. Only the named HF/W&B keys from the user-designated WSL secret
source are provisioned into this profile. Preparation verifies access to the
existing W&B run before downloading pinned models. Local guard budget is $30,
hard limit $28.50, estimate $12 and reserve $1; calendar-month accounting is a
local guard assumption, not verification of Modal's credit reset date.
