# Company evaluation queue (three evaluation seeds)

This queue evaluates exactly one SFT initialization and the two existing MB4,
312-update trajectories. It does not train new models. Atomic/fixed snapshots:
40, 80, 120, 156, 200, 240, 280, 312; SFT is evaluated once as a snapshot.
Each snapshot runs seeds 42, 43, 44 over GSM8K, MATH500, original MBPP and LCB v6:
17 snapshots, 51 snapshot-seed combinations, 204 benchmark cells at most.

## Contract and interpretation

Profile `company-internal-v1` is explicit opt-in. It retains author generation
settings (temperature .6, top_p .95, one sample per request, benchmark token
caps), author GSM8K/MATH/MBPP helpers, and the pinned official LCB tester with
public and private tests. Functional LCB tasks receive the function name and
starter code in a dedicated Solution-method prompt. Math uses the author's
SymPy helpers, not math-verify. The three evaluation seeds measure sampling
variance, not independent training variance. Intermediate test-checkpoint
comparisons are exploratory; do not report the best test checkpoint as an
unbiased paper reproduction. Existing author-code/paper-spec sandbox profiles
are unchanged.

The company-internal subprocess executes generated Python under the current
user, with process time/address-space limits and a clean environment. **These
limits are not filesystem/network/container isolation.** This mode implements
the user's accepted internal-eval workflow for the namespace-restricted node.
Do not describe its timeout/security contract as identical to the paper.

Canonical fallback files total approximately 1 MB and are pinned by HF commit
and SHA256. Existing cached canonical files are reused. Company JSON copies
are compared against canonical normalized contents; incompatible copies are
not silently substituted. LCB reuses the already verified 1055-problem file,
indexes it without loading all hidden tests, and verifies each row before use.
No models or training data are downloaded. No dependencies are auto-installed.
The scorer needs numpy and sympy; parquet preparation needs pyarrow, available
in the existing datasets runtime if its installation is complete.

## Company commands

Pull `pull-eval-queue.sh` from the published, pinned HF revision and verify its
SHA256 using the accompanying chat commands. Running the pull script creates
an isolated checkout and prints EVAL_WORK. It never changes SimCT-git.

```bash
EVAL_WORK=/workspace/storage-shared/nlp/tungks/simct-eval-queue-REPLACE
bash "$EVAL_WORK/source/experiments/runai/transfer/prepare-eval-queue.sh"
# The following command starts GPU workers; run only when ready.
bash "$EVAL_WORK/source/experiments/runai/transfer/start-eval-queue.sh"
source "$EVAL_WORK/queue.env"
/usr/bin/python3.12 "$EVAL_WORK/source/scripts/evaluation/eval_queue.py" summarize --plan "$EVAL_PLAN"
```

Preparation qualifies ten synthetic correct/incorrect scoring fixtures, checks
the two completed training summaries, and hashes every chosen checkpoint.
Hashing roughly 17 model snapshots may take several minutes on shared storage.
It does not consume the GPU budget. Missing snapshots stop preparation rather
than changing the selection silently.

## Budget, ordering, resume

The first worker completing CPU qualification starts one shared 20-hour wall
clock. New snapshots stop being admitted at 18 hours (or earlier if the slowest
observed snapshot would not fit). A server watchdog ends the worker-owned
server at the deadline; process cleanup can take a few seconds beyond it.
Unfinished cells remain partial. There is no guarantee all 204 cells fit.
Both workers take snapshots independently, with barriers between these tiers:

1. SFT, atomic312, fixed312, atomic156, fixed156.
2. Atomic/fixed80 and240.
3. Atomic/fixed40 and200.
4. Atomic/fixed120 and280.

Every selected snapshot finishes all three seeds before its worker moves on.
GPU memory >=1 GiB causes waiting; existing training/oracle jobs are never
killed. Per-GPU and per-job file locks prevent duplicate queue work. Resource
ownership outside this queue still requires the operator to avoid launching
another workload onto an occupied GPU.

Workers append and fsync per-question responses and scores. Resume checks
source, checkpoint, data, request, response and server identities. Running
start-eval-queue.sh again resumes the same plan and clock; it does not grant
another 20 hours. Active duplicate workers fail their GPU lock. A failed job
halts admissions: inspect worker/server logs, fix the cause, then use
`eval_queue.py retry --plan "$EVAL_PLAN"` and start the workers again. Source or
runtime changes that break the recorded contract require a new plan and must
not be merged into old results. Do not delete state.json to extend the clock.

Summary includes a benchmark mean and sample standard deviation only when all
three seeds are complete and hashes match. A checkpoint average requires all
four benchmarks; partial results are not zero-filled. Logs and result files
stay under the plan directory. Nothing is uploaded to W&B automatically.

## Verification boundary

Offline regression tests cover selection, tier barriers, deadline admission,
resume journals, cached responses, source-row integrity, and ten actual CPU
scorer fixtures including functional LCB. SGLang GPU startup and end-to-end
throughput still need verification on the company B200 node.

## Mechanics tokenizer fix included

The same source bundle fixes `real_oracle.py` prompt tokenization: explicitly
request `return_dict=False`, normalize mapping/Encoding outputs into integer
IDs, and create torch.long input tensors. The old failure happened before the
first group result; it is not oracle evidence. Reuse the existing prepared
six-row mechanics JSON in the fresh bundle workspace, then run the copied
03-run-mechanics.sh on GPU1 before starting the evaluation workers. GPU1's
eval worker will wait while mechanics owns memory. GPU0 can evaluate meanwhile.
The tokenizer regression uses CPU mocks; the corrected real-model canary has
not yet run on the company node.

## Gemma 2 context correction

The launcher now uses the native 8192-token context. Before GPU startup, the
existing runtime tokenizer checks every exact chat prompt plus its full output
cap. Overflow fails explicitly: no hidden truncation, omitted questions or
long-context override. The earlier 16384 launcher setting was incorrect.
`recover-startup --from-plan OLD --out NEW` supports only failed startup queues
with no results, reuses verified prepared data, and preserves the original
start/admission/deadline timestamps. It does not renew the 20-hour budget.
