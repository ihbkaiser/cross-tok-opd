# Borrowed two-B200 campaign: eight-hour contract

This is a separate allocation on `tungdd11-sparse-vllm-core-0-0`. The old `tungks-0-0` campaign, its source, processes and deadline are untouched. Shared storage is read for validated prior inputs/checkpoints. Do not rerun the launcher to recover a running campaign; use its recorded queue path.

The start timestamp is recorded once before download/setup and the deadline is exactly start + 8 hours. Setup and idle time count. Preparation refuses more than an hour of setup rather than resetting this clock. If the authorized allocation started earlier, supply that earlier timestamp in the bootstrap's `SIMCT_BORROW_STARTED` variable.

## Fixed dispatch plan

- GPU1: synthetic long-chain numerical qualification + existing energy checkpoint load; five-update soft canary; validate saved canary; fresh soft training, target156, seed42.
- GPU0: evaluate saved random43/fixed43/atomic43 at step200 under the existing four-benchmark/three-evaluation-seed contract; include old soft only if a safely saved checkpoint and same-training-seed baseline are available; then fresh fixed training, target156, seed42.
- Both training jobs share source, runtime, SFT initialization, teacher, data, batch, learning-rate schedule horizon312 and checkpoint cadence20. Energy stays frozen and its SHA is recorded. Canary is separate; training begins from SFT, not the canary checkpoint.
- Training has an absolute cutoff at start+5.5 hours. Before admitting a rollout, trainer reserves300 seconds plus max(120 seconds, 1.5 × previous step duration). It then saves its actual step and records `stopped/deadline_checkpoint_reserve`. A clean partial exit is allowed for evaluation, never labelled full156 completion.
- Evaluate early40 (or first available common step) and last common saved checkpoint, with both GPUs taking jobs from the same evaluation queue. These checkpoints are chosen by a declared step rule, not benchmark scores. Evaluation ends600 seconds before the eight-hour hard deadline; artifact collection gets the final window.
- Partial generation/scoring can remain at cutoff; report cell completeness rather than claiming a full benchmark. Evaluation failures never extend the allocation. A failed numerical/canary validation blocks soft training; unrelated fixed work can still proceed.

## Numerical change and evidence

Semi-Markov alpha/beta/logZ/marginals now use FP64 for the small O(nL) partition DP; model parameters and forward passes are unchanged. Soft training stops on nonfinite coverage or error above1e-6. This prevents silently using a failed partition invariant. It does not guarantee efficacy or remove all rounding from downstream credit pooling.

Local tests cover long-chain coverage, gradients vs marginals, brute-force small partitions, energy double-backward regression, cooperative-stop logic and the real trainer fit method with fake model/Ray boundaries. Real B200 execution is gated by remote numerical qualification and model canary. Synthetic cases are not a replay of the old training trajectories; do not claim the old production discrepancy has been reproduced exactly.

Local CPU comparison on4096 atoms with constant energies3: old commit1dc39895 FP32 coverage error0.4877350926399231; revised FP64 error4.889595395241031e-10. This establishes a long-chain numerical failure of the old DP on that synthetic input, not its precise contribution to old benchmark quality.

## Runtime and operational boundaries

New node must already have `/opt/venvs/simct-b200`, CUDA13/NVRTC headers and existing shared runtime-host libraries. Shared storage does not imply `/opt` exists on the new node. Missing runtime fails preflight without downloading models, installing dependencies or touching any old process. Resolve the node runtime explicitly before retrying preparation with the same clock.

Generic job-manager state is in node-local `/var/tmp/simct-borrow8-*`; its path and host are persisted in `campaign.json` on shared storage. Logs/checkpoints/evaluation outputs use a unique new shared work directory. GPU UUIDs are fixed from the two idle B200s at preparation. Legacy evaluation slot arguments are mapped to those same host indices. Do not reschedule the pod onto another node or run another launcher on these GPUs during the allocation.

The generic worker now tracks observed descendants, including evaluation servers creating another session. PID/start ticks are verified before signals. It cannot guarantee cleanup of instant daemonization that escapes observation; this is not a cgroup sandbox. Benchmark code execution retains the existing explicitly authorized internal evaluation profile.

## Inspect and collect

Read `campaign.json` for `state`, `manager`, `started`, `deadline`. On the new host, run from the unpacked job-manager checkout:

```bash
python3 -m job_manager --state /var/tmp/RECORDED-STATE status
python3 -m job_manager --state /var/tmp/RECORDED-STATE report
python3 -m job_manager --state /var/tmp/RECORDED-STATE events --limit 100
```

`collect` creates a SQLite backup plus job-manager logs and evaluation summaries under shared work. That snapshot is taken while the collect job itself is running; inspect live state to verify its final exit. No model weights are uploaded to GitHub/HF. Source publication contains only committed code and a pinned job-manager source archive.
# Evaluation default

New evaluations use **separated generation/scoring**, with **256 concurrent
generation requests per GPU worker**. This is request concurrency, not a fixed
SGLang batch size. `eval_queue.py worker` defaults to `--phase generate
--concurrency 256 --score-buffer 256`; launch separate `score-spool`
workers for the same plan, on any shared-storage node. Without a scorer only responses are
generated, not final scores. The borrowed campaign submits CPU scoring jobs
alongside generation and waits for both before collecting results. Scoring
backlog does not throttle generation; the generation disk-space guard remains.

Combined mode is an explicit override and requires concurrency at most 64.
Existing running jobs and pinned plans keep their original source/configuration;
do not swap source hashes or start a separated scorer over an active combined
worker. Migrate only after the old workers have stopped and journals are verified.
# Follow-up SimCT and soft diagnostics

`queue_soft_next.py submit` on each node registers the unattended soft50 chain.
Owner: successful soft50 -> verified step50 evaluation plan; owner gen completion
and plan readiness -> two-hour adapter oracle diagnostic; CPU scoring -> LCBfix.
Borrowed: a CPU-only readiness waiter and successful gen2 completion -> soft50
generation at concurrency256. The borrowed deadline remains in force. Existing
jobs are not cancelled. Oracle uses the original guide data on the soft50 student,
not an independent benchmark or direct frozen-energy soft selector comparison.
Submission checks that the original guide exists. No long training is promoted.

`submit_followup.py simct` runs only on the borrowed host and creates a separate
GPU1 manager with no allocation deadline. The full312 seed43 SimCT job uses
`span_ctkd` and the historical `sdpa` backend. `MP_ATTN_IMPLEMENTATION` defaults
to eager and MP-OPD remains restricted to eager for parity qualification.
This is not a claim of byte-identical reproduction of the historical runtime.

`submit_followup.py soft` runs only on the owner host, requires the completed
capture canary, and submits a 50-update seed42 diagnostic with unchanged parity
thresholds and failure capture. Neither command changes running jobs. Both new
jobs retain a 24-hour watchdog; no eight-hour cutoff is inherited.

Existing worker deadlines are captured at startup. Editing queue config or job
JSON cannot extend them, nor can it change a running trainer's environment.
Existing borrowed random and generation workers must not be described as
extended by these commands. Infrastructure allocation is independently controlled
by RunAI.

`audit_lcb_outputs.py --root HISTORICAL_QUEUE_ROOT` reads completed LCB journals,
checks their hashes, and reports truncation, extraction, syntax and outer timeout
flags separately from correctness. It never edits metrics or reruns code.
