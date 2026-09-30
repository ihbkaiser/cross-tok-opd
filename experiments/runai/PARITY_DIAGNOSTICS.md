# Behavior parity failure capture

The September 11 soft canary failed on its fourth rollout iteration with
mean absolute gap 0.053383, p99 0.845674, maximum 4.520025. This is a behavior
policy/trainer mismatch, separate from the FP64 partition DP correction.
Its cause is not established. Keep the existing mean/p99 gate unchanged.

On an isolated single-GPU diagnostic canary, set `MP_PARITY_CAPTURE_DIR` to
a new directory on the internal shared storage. The single-GPU launcher
explicitly propagates this and the source commit into Ray workers.
On parity failure, the algorithm saves exact student token IDs, attention and
loss masks, behavior logprobs, gathered trainer logprobs and a student weights
snapshot, then raises the original failure. No optimizer update is performed
by the capture. The snapshot is for forward replay, not resumable training.
Multi-rank capture is refused to avoid a collective save deadlock.

Artifacts contain internal data and model weights. Do not publish them to
GitHub, Hugging Face, W&B or chat. Share only metadata and numerical reports.
No capture is made unless explicitly enabled. Model saving adds time/disk use
only after failure; leave enough allocation time for the snapshot.

After `metadata.json` records `checkpoint_complete: true`, expose one idle GPU
and use the existing `python-b200-host.sh` wrapper to run
`experiments/runai/replay_parity.py CAPTURE_DIR`. Repeat with `--layout unpadded`
and `--dtype float32` separately. These are diagnostic forwards, not training.
Compare each result against both recorded behavior and captured trainer.
Differences isolate batch layout and precision effects; they do not by
themselves establish a SGLang or synchronization bug. Further SGLang replay
and weight-version checks may be required.

Do not retry the long soft run until the cause is understood and a fresh
five-update canary passes. Do not modify the source checkout used by running
random/atomic/evaluation jobs; deploy a separate checkout for diagnostics.
