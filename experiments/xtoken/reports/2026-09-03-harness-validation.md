# Harness packaging validation — 2026-09-03

This is validation of the versioned **setup harness**, not a rerun of the
historical NVIDIA unit/CUDA experiment.

- Base: SimCT `main`, `cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e`.
- Execution environment: local WSL, Python 3.12.3, launcher uv 0.10.2.
  No ML dependency environment was installed. Workload uv remains pinned to
  0.12.7 and is separately checked by the harness before install/test/smoke.
- `unittest discover -s experiments/xtoken/tests -v`: **36 tests passed**.
- Shell syntax: `bash -n` passed for prepare/install-base/test-base scripts.
- CLI help and source parsing: passed without importing Torch.
- Actual Git fixture: local-origin prepare, detached checkout, exact pin/lock
  verification, idempotent re-prepare, and dirty-checkout rejection passed.
  This uses local fixture repositories, not an upstream network bootstrap.
- Process lifecycle: nonzero exit preservation, no retry, distinct output roots,
  timeout, missing GPU telemetry, Ctrl-C and descendant cleanup passed.
- GPU smoke math: the AST of `sparse_reference`, `bf16_sdpa` and `large_vocab_pkl`
  matched the retained original script. Only checkout/output path handling changed.
- Historical evidence: recorded SHA-256 values match the retained original
  environment, unit-test XML and synthetic-GPU summary files.
- Publication review: new source/docs contain no detected credential patterns
  or known machine access identifiers. This is a scoped static check, **not** an
  exact-value audit of every credential store or of uncommitted external logs.

No rented-host SSH command, GPU workload, authenticated HF action, model
download, Kaggle/Modal operation or infrastructure change was performed for this
packaging work. KDFlow source, SimCT dependency files, and the existing Tunix and
learning branches were not modified. The baseline has no VERSION/versioning
script, PDF, mindmap or milestone/plan artifacts requiring an update here.

Remaining: validate the packaged commands on the rented GPU host, complete the
Automodel worker backend, and conduct a separately authorized end-to-end model
training smoke. Neither these 36 harness tests nor the historical 133 upstream
unit tests establish model-training success or two-GPU performance.
