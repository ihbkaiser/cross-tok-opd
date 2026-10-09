# Run logs

Independent orphan history for reviewed experiment logs and metrics.
Mirrors: sontungkieu/SimCT and ihbkaiser/cross-tok-opd, branch log.
Code snapshot when initialized: 0a2b556aa012e667ef801d57caef6157b5733650 (vdt/ops/b200-portable).

Initial evidence: Phi/Gemma parity probe r1 and paired SimCT canary r2.
R1 ended with a harness import error after saving its measurements; r2 completed.
See PHI_GEMMA_DEBUG.md for provenance and limits. Each run has a SHA-256 manifest.
Only reviewed logs/JSON/text are archived. No credentials, weights, datasets or
optimizer state belong here. Historical runs outside this snapshot are not yet archived.

Code checkout command: scripts/ops/sync_github_branches.ps1 synchronizes both refs
from local to both GitHub repositories and verifies identical commit hashes.
