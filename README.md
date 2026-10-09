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

## User-supplied archives added 2026-10-10

- `runs/chunk-r101c-20261009/`: GRASS-Chunk r101c log, launch config and TensorBoard; original ZIP retained.
- `runs/mp-opd-newmath-grass-trustr-20261009/`: GRASS new-math and TRUST-R logs/TensorBoard; original TGZ retained.
Each directory contains `manifest.json` with source and per-file SHA-256. Archives were validated, paths checked and payloads scanned for credential patterns before commit. These are supplied run artifacts; importing does not validate scientific correctness or performance.
