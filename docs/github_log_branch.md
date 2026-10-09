# Code and log synchronization

Code lives on `vdt/ops/b200-portable`. Curated run evidence lives on `log`,
an orphan branch with no shared ancestor with code. Both branches are mirrored
with identical commit IDs to `sontungkieu/SimCT` and `ihbkaiser/cross-tok-opd`.

Run from Windows PowerShell after committing the intended code/log changes:

```powershell
./scripts/ops/sync_github_branches.ps1 -DryRun
./scripts/ops/sync_github_branches.ps1
```

The command pushes exact local commit snapshots, uses atomic updates within each
repository, rejects non-fast-forward updates, and verifies both remote refs.
Two repositories cannot be updated as one transaction: if the second fails, the
command reports failure and can be rerun after resolving the problem. It never
force-pushes, changes the default branch, or uploads all local branches.

Use a separate checkout to add evidence without switching the active code tree:

```powershell
git worktree add D:/dev/codex/simct-log log
```

In that checkout, add only reviewed logs/metrics under `runs/<run-id>/`, update
SHA-256 manifests, and commit. Keep credentials, model weights, optimizer states,
datasets and unrelated raw artifacts out of the branch. Then run the sync command
from the code checkout. The parent workspace rule against staging raw
`remote_artifacts/` still applies to the code branch; the user's explicit request
authorizes reviewed evidence on the separate log branch.
