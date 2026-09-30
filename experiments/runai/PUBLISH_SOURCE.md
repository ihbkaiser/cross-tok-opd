# Publish GitHub and Hugging Face together

Run from the existing WSL repository, after committing the intended changes:

```bash
/home/tung/Collaborative-MORL/.venv/bin/python experiments/runai/publish_source.py
# With HF_TOKEN supplied by the existing secret environment:
/home/tung/Collaborative-MORL/.venv/bin/python experiments/runai/publish_source.py --publish
```

Default is a local check only. --publish checks the exact authorized branch and
GitHub push URL, rejects tracked edits and shallow history, bundles committed
source only, pushes GitHub without force, and verifies its ref. It preserves
existing HF manifest asset metadata, publishes the complete bundle and updated
manifest in one HF commit guarded by parent revision, then reads the immutable
HF files back and verifies GitHub again. DUAL_PUBLISH_PASS means both match.
Untracked files, credentials, datasets, checkpoints and W&B outputs are not
added to Git; untracked remote_artifacts remains local.

Targets: sontungkieu/SimCT branch vdt/ops/b200-portable and HF model repository
codemaivanngu/simct. No remote is created or changed. No dependency installation,
GPU launch or company checkout update occurs. Company update-simct.sh continues
to fetch the published bundle using its existing fast-forward checks.

The two services cannot be committed atomically. A failure after GitHub push
leaves a receipt under remote_artifacts/publish-*/receipt.json with partial
status. Inspect it before retrying, especially hf_submission=pending. If the
same source/bundle is already on HF, a retry verifies it without a new HF commit.
An HF parent conflict stops publication rather than overwriting another update.
Historical standalone helper filenames remain pinned to their original versions;
this command synchronizes canonical Git source through the bundle, not every
historical helper alias. No git push --mirror or force push is used.
