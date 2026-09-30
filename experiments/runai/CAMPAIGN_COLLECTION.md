# Recovering evidence after pod loss

Run `collect_campaign.py` with the system Python on any pod mounting the shared
storage. No queue database, GPU runtime, credentials or original hostname is
required. Default roots cover the borrowed campaign, the historical trial256
evaluation and SimCT/runs. Missing roots are reported. Other run roots can be
specified by repeating `--root`; this replaces the defaults.

First inventory, then export:

```bash
python3 collect_campaign.py --out /workspace/storage-shared/nlp/tungks/borrow8-8MgodXcM/evidence-exports --responses
python3 collect_campaign.py --out /workspace/storage-shared/nlp/tungks/borrow8-8MgodXcM/evidence-exports --responses --export
```

The archive includes logs, JSON control/evaluation metadata, score journals and
(with --responses) responses. It excludes source trees, caches, environments,
W&B directories, credentials by filename, symlinks, tokenizer.json, vocab.json
and model weights. Small tokenizer configuration files remain included. Weight
filenames and sizes are inventoried, not validated. Nothing is restarted,
marked completed, retried or assigned a zero score.

Outputs contain original data and are private. Filename filtering is not a
secret scrubber: review/redact log and config contents before external upload.
Keep the tar.gz and its SHA256 sidecar together. Each archived file has a digest
in manifest.json. Concurrent changes and unreadable files are reported; this is
an evidence copy, not a transactional snapshot of a live campaign. Collector
memory use is bounded by the largest selected file. Run on one shared-storage
pod, not both, to avoid duplicate archives. Do not put the output under an
unexcluded input directory.

Before W&B backfill, validate the archive, reconcile run identities with existing
W&B runs, retain interrupted source status, and distinguish training seeds from
evaluation seeds. Existing importers that require completed 312-step runs must
not be used for partial runs. A successful import does not prove the source run
completed. Raw responses require separate review before external publication.
