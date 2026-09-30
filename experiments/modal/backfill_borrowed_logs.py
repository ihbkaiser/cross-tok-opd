"""Import archived borrowed-server logs; never infer missing training metrics."""
import argparse, hashlib, json, math, os, re, tarfile
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP = re.compile(r"epoch \[(\d+)/(\d+)\], step \[(\d+)/(\d+)\]")
METRIC = re.compile(r"(?:^|, )([\w/]+): (-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

def parse(archive):
    records = []
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith("/train.log"):
                continue
            raw = tf.extractfile(member).read()
            log = ANSI.sub("", raw.decode())
            rows = []
            for line in log.splitlines():
                match = STEP.search(line)
                if not match:
                    continue
                epoch, epochs, step, horizon = map(int, match.groups())
                metrics = {k: float(v) for k, v in METRIC.findall(line[match.end():])}
                assert all(math.isfinite(v) for v in metrics.values())
                rows.append(dict(metrics, optimizer_step=step, epoch=epoch))
            assert [r["optimizer_step"] for r in rows] == list(range(1, len(rows)+1))
            completed = "Training completed and model saved." in log
            failed = not completed and "Traceback (most recent call last)" in log
            status = "completed" if completed else "failed" if failed else "incomplete_snapshot"
            records.append(dict(name=Path(member.name).parent.name, rows=rows,
                status=status, log_sha256=hashlib.sha256(raw).hexdigest(),
                archive_member=member.name))
    return records

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    records = parse(a.archive)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    (out / "parsed.json").write_text(json.dumps(records, indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k != "rows"} | {"steps":len(r["rows"])} for r in records]))
    if not a.upload:
        return
    import wandb
    entity = "kieusontung8-hanoi-university-of-science-and-technology"
    project = "vdt-simct-tunix-reproduction"
    api = wandb.Api()
    existing = {r.id:r for r in api.runs(f"{entity}/{project}")}
    receipts=[]
    for item in records:
        name=item["name"]
        run_id="bf-" + hashlib.sha256(name.encode()).hexdigest()[:16]
        if run_id in existing:
            old=existing[run_id]
            if old.config.get("source_log_sha256") != item["log_sha256"]:
                raise RuntimeError("Existing snapshot differs; explicit incremental import required: " + name)
            receipts.append(dict(name=name, url=old.url, skipped=True)); continue
        tags=["backfilled_from_log", "platform:borrowed-b200", "method:" + ("sft" if "-sft-" in name else "simct"), "source-status:"+item["status"]]
        run=wandb.init(entity=entity, project=project, id=run_id, name=name,
            group="qwen-gemma-borrowed-b200", job_type="log-backfill", tags=tags,
            dir=str(out), config=dict(source_log_sha256=item["log_sha256"],
                source_archive_member=item["archive_member"], backfilled_from_log=True,
                metric_precision="original text log; rounded values retained",
                wandb_state_semantics="import job state; consult source_training_status for training status"))
        run.define_metric("optimizer_step")
        run.define_metric("*", step_metric="optimizer_step")
        for row in item["rows"]:
            run.log(row, step=row["optimizer_step"])
        run.summary.update(dict(source_training_status=item["status"],
            observed_optimizer_updates=len(item["rows"]),
            checkpoint_save_confirmed=item["status"]=="completed"))
        url=run.url
        run.finish(exit_code=0)
        verified=wandb.Api().run(f"{entity}/{project}/{run_id}")
        assert verified.summary["observed_optimizer_updates"] == len(item["rows"])
        assert verified.summary["source_training_status"] == item["status"]
        receipts.append(dict(name=name, url=url, steps=len(item["rows"]), status=item["status"]))
        (out/"receipts.json").write_text(json.dumps(receipts,indent=2))
    print(json.dumps(receipts))

if __name__ == "__main__":
    main()
