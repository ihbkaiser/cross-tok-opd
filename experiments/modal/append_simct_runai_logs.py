"""Append verified missing SimCT steps to an existing historical W&B run."""
import argparse
import hashlib
import json
import math
import netrc
import os
from pathlib import Path
from import_mp_opd_runai_logs import parse_metrics

def verify_prefix(source, history):
    if not history:
        raise ValueError("Existing run has no history")
    steps=[int(row["optimizer_step"]) for row in history]
    if steps != list(range(1,len(history)+1)):
        raise ValueError("Existing history is not a unique contiguous prefix")
    if len(history)>len(source):
        raise ValueError("Remote history exceeds source")
    for original, remote in zip(source, history):
        for key,value in original.items():
            if key not in remote or not math.isclose(float(remote[key]),value,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError(f"Prefix mismatch: step={original['optimizer_step']} key={key}")
    return len(history)

def canonical_readback(history):
    """Collapse only byte-equivalent API rows; conflicting duplicates are fatal."""
    by_step={}
    for row in history:
        step=int(row["optimizer_step"])
        if step in by_step and by_step[step] != row:
            raise ValueError(f"Conflicting duplicate API row at step {step}")
        by_step[step]=row
    return [by_step[k] for k in sorted(by_step)]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--upload",action="store_true")
    args=p.parse_args()
    log=next(args.root.rglob("train.log"))
    source=parse_metrics(log.read_text())
    summary=json.loads((log.parent/"checkpoint/run-summary.json").read_text())
    assert summary["status"]=="completed" and summary["optimizer_updates"]==312
    assert [r["optimizer_step"] for r in source]==list(range(1,313))
    snapshot=json.loads((args.root/"wandb-before.json").read_text())
    count=verify_prefix(source,snapshot["history"])
    print("LOCAL_PREFIX_VERIFIED",count,"MISSING",len(source)-count,flush=True)
    if not args.upload:return
    os.environ["WANDB_API_KEY"]=netrc.netrc("/mnt/c/Users/Tung/_netrc").authenticators("api.wandb.ai")[2]
    import wandb
    entity="kieusontung8-hanoi-university-of-science-and-technology"
    project="vdt-simct-tunix-reproduction"
    run_id="bf-36d2f25d97568e93"
    api=wandb.Api(timeout=60)
    remote=api.run(f"{entity}/{project}/{run_id}")
    assert remote.name=="qwen-gemma-simct-2ep-mb4-20260908-093304"
    current=list(remote.scan_history(page_size=1000))
    count=verify_prefix(source,current)
    before=dict(remote.config)
    (args.root/"wandb-preappend.json").write_text(json.dumps({"history":current,"summary":dict(remote.summary),"config":before},default=str))
    run=wandb.init(entity=entity,project=project,id=run_id,resume="must",dir=str(args.root),
                  settings=wandb.Settings(x_disable_stats=True,disable_git=True,disable_code=True))
    assert run.resumed
    run.config.update({
        "previous_snapshot_log_sha256":before.get("source_log_sha256"),
        "source_log_sha256":hashlib.sha256(log.read_bytes()).hexdigest(),
        "source_training_status":"completed",
        "final_log_append_start_step":count+1 if count<len(source) else None,
        "historical_time_fields_note":"Old ETA/Elapsed fields parsed only hour components; do not use them as elapsed seconds.",
    },allow_val_change=True)
    for row in source[count:]:
        step=row["optimizer_step"]
        run.log({**row,"epoch":1 if step<=156 else 2},step=step)
    run.summary.update({
        "source_training_status":"completed","observed_optimizer_updates":312,
        "source_summary":summary,"checkpoint_save_confirmed":True,
        "checkpoint_verification_scope":"training log and run-summary; weights not included in archive",
        "historical_import_rows":312,
    })
    artifact=wandb.Artifact("simct-2ep-final-log-bf-36d2f25d97568e93",type="training-log")
    artifact.add_file(str(log),name="train.log")
    artifact.add_file(str(log.parent/"checkpoint/run-summary.json"),name="run-summary.json")
    run.log_artifact(artifact)
    run.finish()
    final=wandb.Api(timeout=60).run(f"{entity}/{project}/{run_id}")
    history=list(final.scan_history(page_size=1000))
    assert verify_prefix(source,canonical_readback(history))==312
    assert final.summary["source_training_status"]=="completed"
    receipt={"id":run_id,"url":final.url,"state":final.state,"appended_rows":312-count,
             "verified_rows":312,"api_rows":len(history),"exact_duplicate_api_rows":len(history)-312,
             "verified_all_source_scalars":True,"source_log_sha256":hashlib.sha256(log.read_bytes()).hexdigest()}
    (args.root/"upload-receipt.json").write_text(json.dumps(receipt,indent=2))
    print("VERIFIED",json.dumps(receipt),flush=True)

if __name__=="__main__":
    main()
