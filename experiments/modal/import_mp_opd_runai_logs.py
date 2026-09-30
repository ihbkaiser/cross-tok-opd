"""Validate and backfill completed MP-OPD text logs without rerunning training."""
from __future__ import annotations
import argparse, ast, hashlib, json, math, re, statistics, sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PAIR = re.compile(r"(?:^|, )([A-Za-z_][\w/]*):\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?|nan|inf)(?=,|$)")
BYPASS = re.compile(r"MP_PARITY_BYPASS mean=([\d.e+-]+) max=([\d.e+-]+) tokens=(\d+) above_0p5=(\d+)")

def decode_config(node):
    if isinstance(node, ast.Call):
        return {k.arg: decode_config(k.value) for k in node.keywords}
    if isinstance(node, ast.Name) and node.id == "inf":
        return "inf"
    return ast.literal_eval(node)

def parse_metrics(text):
    rows = []
    for line in ANSI.sub("", text).splitlines():
        if "on_policy_kd_trainer.py:logging:" not in line or "step [" not in line:
            continue
        step = int(re.search(r"step \[(\d+)/", line)[1])
        # Start after progress metadata: timestamps/Elapsed/ETA are not metrics.
        payload = line[line.index("content_length_mean:"):]
        row = {k: float(v) for k, v in PAIR.findall(payload)}
        if not row or any(not math.isfinite(v) for v in row.values()):
            raise ValueError(f"Empty/nonfinite metrics at step {step}")
        row["optimizer_step"] = step
        rows.append(row)
    return rows

def analyze(root, mode):
    dirs = [p for p in (root/mode).iterdir() if p.is_dir()]
    if len(dirs) != 1:
        raise ValueError(f"Ambiguous {mode} run")
    run_dir = dirs[0]
    text = (run_dir/"train.log").read_text()
    summary = json.loads((run_dir/"checkpoint/run-summary.json").read_text())
    rows = parse_metrics(text)
    expected = summary["optimizer_updates"]
    assert summary["status"] == "completed" and expected == 312
    assert (run_dir/"exitcode").read_text().strip() == "0"
    assert [r["optimizer_step"] for r in rows] == list(range(1,expected+1))
    assert all(r["completed_optimizer_updates"] == r["optimizer_step"] for r in rows)
    cfgline = next(l for l in text.splitlines() if l.startswith("AllArguments("))
    config = decode_config(ast.parse(cfgline, mode="eval").body)
    bypass = [dict(mean=float(a), max=float(b), tokens=int(c), above_0p5=int(d))
              for a,b,c,d in BYPASS.findall(text)]
    stats = {}
    for key in rows[0]:
        values = [r[key] for r in rows if key in r]
        stats[key] = dict(first=values[0],last=values[-1],min=min(values),max=max(values),
                          mean=statistics.mean(values),first20=statistics.mean(values[:20]),
                          last20=statistics.mean(values[-20:]))
    valid = sum(r["trajectory_valid_sample_count"] for r in rows)
    derived = dict(
        observed_bypass_lines=len(bypass),
        observed_bypass_max=max((b["max"] for b in bypass),default=0),
        observed_bypass_tokens_above_0p5=sum(b["above_0p5"] for b in bypass),
        valid_samples=int(valid), total_samples=expected*64,
        invalid_samples=int(expected*64-valid),
        total_processed_student_tokens=sum(r["processed_valid_student_tokens"] for r in rows),
        total_step_wall_seconds=sum(r["step_wall_time"] for r in rows),
    )
    derived["weighted_end_to_end_tokens_per_second"] = derived["total_processed_student_tokens"]/derived["total_step_wall_seconds"]
    return dict(name=run_dir.name,run_dir=str(run_dir),config=config,summary=summary,
                rows=rows,bypass=bypass,stats=stats,derived=derived,
                log_sha256=hashlib.sha256((run_dir/"train.log").read_bytes()).hexdigest())

def import_profile(e, profile):
    if profile == "legacy-bypass":
        return dict(bypass=True, group="qwen-gemma-runai-20260908",
                    provenance="collected after training; not an immutable launch snapshot",
                    note="Completed diagnostic MP-OPD run with finite parity outliers bypassed.")
    launch = json.loads((Path(e["run_dir"])/"launch-config.json").read_text())
    opts = launch["options"]
    if launch.get("source_commit") != "f0c218183bb7043afa8e2b07c8313cac2ad1602e" or launch.get("source_dirty") != "":
        raise ValueError("MB4 source does not match audited clean launch commit")
    for key, value in {"micro_train_batch_size": 4, "train_batch_size": 64,
                       "diagnostic_max_updates": 0, "exact_token_trajectory": True,
                       "diagnostic_collapse_gate": True, "use_wandb": False}.items():
        if opts.get(key) != value:
            raise ValueError(f"MB4 launch mismatch: {key}")
    if opts["mp_opd_mode"] != e["config"]["kd"]["mp_opd_mode"] or e["config"]["train"]["micro_train_batch_size"] != 4:
        raise ValueError("launch and logged config disagree")
    if e["bypass"]:
        raise ValueError("Guarded MB4 profile contains bypass messages")
    for row in e["rows"]:
        if not (0 <= row["trajectory_logprob_abs_mean"] <= .1 and 0 <= row["trajectory_logprob_abs_p99"] <= .5):
            raise ValueError("MB4 aggregated parity metrics out of contract")
    return dict(bypass=False, group="qwen-gemma-runai-20260909-mb4",
                provenance="launch-config records clean source commit; log/summary collected after completion",
                launch=launch,
                note="Completed MB4 run with temperature-aware mean/p99 parity guard; no bypass messages observed.")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--upload",action="store_true")
    p.add_argument("--profile",choices=("legacy-bypass","mb4-guarded"),default="legacy-bypass")
    args=p.parse_args()
    evidence=[analyze(args.root,m) for m in ("atomic","fixed")]
    for e in evidence:
        e["import_profile"] = import_profile(e, args.profile)
    (args.root/"analysis.json").write_text(json.dumps(evidence,indent=2))
    for e in evidence:
        print(e["name"],json.dumps(e["derived"]))
    if not args.upload:
        return
    import netrc,os
    auth=netrc.netrc("/mnt/c/Users/Tung/_netrc").authenticators("api.wandb.ai")
    if not auth: raise RuntimeError("W&B credential unavailable")
    os.environ["WANDB_API_KEY"]=auth[2]
    import wandb
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from kdflow.wandb_schema import build_wandb_tags
    entity="kieusontung8-hanoi-university-of-science-and-technology"
    project="vdt-simct-tunix-reproduction"
    api=wandb.Api(timeout=60)
    existing=list(api.runs(f"{entity}/{project}",filters={"displayName":{"$in":[e["name"] for e in evidence]}}))
    if existing:
        raise RuntimeError("Matching run names already exist; inspect before modifying: "+str([r.id for r in existing]))
    receipts=[]
    for e in evidence:
        profile=e["import_profile"]
        mode=e["config"]["kd"]["mp_opd_mode"]
        run_id="mpbackfill-"+hashlib.sha256(e["name"].encode()).hexdigest()[:16]
        tags=build_wandb_tags(method="mp-opd",regime="on-policy",objective="path-credit",
            variant="atomic" if mode=="atomic" else "fixed2",platform="runai",
            accelerator="b200x1",budget="312-update",stage="train",
            student="gemma2-sft-paper",teacher="qwen2.5-7b-instruct",
            extras=(("import","historical-log"),("parity","diagnostic-bypass" if profile["bypass"] else "temperature-mean-p99-guard"))).split(",")
        config={**e["config"],"source_run_name":e["name"],"source_log_sha256":e["log_sha256"],
                "historical_import":True,"original_training_wandb_enabled":False,
                "parity_diagnostic_bypass":profile["bypass"],"provenance_snapshot":profile["provenance"],
                "launch_manifest":profile.get("launch"),
                "resource_telemetry_scope":"node snapshots; concurrent run may be included",
                "metric_aggregation":"trainer averages microbatch metrics, including trajectory_logprob_abs_max",
                "sampling_timestamp_timezone":"unspecified in source log"}
        run=wandb.init(entity=entity,project=project,id=run_id,name=e["name"],
            group=profile["group"],job_type="historical-import",tags=tags,
            config=config,resume="never",dir=str(args.root),
            notes=profile["note"]+" Not evidence of math/code efficacy. Historical metrics imported from rounded text logs.",
            settings=wandb.Settings(x_disable_stats=True,disable_git=True,disable_code=True))
        run.define_metric("train/global_step")
        run.define_metric("train/*",step_metric="train/global_step")
        for row in e["rows"]:
            run.log({"train/global_step":row["optimizer_step"],
                     **{"train/"+k:v for k,v in row.items() if k!="optimizer_step"}},
                     step=row["optimizer_step"])
        run.summary.update({"source_summary":e["summary"],"analysis":e["derived"],
                            "historical_import_rows":len(e["rows"]),
                            "parity_warning_scope":"counts are observed log lines; no per-token IDs or exact per-step attribution"})
        table=wandb.Table(columns=["warning_index","mean","max","tokens","above_0p5"],
                         data=[[i,b["mean"],b["max"],b["tokens"],b["above_0p5"]] for i,b in enumerate(e["bypass"])])
        run.log({"diagnostics/parity_bypass_warnings":table},step=313)
        artifact=wandb.Artifact(run_id+"-evidence",type="training-log")
        for file in ("train.log","exitcode","checkpoint/run-summary.json"):
            artifact.add_file(str(Path(e["run_dir"])/file),name=file)
        artifact.add_file(str(args.root/"analysis.json"),name="analysis.json")
        report=args.root/"report.md"
        if report.is_file(): artifact.add_file(str(report),name="report.md")
        run.log_artifact(artifact)
        url=run.url
        run.finish()
        remote=wandb.Api(timeout=60).run(f"{entity}/{project}/{run_id}")
        imported=list(remote.scan_history(keys=["train/global_step","train/loss","train/content_length_mean"],page_size=1000))
        assert len(imported)==312, len(imported)
        assert [r["train/global_step"] for r in imported]==list(range(1,313))
        for observed, original in zip(imported,e["rows"]):
            assert observed["train/loss"]==original["loss"]
            assert observed["train/content_length_mean"]==original["content_length_mean"]
        receipts.append(dict(name=e["name"],id=run_id,url=url,state=remote.state,verified_rows=len(imported),sha256=e["log_sha256"]))
        (args.root/"upload-receipt.json").write_text(json.dumps(receipts,indent=2))
        print("VERIFIED",json.dumps(receipts[-1]))
if __name__=="__main__":
    main()
