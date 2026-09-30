"""Append summary-only eval to four verified finished runs; never replay training."""
import argparse, hashlib, json, math
from pathlib import Path
import wandb
PROJECT="kieusontung8-hanoi-university-of-science-and-technology/vdt-simct-tunix-reproduction"
RUNS={
 "sft":("bf-014aaf01c69ff120","qwen-gemma-sft-paper-20260908-045828"),
 "simct":("bf-36d2f25d97568e93","qwen-gemma-simct-2ep-mb4-20260908-093304"),
 "atomic":("mpbackfill-ec7c67c50e92a2e6","qwen-gemma-mp-atomic-gpu0-limit0-20260909-052816-153554"),
 "fixed":("mpbackfill-4ad5d4d6ba1a4b1e","qwen-gemma-mp-fixed-gpu1-limit0-20260909-052816-153555")}
BENCHES=("gsm8k","math500","mbpp","live-code-bench-v6")
def main():
 p=argparse.ArgumentParser();p.add_argument("--summary",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--publish",action="store_true");a=p.parse_args()
 data=json.loads(a.summary.read_text());digest=hashlib.sha256(a.summary.read_bytes()).hexdigest()
 assert data["profile"]=="company-internal-v1" and data["seeds"]==[42,43,44]
 assert len(data["checkpoints"])==25
 for name,v in data["checkpoints"].items():
  assert v["complete"] and set(v["benchmarks"])==set(BENCHES)
  for b in v["benchmarks"].values():
   assert len(b["scores"])==3 and all(math.isfinite(z) and 0<=z<=1 for z in b["scores"])
   assert abs(sum(b["scores"])/3-b["mean"])<1e-12
 api=wandb.Api(timeout=120);a.out.mkdir(parents=True,exist_ok=True)
 receipts=[]
 for mode,(rid,name) in RUNS.items():
  remote=api.run(PROJECT+"/"+rid);assert remote.name==name and remote.state=="finished"
  marker=remote.summary.get("eval/source_sha256")
  if marker:
   assert marker==digest
   remote.tags=list(dict.fromkeys([*remote.tags,"evaluated"]));remote.update()
   receipts.append({"mode":mode,"status":"already-published","url":remote.url});continue
  assert not any(k.startswith("eval/") for k in remote.summary.keys()),"Existing partial eval; inspect before retry"
  before=list(remote.scan_history(page_size=1000))
  (a.out/(mode+"-history-before.json")).write_text(json.dumps(before))
  selected=sorted([(int(n.split("-")[-1]),v) for n,v in data["checkpoints"].items() if n.startswith(mode+"-")])
  assert [s for s,v in selected]==([0] if mode=="sft" else [40,80,120,156,200,240,280,312])
  if not a.publish: print("VERIFIED",mode,rid,len(before));continue
  run=wandb.init(entity=PROJECT.split('/')[0],project=PROJECT.split('/')[1],id=rid,resume="must",dir=str(a.out),settings=wandb.Settings(x_disable_stats=True))
  try:
   run.define_metric("eval/checkpoint_step")
   run.define_metric("eval/curve/*",step_metric="eval/checkpoint_step")
   columns=["checkpoint_step","benchmark","mean","sample_std","seed42","seed43","seed44"]
   table=wandb.Table(columns=columns)
   for step,v in selected:
    row={"eval/checkpoint_step":step,"eval/curve/macro":v["average"]}
    for benchmark,b in v["benchmarks"].items():
     table.add_data(step,benchmark,b["mean"],b["sample_std"],*b["scores"])
     row["eval/curve/"+benchmark]=b["mean"]
    run.log(row)
   run.log({"eval/checkpoints":table})
   endpoint=selected[-1][1]
   for b,v in endpoint["benchmarks"].items():
    run.summary["eval/endpoint/"+b]=v["mean"]
    run.summary["eval/endpoint_std/"+b]=v["sample_std"]
   run.summary["eval/endpoint/macro"]=endpoint["average"]
   run.summary["eval/endpoint_step"]=selected[-1][0]
   best=max(selected,key=lambda z:z[1]["average"])
   run.summary["eval/exploratory_best_step"]=best[0]
   run.summary["eval/exploratory_best_macro"]=best[1]["average"]
   run.summary["eval/profile"]=data["profile"]
   run.summary["eval/seeds"]=data["seeds"]
   run.summary["eval/scope"]="3 evaluation seeds, not training replications; checkpoint selection on test is exploratory; concurrency varied during resume; company-internal-v1, not exact paper replication"
   artifact=wandb.Artifact("eval-"+rid+"-"+digest[:12],type="evaluation",metadata={"summary_sha256":digest,"mode":mode,"run_id":rid,"provenance":"user-supplied aggregate; no per-item journal audit"})
   artifact.add_file(str(a.summary),name="summary-3seeds.json")
   artifact.add(table,"checkpoints")
   run.log_artifact(artifact).wait()
   run.summary["eval/source_sha256"]=digest
  finally: run.finish()
  api.flush();after=api.run(PROJECT+"/"+rid)
  history=list(after.scan_history(page_size=1000))
  assert history[:len(before)]==before,"Training history changed"
  assert after.summary.get("eval/source_sha256")==digest and after.state=="finished"
  assert after.summary["eval/endpoint/macro"]==endpoint["average"]
  after.tags=list(dict.fromkeys([*after.tags,"evaluated"]));after.update()
  receipts.append({"mode":mode,"id":rid,"url":after.url,"training_history_prefix_preserved":True,"rows_before":len(before),"rows_after":len(history),"sha256":digest})
  (a.out/"receipt.json").write_text(json.dumps(receipts,indent=2))
  print("VERIFIED_UPLOAD",mode,after.url,flush=True)
if __name__=="__main__":main()
