"""Nested 1/4/8-reference FP32 mechanics canary, lhtu05; no downloads or real training."""
import json,os,subprocess,time
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path("/opt/overlay")
IMAGE="docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f"
app=modal.App("mp-opd-multiref-oracle-20260910")
image=(modal.Image.from_registry(IMAGE).entrypoint([])
 .add_local_dir(str(ROOT/"kdflow"),"/opt/overlay/kdflow",ignore=["**/__pycache__/**","**/*.pyc"])
 .add_local_file(str(ROOT/"experiments/mp_opd/real_oracle.py"),"/opt/overlay/experiments/mp_opd/real_oracle.py"))
assets=modal.Volume.from_name("simct-phi-gemma-assets",create_if_missing=False)
outputs=modal.Volume.from_name("mp-opd-fp32-oracle-20260910",create_if_missing=True)
@app.function(image=image,gpu="A100-80GB",cpu=4,memory=49152,timeout=900,retries=0,max_containers=1,volumes={"/assets":assets,"/runs":outputs})
def probe(run_id,commit):
    start=time.time();root=Path("/runs")/run_id;root.mkdir(exist_ok=False)
    groups=[]
    for i,(noun,color) in enumerate([("cup","blue"),("door","red"),("book","green"),("hat","white")]):
        def row(role,place):
            return {"id":f"{i}-{role}-{place}","messages":[{"role":"user","content":f"Write one short sentence about a {color} {noun} on a {place}."}],"reference":f"The {color} {noun} is on the {place}."}
        groups.append({"rollout":row("rollout","desk"),
            "select":[row("select",place) for place in ("shelf","counter","mat","stool","cabinet","windowsill","crate","box")],
            "eval":[row("eval",place) for place in ("chair","table","bench","ledge")]})
    data=root/"mechanics.json";data.write_text(json.dumps({"schema":"mp-oracle-data-v2","groups":groups}))
    env=dict(os.environ,PYTHONPATH="/opt/overlay",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",OMP_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false",PYTHONUNBUFFERED="1")
    env["LD_LIBRARY_PATH"]=":".join(["/usr/local/cuda/lib64","/usr/local/nvidia/lib64"]+[str(p) for p in Path("/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia").glob("*/lib")])
    cmd=["/opt/venvs/simct-b200/bin/python","/opt/overlay/experiments/mp_opd/real_oracle.py","run","--student","/assets/student","--teacher","/assets/teacher","--data",str(data),"--output",str(root/"results"),"--adapter-module","model.layers.25.self_attn.q_proj","--model-dtype","float32","--virtual-lr","0.1","--max-new-tokens","32","--max-reference-tokens","64","--source-commit",commit,"--select-counts","1","4","8"]
    (root/"invocation.json").write_text(json.dumps({"command":cmd,"commit":commit,"image":IMAGE,"scope":"base Gemma/Phi; 4 synthetic groups each 8 select/4 eval; not company SFT"},indent=2))
    try:
        files={}
        for lr in (0.1,):
            run_cmd=list(cmd);run_cmd[run_cmd.index("--virtual-lr")+1]=str(lr)
            dest=root/("lr-"+str(lr));run_cmd[run_cmd.index("--output")+1]=str(dest)
            subprocess.run(run_cmd,env=env,check=True,timeout=max(1,780-(time.time()-start)))
            for p in dest.glob("*.json*"): files["lr-"+str(lr)+"-"+p.name]=p.read_text()
        return {"run_id":run_id,"elapsed_seconds":time.time()-start,"files":files}
    finally:outputs.commit()
@app.local_entrypoint()
def main(run_id:str,commit:str):
    result=probe.remote(run_id,commit)
    out=ROOT/"remote_artifacts/modal-oracle-fp32-20260910"/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/"download.json").write_text(json.dumps(result,indent=2))
    for name,text in result["files"].items(): (out/name).write_text(text)
    print("ORACLE_PROBE_DONE",result["elapsed_seconds"],list(result["files"]))
