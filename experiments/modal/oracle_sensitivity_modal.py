"""One A10, no retries, <=1800s GPU function; authorized $5 envelope."""
import json,os,subprocess
from pathlib import Path
import modal
IMAGE="docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f"
app=modal.App("mp-opd-oracle-sensitivity-20260910")
root=Path(__file__).resolve().parents[2] if modal.is_local() else Path("/opt/overlay")
image=modal.Image.from_registry(IMAGE).entrypoint([]).add_local_file(str(root/"experiments/modal/oracle_sensitivity_worker.py"),"/opt/probe.py")
assets=modal.Volume.from_name("mp-opd-gemma-parity-assets-v1",create_if_missing=False)
outputs=modal.Volume.from_name("mp-opd-gemma-parity-results-v1",create_if_missing=False)
@app.function(image=image,gpu="A10",cpu=4,memory=32768,timeout=1800,retries=0,max_containers=1,volumes={"/assets":assets,"/runs":outputs})
def probe(run_id,commit):
    ready=json.loads(Path("/assets/ready.json").read_text())
    assert ready["revision"]=="299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
    target=Path("/runs")/run_id
    if target.exists():raise ValueError("run exists; refusing duplicate")
    env=dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="4")
    libs=["/usr/local/cuda/lib64","/usr/local/nvidia/lib64"]
    libs.extend(str(p) for p in Path("/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia").glob("*/lib"))
    env["LD_LIBRARY_PATH"]=":".join(libs)+":"+env.get("LD_LIBRARY_PATH","")
    try:
        subprocess.run(["/opt/venvs/simct-b200/bin/python","/opt/probe.py","--output",str(target)],env=env,check=True,timeout=1650)
        result=json.loads((target/"result.json").read_text());result.update(source_commit=commit,image=IMAGE)
        (target/"result.json").write_text(json.dumps(result,indent=2));return result
    finally:outputs.commit()
@app.local_entrypoint()
def main(run_id:str,commit:str):
    result=probe.remote(run_id,commit)
    print("RESULT_JSON="+json.dumps(result))
    out=root/"remote_artifacts/modal-oracle-sensitivity-20260910/result.json"
    out.write_text(json.dumps(result,indent=2))
