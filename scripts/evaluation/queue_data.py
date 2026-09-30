"""Pinned data/trajectory and checkpoint plan for three-seed internal eval."""
import hashlib
import json
from pathlib import Path
import urllib.request
import contract_eval as E

PROFILE="company-internal-v1"
SEEDS=(42,43,44)
PINNED={
 "gsm8k":("openai/gsm8k","740312add88f781978c0658806c59bc2815b9866","main/test-00000-of-00001.parquet","ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"),
 "math500":("HuggingFaceH4/MATH-500","6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be","test.jsonl","35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"),
 "mbpp":("google-research-datasets/mbpp","4bb6404fdc6cacfda99d4ac4205087b89d32030c","full/test-00000-of-00001.parquet","566fd53060ffba5766dace1d1e2f4c38906781526de222b0dfbdbc325b696c77")}
LCB_SHA="4519422b52d7dd243358ee721ecfe94e26ab364a7af2938c8e5d170a17bcadcd"
RUNS={"atomic":"qwen-gemma-mp-atomic-gpu0-limit0-20260909-052816-153554",
      "fixed":"qwen-gemma-mp-fixed-gpu1-limit0-20260909-052816-153555"}


def script_hashes():
    return {n:E.file_hash(E.HERE/n) for n in ("eval_queue.py","context_check.py","queue_data.py","internal_worker.py","contract_worker.py","contract_eval.py","evaluation.py","vendor/lcb_testing_util.py","vendor/provenance.json")}


def tiers():
    return [[("sft",0),("atomic",312),("fixed",312),("atomic",156),("fixed",156)],
            [(mode,step) for step in (80,240) for mode in ("atomic","fixed")],
            [(mode,step) for step in (40,200) for mode in ("atomic","fixed")],
            [(mode,step) for step in (120,280) for mode in ("atomic","fixed")]]


def prompt(benchmark,item):
    if benchmark=="live-code-bench-v6" and item.get("fn_name"):
        system="Write a Python solution. Return only Python code in a code block. Implement the method named "+item["fn_name"]+" in class Solution. Do not read stdin or print the answer; return the result."
        content=system+"\n\n"+item["prompt"]
        if item.get("starter_code"):
            content+="\n\nStarter code:\n"+item["starter_code"]
        return [{"role":"user","content":content}]
    return E.messages(benchmark,item,"author-code","merge")


def normalize_rows(benchmark,rows):
    items=[E.normalize(benchmark,row,"author-code") for row in rows]
    if len(items)!=E.COUNTS[benchmark] or len({x["id"] for x in items})!=len(items):
        raise ValueError("dataset count/IDs mismatch")
    if benchmark=="mbpp" and {x["id"] for x in items}!={str(i) for i in range(11,511)}:
        raise ValueError("MBPP requires original full test IDs 11..510")
    for x in items: x["messages"]=prompt(benchmark,x)
    return items


def acquire(benchmark,cache,company,proxy):
    folder={"gsm8k":"GSM8K","math500":"MATH500","mbpp":"MBPP"}[benchmark]
    candidate=company/folder/"data.jsonl"
    repo,revision,name,sha=PINNED[benchmark]
    target=cache/(benchmark+Path(name).suffix)
    if not target.exists():
        url=f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({"http":proxy,"https":proxy}))
        tmp=target.with_suffix(target.suffix+".part")
        with opener.open(url,timeout=180) as response,tmp.open("wb") as out:
            while chunk:=response.read(1024**2): out.write(chunk)
        if E.file_hash(tmp)!=sha: raise ValueError("download checksum mismatch")
        tmp.rename(target)
    if E.file_hash(target)!=sha: raise ValueError("cached canonical data checksum mismatch")
    if target.suffix==".parquet":
        import pyarrow.parquet as pq
        rows=pq.read_table(target).to_pylist()
    else: rows=[json.loads(x) for x in target.read_text().splitlines() if x.strip()]
    items=normalize_rows(benchmark,rows)
    source={"path":str(target),"sha256":sha,"repo":repo,"revision":revision}
    if candidate.is_file():
        try:
            company_items=normalize_rows(benchmark,[json.loads(x) for x in candidate.read_text().splitlines() if x.strip()])
            # IDs derived locally can differ; compare every prompt, reference and test.
            def content(xs):
                return sorted(E.encoded({k:v for k,v in x.items() if k!="id"}) for x in xs)
            if content(company_items)==content(items):
                source["verified_company_copy"]={"path":str(candidate),"sha256":E.file_hash(candidate)}
                print("COMPANY_CONTENT_MATCH",benchmark,str(candidate),flush=True)
            else: print("COMPANY_CONTENT_DIFFERS",benchmark,"using canonical",flush=True)
        except (KeyError,ValueError,TypeError):
            print("COMPANY_SCHEMA_INCOMPATIBLE",benchmark,"using canonical",flush=True)
    return items,source


def lcb_index(path):
    items=[]; hasher=hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            offset=stream.tell(); line=stream.readline()
            if not line: break
            hasher.update(line)
            if not line.strip(): continue
            row=json.loads(line)
            metadata=row["metadata"]
            if isinstance(metadata,str): metadata=json.loads(metadata)
            item={"id":str(row["question_id"]),"prompt":row["question_content"],
                  "fn_name":metadata.get("func_name"),"starter_code":row.get("starter_code",""),
                  "raw":{"path":str(path),"offset":offset,"length":len(line),"sha256":E.digest(line)}}
            item["messages"]=prompt("live-code-bench-v6",item)
            items.append(item)
    if hasher.hexdigest()!=LCB_SHA or len(items)!=1055 or len({x["id"] for x in items})!=1055:
        raise ValueError("LCB-v6 file differs from verified 1055-row download")
    return items,{"path":str(path),"sha256":LCB_SHA,"revision":"0fe84c3912ea0c4d4a78037083943e8f0c4dd505","release":"release_v6"}


def score_item(benchmark,item):
    if benchmark!="live-code-bench-v6": return item
    raw=item["raw"]
    with open(raw["path"],"rb") as f:
        f.seek(raw["offset"]); line=f.read(raw["length"])
    if E.digest(line)!=raw["sha256"]: raise ValueError("LCB source changed after preparation")
    result=E.normalize(benchmark,json.loads(line),"paper-spec")
    if result["id"]!=item["id"]: raise ValueError("LCB index ID mismatch")
    return result


def prepare(args):
    root=args.out.resolve(); root.mkdir(parents=True,exist_ok=False)
    cache=args.shared/"simct-eval-data/queue-canonical-cache";cache.mkdir(parents=True,exist_ok=True)
    data={}
    for benchmark in E.CAPS:
        if benchmark=="live-code-bench-v6":
            items,source=lcb_index(args.shared/"simct-eval-data/lcb-v6-0fe84c3/problems.jsonl")
        else: items,source=acquire(benchmark,cache,args.company_data,args.proxy)
        path=root/(benchmark+".json")
        E.write_new(path,{"benchmark":benchmark,"items":items,"source":source,"profile":PROFILE})
        data[benchmark]={"path":str(path),"sha256":E.file_hash(path),"count":len(items),"source":source}
    jobs=[]
    sft=args.shared/"SimCT/runs/qwen-gemma-sft-paper-20260908-045828/checkpoint"
    for mode,name in RUNS.items():
        run=args.shared/"simct-runs"/name
        summary=E.read_json(run/"checkpoint/run-summary.json")
        launch=E.read_json(run/"launch-config.json")
        if summary.get("status")!="completed" or summary.get("optimizer_updates")!=312 or Path(str(run)+".exitcode").read_text().strip()!="0":
            raise ValueError("MP run is not complete")
        if Path(summary["student"]).resolve()!=sft.resolve(): raise ValueError("SFT initialization mismatch")
        if launch["options"]["mp_opd_mode"]!=mode or launch["options"]["micro_train_batch_size"]!=4:
            raise ValueError("wrong MP training run")
    for tier,rows in enumerate(tiers()):
        for mode,step in rows:
            path=(args.shared/"SimCT/runs/qwen-gemma-sft-paper-20260908-045828/checkpoint" if mode=="sft"
                  else args.shared/"simct-runs"/RUNS[mode]/"checkpoint"/f"step{step}")
            print("HASH_CHECKPOINT",mode,step,str(path),flush=True)
            identity=E.checkpoint_identity(path)
            jobs.append({"id":f"{mode}-{step}","mode":mode,"step":step,"tier":tier,"checkpoint":identity})
    plan={"schema":"eval-queue-v1","profile":PROFILE,"seeds":list(SEEDS),"data":data,"jobs":jobs,
          "hours":args.hours,"admit_hours":min(18.,args.hours-1),"source":script_hashes(),
          "protocol":{"context_length":8192,"temperature":.6,"top_p":.95,"n":1,"caps":E.CAPS,
             "math":"author helpers, not math-verify","mbpp":"author assertion helper 10 seconds/problem",
             "lcb":"pinned official tester; public+private; functional-specific prompt",
             "code_execution":"explicit internal subprocess, NOT namespace/container isolation",
             "limits":"2 GiB address space, 120s CPU/problem, <=120s wall/problem",
             "scope":"3 evaluation seeds, exploratory checkpoint selection, not exact paper replication"}}
    E.write_new(root/"plan.json",plan)
    print("PLAN_READY="+str(root/"plan.json"),flush=True)
