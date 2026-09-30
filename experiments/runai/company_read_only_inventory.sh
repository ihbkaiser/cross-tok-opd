#!/usr/bin/env bash
# Read-only exact asset/campaign inventory. No copy/upload/queue mutation.
set -euo pipefail
if (( $# != 4 )); then
  echo "usage: $0 CASE BASE REPO PY" >&2
  exit 2
fi
CASE=$1
BASE=$2
REPO=$3
PY=$4
EXPECTED_ENERGY_SHA=c7fe0fff9d9f46bb64962d4d0baccd119c6f4fd60a232f05aa5ea026113817b6
sha256() { sha256sum "$1" | awk '{print $1}'; }
emit_file() {
  local p="$1"
  if [[ -f "$p" ]]; then
    echo "FILE_PRESENT path=$p bytes=$(stat -c %s "$p") sha256=$(sha256 "$p")"
  elif [[ -d "$p" ]]; then
    echo "DIR_PRESENT path=$p"
  else
    echo "MISSING path=$p"
  fi
}
echo "READ_ONLY_META_ENERGY_INVENTORY v1"
echo "CASE=$CASE"
echo "REPO=$REPO"
echo "HEAD=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
echo "EXPECTED_ENERGY_SHA256=$EXPECTED_ENERGY_SHA"
shopt -s nullglob
for p in "$CASE/campaign.json" "$CASE/eval-template.json" "$CASE/queue-plan.json" "$CASE/queue-receipt.json" "$CASE/data-audit.json" "$CASE"/nodes/*/receipt.json "$CASE"/nodes/*/campaign.json "$CASE"/dispatch/*.json; do
  emit_file "$p"
done
for p in "$BASE/SimCT/data/qwen-author/data/prompts.parquet" "$BASE/SimCT/data/qwen-author/data/selected.parquet" "$BASE/owner-tungks-0-1/energy-select-4.pt" "$BASE/partition-qualify-fix-IcuIufCf/work/energy/energy-select-4.pt" "/workspace/storage-shared/models/Qwen2.5-7B-Instruct" "$BASE/SimCT/runs/qwen-gemma-sft-paper-20260908-045828/checkpoint"; do
  emit_file "$p"
done
"$PY" - "$CASE" "$BASE" "$EXPECTED_ENERGY_SHA" "$REPO" <<'PY'
import hashlib, json, sys
from pathlib import Path
case, base, expected, repo = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
failed = False
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def show_json(p):
    global failed
    if not p.is_file(): print("JSON_MISSING",p); return
    try: d=json.loads(p.read_text())
    except Exception as e: print("JSON_INVALID",p,type(e).__name__); failed=True; return
    print("JSON_META",p,"keys="+",".join(sorted(d)[:80]))
    for k in ("source","commit","meta_policy","student","teacher","dataset","meta","energy","energy_sha256","partition_dp_dtype"):
        if k in d: print("JSON_FIELD",p,k,repr(d[k])[:500])
    if isinstance(d.get("input_hashes"),dict):
        for k in sorted(d["input_hashes"]): print("INPUT_HASH",k,d["input_hashes"][k])
for p in (case/"campaign.json",case/"eval-template.json",case/"queue-plan.json",case/"queue-receipt.json",case/"data-audit.json"):
    show_json(p)
for p in sorted(case.glob("nodes/*/receipt.json"))+sorted(case.glob("nodes/*/campaign.json")):
    show_json(p)
b=base/"SimCT/data/qwen-author/data/prompts.parquet"; m=base/"SimCT/data/qwen-author/data/selected.parquet"
energy=base/"owner-tungks-0-1/energy-select-4.pt"
if not energy.is_file(): energy=base/"partition-qualify-fix-IcuIufCf/work/energy/energy-select-4.pt"
for p in (b,m,energy):
    if p.is_file(): print("ASSET_HASH",p,digest(p))
    else: print("ASSET_MISSING",p)
try:
    import pyarrow.parquet as pq
except Exception as e:
    print("META_CONTRACT_UNAVAILABLE",type(e).__name__); failed=True
else:
    if not b.is_file() or not m.is_file():
        print("META_CONTRACT_BLOCKED missing B or M parquet"); failed=True
    else:
        bt,mt=pq.read_table(b),pq.read_table(m); B,M=bt.to_pylist(),mt.to_pylist()
        print("PARQUET_SCHEMA B",bt.schema.names,"rows",len(B)); print("PARQUET_SCHEMA M",mt.schema.names,"rows",len(M))
        if "messages" not in bt.schema.names or not {"messages","label"}.issubset(mt.schema.names): failed=True
        def text(r):
            v=r.get("messages")
            if isinstance(v,str):
                try:v=json.loads(v)
                except Exception:pass
            return "\\n".join(str(x.get("content","")) for x in v if isinstance(x,dict) and x.get("role")=="user") if isinstance(v,list) else str(v or "")
        import re,unicodedata
        norm=lambda x:re.sub(r"\\s+"," ",unicodedata.normalize("NFKC",x)).strip().casefold()
        bids={norm(text(r)) for r in B}; mids={norm(text(r)) for r in M}
        missing=sum(not isinstance(r.get("label"),str) or not r["label"].strip() for r in M); outside=len(mids-bids)
        print("META_RAW_AUDIT B_rows",len(B),"M_rows",len(M),"missing_label",missing,"M_outside_B",outside)
        if len(B)!=10000 or missing or outside: failed=True
        student=base/"SimCT/runs/qwen-gemma-sft-paper-20260908-045828/checkpoint"
        try:
            sys.path.insert(0,str(repo))
            from transformers import AutoTokenizer
            from kdflow.datasets.prompts_dataset import PromptDataset
            from kdflow.meta_data import MetaSampler,prepare_meta,POLICY
            tok=AutoTokenizer.from_pretrained(str(student),local_files_only=True,trust_remote_code=False)
            renderer=PromptDataset.__new__(PromptDataset); renderer.apply_chat_template=True; renderer.image_key=None; renderer.enable_thinking=False
            render=lambda r:renderer._build_prompt(r,tok,"messages")
            prepared,excluded=prepare_meta(M,render,tok,4096); sampler=MetaSampler(prepared,[render(r) for r in B],42,16); picked=sampler.batch(0,[render(r) for r in B[:64]])
            print("META_PRODUCTION_AUDIT policy",POLICY,"prepared",len(prepared),"excluded_length",excluded,"unique_groups",len(sampler.rows),"groups_sha256",sampler.audit["groups_sha256"],"step0_selected",len(picked))
            if len(sampler.rows)<80 or len(picked)!=16: failed=True
        except Exception as e:
            print("META_PRODUCTION_AUDIT_FAIL",type(e).__name__,str(e)[:500]); failed=True
if not energy.is_file():
    print("ENERGY_CONTRACT_BLOCKED no candidate checkpoint"); failed=True
else:
    eh=digest(energy); print("ENERGY_IDENTITY",energy,eh)
    if eh!=expected: print("ENERGY_SHA_MISMATCH expected",expected); failed=True
    try:
        import torch
        from kdflow.algorithms._mp_opd_energy import MPAtomEnergy,load_energy_checkpoint
        payload=torch.load(energy,map_location="cpu",weights_only=False)
        net=MPAtomEnergy(10,32,2); opt=torch.optim.AdamW(net.parameters())
        step=load_energy_checkpoint(energy,net,opt,expected_extra_config={"max_span_length":2})
        finite=all(bool(torch.isfinite(p).all()) for p in net.parameters())
        print("ENERGY_CONTRACT format",payload.get("format"),"step",step,"finite",finite,"config",payload.get("config"))
        if step!=128 or not finite: failed=True
    except Exception as e:
        print("ENERGY_CONTRACT_FAIL",type(e).__name__,str(e)[:500]); failed=True
print("READ_ONLY_RESULT","BLOCKED" if failed else "READY_FOR_EXACT_COMPANY_CANARY")
raise SystemExit(3 if failed else 0)
PY
