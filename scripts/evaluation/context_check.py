"""CPU-only exact chat-template context qualification; never truncate."""
import argparse
import json
from pathlib import Path
CONTEXT_LENGTH=8192

def check_items(tokenizer, items, cap):
    longest=0; oversized=[]
    for item in items:
        ids=tokenizer.apply_chat_template(item["messages"], tokenize=True,
            add_generation_prompt=True, return_dict=False, enable_thinking=False)
        if hasattr(ids,"keys"): ids=ids["input_ids"]
        if hasattr(ids,"ids"): ids=ids.ids
        if not isinstance(ids,list) or not all(type(x) is int for x in ids):
            raise ValueError("expected flat chat token IDs")
        n=len(ids);longest=max(longest,n)
        if n+cap>CONTEXT_LENGTH:
            oversized.append({"id":item["id"],"prompt_tokens":n,"max_tokens":cap})
    return {"max_prompt_tokens":longest,"oversized":oversized,"count":len(items)}

def main():
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--checkpoint",required=True);a=p.parse_args()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.checkpoint,local_files_only=True,trust_remote_code=False)
    plan=json.loads(a.plan.read_text());reports={}
    for name,data in plan["data"].items():
        items=json.loads(Path(data["path"]).read_text())["items"]
        reports[name]=check_items(tokenizer,items,plan["protocol"]["caps"][name])
    print(json.dumps({"context_length":CONTEXT_LENGTH,"reports":reports}),flush=True)
    if any(r["oversized"] for r in reports.values()):
        raise ValueError("prompt + full output cap exceeds 8192; no truncation or benchmark omission permitted")
if __name__=="__main__":main()
