"""Real Gemma numerical probe, synthetic mechanics references, no training."""
import argparse,json,time,os,hashlib
from pathlib import Path

def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from torch.func import functional_call
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args()
    torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    start=time.time();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained("/assets/student",local_files_only=True)
    prompt=[{"role":"user","content":"Write one short sentence about a blue cup."}]
    reference="The blue cup is on the table."
    prefix=tok.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True,return_dict=False)
    if hasattr(prefix,"keys"): prefix=prefix["input_ids"]
    response=tok.encode(reference,add_special_tokens=False)
    ids=torch.tensor([prefix+response],device="cuda",dtype=torch.long)
    records=[]
    for dtype in (torch.bfloat16,torch.float32):
        torch.manual_seed(42)
        model=AutoModelForCausalLM.from_pretrained("/assets/student",local_files_only=True,
            torch_dtype=dtype,attn_implementation="eager").cuda().eval()
        model.requires_grad_(False)
        name="model.layers.25.self_attn.q_proj";base=model.get_submodule(name)
        class Adapter(torch.nn.Module):
            def __init__(self,base):
                super().__init__();self.base=base
                self.register_buffer("a",torch.randn(4,base.in_features,device="cuda",dtype=torch.float32)/base.in_features**.5)
                self.b=torch.nn.Parameter(torch.zeros(base.out_features,4,device="cuda",dtype=torch.float32))
            def forward(self,x):
                delta=torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.a),self.b)/4
                return self.base(x)+delta.to(x.dtype)
        adapter=Adapter(base);model.model.layers[25].self_attn.q_proj=adapter
        param=name+".b"
        def forward(b):
            out=functional_call(model,{param:b},(),dict(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False))
            logits=out.logits[0,len(prefix)-1:len(prefix)+len(response)-1].float()
            labels=ids[0,len(prefix):]
            lp=torch.log_softmax(logits,-1).gather(1,labels[:,None]).squeeze(1)
            return -lp.mean(),logits
        torch.cuda.reset_peak_memory_stats()
        loss,logits=forward(adapter.b)
        grad=torch.autograd.grad(loss,adapter.b)[0].detach()
        baseline=logits.detach();before=loss.item();del loss,logits
        for lr in (0.,.001,.01,.1,1.):
            with torch.no_grad():
                updated=-lr*grad
                after,actual=forward(updated)
                delta=(actual-baseline).abs()
                row={"dtype":str(dtype),"control":"skip" if lr==0 else "meta_sft",
                    "lr":lr,"before_nll":before,"after_nll":after.item(),"nll_gain":before-after.item(),
                    "gradient_norm":grad.norm().item(),"parameter_delta_norm":updated.norm().item(),
                    "predicted_gain":lr*grad.square().sum().item(),
                    "logit_max_abs_delta":delta.max().item(),"changed_logit_fraction":(delta>0).float().mean().item(),
                    "peak_gpu_bytes":torch.cuda.max_memory_allocated()}
                records.append(row)
                with (root/"records.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
                print("SENSITIVITY_ROW="+json.dumps(row),flush=True)
        del actual,after,baseline,grad,adapter,model,base,updated,delta
        import gc;gc.collect();torch.cuda.empty_cache()
    result={"status":"completed","scope":"base Gemma cached checkpoint; synthetic mechanics sentence; not company SFT or MP-OPD efficacy",
        "model_manifest":json.loads(Path("/assets/ready.json").read_text()),"prompt":prompt,"reference":reference,
        "gpu":torch.cuda.get_device_name(),"torch":torch.__version__,"cuda":torch.version.cuda,
        "elapsed_seconds":time.time()-start,"records":records}
    (root/"result.json").write_text(json.dumps(result,indent=2));print("SENSITIVITY_COMPLETE="+json.dumps(result),flush=True)
if __name__=="__main__":main()
