"""Float32 eager and causal-prefix SDPA adjudication on existing exact inputs."""
import gc
import inspect
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from mp_opd_parity_worker import _stats

def main():
    root=Path('/results');x=json.loads((root/'scoring-isolation.json').read_text())
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    result={}
    for attention,dtype in [('sdpa',torch.bfloat16),('eager',torch.float32)]:
        model=AutoModelForCausalLM.from_pretrained('/teacher_assets/teacher',local_files_only=True,
            dtype=dtype,attn_implementation=attention).to('cuda').eval()
        scores=[];prefix=[]
        with torch.inference_mode():
            for j in x['jobs']:
                ids=torch.tensor([j['ids']],device='cuda');logits=model(ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0]
                chosen=logits[len(j['prompt'])-1:len(j['ids'])-2].float().log_softmax(-1)
                labels=torch.tensor(j['content'],device='cuda');scores.extend(chosen.gather(-1,labels[:,None]).squeeze(-1).tolist())
            for j in x['prefix_jobs']:
                ids=torch.tensor([j['prefix']],device='cuda');logits=model(ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1].float()
                prefix.append(float(logits.log_softmax(-1)[j['label']]))
        result[attention]={'dtype':str(dtype),'full_scores':scores,'prefix_scores':prefix}
        del model;gc.collect();torch.cuda.empty_cache()
    summary={b:{'vs_fp32_eager':_stats(v['full_scores'],result['eager']['full_scores']),
        'prefix_vs_fp32_eager':_stats([r['score'] for r in v['causal_prefix_scores']],result['eager']['prefix_scores'])} for b,v in x['backends'].items()}
    summary['bf16_sdpa_vs_fp32_eager']=_stats(result['sdpa']['full_scores'],result['eager']['full_scores'])
    summary['bf16_eager_vs_fp32_eager']=_stats(x['hf'][0]['scores'],result['eager']['full_scores'])
    summary['bf16_sdpa_prefix_vs_fp32_eager_prefix']=_stats(result['sdpa']['prefix_scores'],result['eager']['prefix_scores'])
    import transformers.models.qwen2.modeling_qwen2 as qwen
    result['attention_source']=inspect.getsource(qwen.eager_attention_forward)
    result['summary']=summary
    (root/'scoring-reference.json').write_text(json.dumps(result,indent=2))
    print('QWEN_REFERENCE_SUMMARY='+json.dumps(summary),flush=True)

if __name__=='__main__':main()
