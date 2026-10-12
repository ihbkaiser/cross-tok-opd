"""Check recorded zero decode logprobs at their exact, possibly long contexts."""
import json
from pathlib import Path
import sys
import time

from mp_opd_parity_worker import _stats


def zero_windows(samples,limit=3):
    rows=[]
    # Prefer different capped cases; each window includes an observed exact zero.
    for sample in sorted(samples,key=lambda s:s['tokens']!=4096):
        ids=sample['output_ids'];lp=sample['decode_logprobs']
        if len(ids)!=len(lp):raise ValueError('Output IDs/logprobs must align')
        zeros=[i for i,x in enumerate(lp) if x==0 and ids[i] not in (1,107)]
        if not zeros:continue
        tail_start=max(0,len(ids)-64)
        if sample['tokens']==4096 and any(i>=tail_start for i in zeros):
            start=tail_start
        else:
            anchor=next((i for i in zeros if i>=1024),zeros[0])
            start=max(0,anchor-8)
        end=min(len(ids),start+64)
        rows.append({'case_id':sample['case']['id'],'source_capped':sample['tokens']==4096,
            'position':start,'prefix_ids':sample['prompt_ids']+ids[:start],
            'target_ids':ids[start:end],'recorded_decode_logprob':lp[start:end]})
        if len(rows)==limit:break
    if not rows:raise ValueError('Policy has no recorded zeros to audit')
    return rows


def main():
    import torch
    from transformers import AutoModelForCausalLM
    root=Path('/results');arm=sys.argv[2]
    path=json.loads(Path('/warmup/ready.json').read_text())['models']['qwen'] if arm=='warmup' else json.loads(Path('/checkpoints/ready.json').read_text())['models'][arm]['model_path']
    summary=json.loads((root/(arm+'-summary.json')).read_text());temperature=summary['sampling']['temperature']
    rows=zero_windows(json.loads((root/(arm+'-responses.json')).read_text()))
    started=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    left=[];right=[]
    with torch.inference_mode():
        for row in rows:
            targets=row['target_ids'];n=len(targets)
            # Last n causal input positions predict exactly the n target IDs.
            ids=row['prefix_ids']+targets[:-1]
            logits=model(torch.tensor([ids],device='cuda'),use_cache=False,logits_to_keep=n).logits
            assert logits.shape[1]==n
            lp=(logits[0].float()/temperature).log_softmax(-1)
            selected=lp.gather(-1,torch.tensor(targets,device='cuda')[:,None]).squeeze(-1).tolist()
            row['hf_sdpa_temperature_matched_logprob']=selected
            row['parity']=_stats(row['recorded_decode_logprob'],selected)
            zero_pairs=[b for a,b in zip(row['recorded_decode_logprob'],selected) if a==0]
            row['recorded_zero_count']=len(zero_pairs)
            row['max_abs_hf_at_recorded_zero']=max(abs(b) for b in zero_pairs)
            row['context_tokens']=len(ids)
            left.extend(row['recorded_decode_logprob']);right.extend(selected)
    result={'arm':arm,'model_path':path,'temperature':temperature,'records':rows,
        'summary':_stats(left,right),'elapsed_seconds':time.monotonic()-started,
        'scope':'Recorded SGLang decode vs independent HF SDPA at identical full context, token IDs and temperature; selected zero windows only'}
    (root/(arm+'-zeroaudit.json')).write_text(json.dumps(result,indent=2))
    print('ZERO_LOGPROB_AUDIT='+json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
