"""Independent exact-prefix replay at early and late RL3 loop positions."""
import gc
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import sglang
from mp_opd_parity_worker import _stats,_engine_kwargs


def healthy_terminal_prefixes(root):
    """Common completed SFT answers, selected only after successful math grading."""
    samples=json.loads((root/'warmup-responses.json').read_text())
    grades={r['id']:r for r in json.loads((root/'warmup-mathverify.json').read_text())}
    rows=[]
    for sample in samples:
        case_id=sample['case']['id']
        if case_id not in ('math-apples','math-speed','math-age'):continue
        if grades.get(case_id,{}).get('passed') is not True:continue
        output=list(sample['output_ids'])
        if not output or output[-1] not in (1,107):continue
        while output and output[-1] in (1,107):output.pop()
        if output:rows.append({'case_id':case_id,'prefix_ids':sample['prompt_ids']+output})
    return rows

def main():
    OUT=Path('/results')
    arm=sys.argv[-1]
    if arm=='warmup':path=json.loads(Path('/warmup/ready.json').read_text())['models']['qwen']
    else:path=json.loads(Path('/checkpoints/ready.json').read_text())['models'][arm]['model_path']
    probe_arm=sys.argv[3] if sys.argv[1]=='deep-prefix' else 'temporal-rl2-s44-step160' if sys.argv[1]=='deep-temporal' else 'grass-rl3-s42-step312'
    source=OUT/(probe_arm+'-responses.json')
    samples=json.loads(source.read_text())
    targets=[s for s in samples if s['tokens']==4096][:3]
    if not targets:raise ValueError('Source-policy screen has no capped cases to probe')
    records=[]
    for sample in targets:
        for position in (128,768):
            prefix=sample['prompt_ids']+sample['output_ids'][:position]
            labels=sample['output_ids'][position:position+64]
            records.append({'case_id':sample['case']['id'],'position':position,
                            'prefix_ids':prefix,'target_ids':labels})
    kwargs=_engine_kwargs(path,torch.cuda.get_device_name(0),'default',None)
    kwargs.update(dtype='bfloat16',context_length=8192,disable_cuda_graph=True,disable_radix_cache=True)
    started=time.monotonic()
    engine=sglang.Engine(**kwargs)
    try:
        outputs=engine.generate(input_ids=[r['prefix_ids']+r['target_ids'] for r in records],
            sampling_params={'max_new_tokens':0,'temperature':1},return_logprob=True,logprob_start_len=0)
        for row,out in zip(records,outputs):
            entries=out['meta_info']['input_token_logprobs'];ids=row['prefix_ids']+row['target_ids']
            assert [int(x[1]) for x in entries]==ids
            row['sg_raw_logprob']=[float(x[0]) for x in entries[len(row['prefix_ids']):]]
        outputs=engine.generate(input_ids=[r['prefix_ids'] for r in records],
            sampling_params={'temperature':0,'max_new_tokens':64,'ignore_eos':False})
        for row,out in zip(records,outputs):
            row['sg_continuation_ids']=list(map(int,out['output_ids']))
            row['sg_continuation']=out['text']
    finally:
        engine.shutdown();del engine;gc.collect();torch.cuda.empty_cache()

    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    sg_all=[];hf_all=[]
    healthy=[]
    with torch.inference_mode():
        for row in records:
            prefix=row['prefix_ids'];labels=row['target_ids']
            inp=torch.tensor([prefix+labels],device='cuda')
            # Full logits at <=~1k tokens are bounded on B200; keep only causal target rows.
            logits=model(inp,use_cache=False).logits[0,len(prefix)-1:len(prefix)+len(labels)-1].float()
            lp=logits.log_softmax(-1)
            targets=torch.tensor(labels,device='cuda')
            selected=lp.gather(-1,targets[:,None]).squeeze(-1)
            row['hf_sdpa_raw_logprob']=selected.tolist()
            row['mean_target_nll']=float(-selected.mean())
            row['mean_raw_stop_mass']=float(lp.exp()[:,[1,107]].sum(-1).mean())
            row['mean_entropy']=float(-(lp.exp()*lp).sum(-1).mean())
            row['sg_vs_hf_raw']=_stats(row['sg_raw_logprob'],row['hf_sdpa_raw_logprob'])
            sg_all.extend(row['sg_raw_logprob']);hf_all.extend(row['hf_sdpa_raw_logprob'])
            pre=torch.tensor([prefix],device='cuda')
            output=model.generate(pre,attention_mask=torch.ones_like(pre),do_sample=False,
                                 max_new_tokens=64,pad_token_id=tok.pad_token_id)
            new=output[0,len(prefix):].tolist()
            row['hf_continuation_ids']=new;row['hf_continuation']=tok.decode(new,skip_special_tokens=True)
            row['greedy_equal_ids']=new==row['sg_continuation_ids']
            row['greedy_common_prefix']=next((i for i,(a,b) in enumerate(zip(new,row['sg_continuation_ids'])) if a!=b),
                                           min(len(new),len(row['sg_continuation_ids'])))
        # Matched healthy answer endpoints distinguish entry/stopping drift from
        # persistence inside already repetitive prefixes. These are HF-only probes.
        for probe in healthy_terminal_prefixes(OUT):
            logits=model(torch.tensor([probe['prefix_ids']],device='cuda'),use_cache=False).logits[0,-1].float()
            lp=logits.log_softmax(-1);top=lp.topk(8).indices.tolist()
            healthy.append({**probe,'raw_stop_mass':float(lp.exp()[[1,107]].sum()),
                            'eos_logprob':float(lp[1]),'eot_logprob':float(lp[107]),
                            'top_ids':top,'top_text':[tok.decode([x]) for x in top],
                            'top_logprobs':[float(lp[x]) for x in top],
                            'scope':'HF SDPA only; identical correctly graded SFT endpoint; not own-policy termination frequency'})
    del model;gc.collect();torch.cuda.empty_cache()
    result={'arm':arm,'probe_arm':probe_arm,'model_path':path,'records':records,'summary':_stats(sg_all,hf_all),
            'healthy_terminal_probes':healthy,
            'greedy_equal':sum(r['greedy_equal_ids'] for r in records),'n':len(records),
            'elapsed_seconds':time.monotonic()-started,
            'scope':'Same source-policy prefix and targets forced into all policies; not training credit or full benchmark'}
    (OUT/(arm+'-deep-prefix-'+probe_arm+'.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('DEEP_PREFIX_SUMMARY='+json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=="__main__":main()
