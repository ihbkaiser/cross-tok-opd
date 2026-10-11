"""Independent causal-prefix and backend checks for Qwen teacher scoring."""
import gc
import json
from pathlib import Path
import numpy as np
import sglang
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from mp_opd_parity_worker import _engine_kwargs,_stats,_token_ids

ROOT=Path('/results');PATH='/teacher_assets/teacher'

def main():
    tok=AutoTokenizer.from_pretrained(PATH,local_files_only=True)
    old=json.loads((ROOT/'teacher-hidden-head.json').read_text())
    baseline=json.loads(Path('/prior/qwen-baseline.json').read_text())['samples'][:4]
    jobs=[];prefix_jobs=[]
    for sample,row in zip(baseline,old['samples']):
        prompt=_token_ids(tok.apply_chat_template([{'role':'user','content':sample['case']['prompt']}],tokenize=True,add_generation_prompt=True))
        content=row['content_ids'];jobs.append({'case_id':row['case_id'],'prompt':prompt,'content':content,'ids':prompt+content+[tok.eos_token_id]})
        positions=sorted(set([0,1]+sorted(range(len(content)),key=lambda i:abs(row['direct_sglang_raw'][i]-row['hf_eager_raw'][i]),reverse=True)[:2]))
        for i in positions:prefix_jobs.append({'case_id':row['case_id'],'position':i,'prefix':prompt+content[:i],'label':content[i]})
    result={'jobs':jobs,'prefix_jobs':prefix_jobs,'backends':{}}
    for backend in ['trtllm_mha','flashinfer']:
        kwargs=_engine_kwargs(PATH,torch.cuda.get_device_name(0),'default',None)
        kwargs.update(dtype='bfloat16',context_length=16384,disable_radix_cache=True,chunked_prefill_size=-1,
            disable_cuda_graph=True,attention_backend=backend)
        engine=sglang.Engine(**kwargs)
        try:
            full=engine.generate(input_ids=[j['ids'] for j in jobs],sampling_params={'max_new_tokens':0},return_logprob=True,logprob_start_len=0)
            scores=[]
            for j,o in zip(jobs,full):
                entries=o['meta_info']['input_token_logprobs'];assert [int(e[1]) for e in entries]==j['ids']
                scores.extend(float(v[0]) for v in entries[len(j['prompt']):-1])
            causal=[]
            for j in prefix_jobs:
                out=engine.generate(input_ids=j['prefix'],sampling_params={'max_new_tokens':1,'temperature':0,'ignore_eos':True},
                    return_logprob=True,logprob_start_len=-1,token_ids_logprob=[j['label']])
                entries=out['meta_info']['output_token_ids_logprobs'][0]
                score=next(float(v[0]) for v in entries if int(v[1])==j['label'])
                causal.append({'case_id':j['case_id'],'position':j['position'],'score':score,'greedy_id':out['output_ids'][0]})
            result['backends'][backend]={'full_scores':scores,'causal_prefix_scores':causal}
        finally:engine.shutdown();del engine;gc.collect();torch.cuda.empty_cache()
    hf=[];causal_hf=[];wrong_mass=[]
    bad=json.loads((ROOT/'tokenizer-audit.json').read_text())['overlap_mismatch_rows']
    for attention in ['eager','sdpa']:
        model=AutoModelForCausalLM.from_pretrained(PATH,local_files_only=True,dtype=torch.bfloat16,attn_implementation=attention).to('cuda').eval()
        scores=[]
        with torch.inference_mode():
            for j in jobs:
                ids=torch.tensor([j['ids']],device='cuda');output=model(ids,attention_mask=torch.ones_like(ids),use_cache=False)
                logits=output.logits[0,len(j['prompt'])-1:len(j['ids'])-2].float()
                lp=logits.log_softmax(-1);labels=torch.tensor(j['content'],device='cuda')
                scores.extend(lp.gather(-1,labels[:,None]).squeeze(-1).tolist())
                if attention=='eager':
                    wrong_ids=torch.tensor([r['teacher_id'] for r in bad],device='cuda')
                    mass=lp[:,wrong_ids].exp().sum(-1)
                    top=sorted(range(len(mass)),key=lambda i:float(mass[i]),reverse=True)[:3]
                    wrong_mass.append({'case_id':j['case_id'],'mean':float(mass.mean()),'max':float(mass.max()),
                        'top_positions':[{'position':i,'total_wrong_mass':float(mass[i]),
                            'top_wrong_candidates':[{**bad[k],'probability':float(lp[i,bad[k]['teacher_id']].exp())} for k in torch.topk(lp[i,wrong_ids],3).indices.tolist()]} for i in top]})
            if attention=='eager':
                for j in prefix_jobs:
                    ids=torch.tensor([j['prefix']],device='cuda');logits=model(ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1].float()
                    causal_hf.append(float(logits.log_softmax(-1)[j['label']]))
        hf.append({'attention':attention,'scores':scores});del model;gc.collect();torch.cuda.empty_cache()
    result['hf']=hf;result['hf_causal_prefix_scores']=causal_hf;result['wrong_shared_candidate_mass']=wrong_mass
    summary={b:{'full_vs_hf_eager':_stats(v['full_scores'],hf[0]['scores']),
        'prefix_vs_hf_eager':_stats([r['score'] for r in v['causal_prefix_scores']],causal_hf)} for b,v in result['backends'].items()}
    summary['hf_eager_vs_sdpa']=_stats(hf[0]['scores'],hf[1]['scores'])
    lookup={(j['case_id'],i):v for j,values in zip(jobs,np.split(np.asarray(hf[0]['scores']),np.cumsum([len(j['content']) for j in jobs])[:-1])) for i,v in enumerate(values)}
    summary['hf_full_vs_causal_prefix']=_stats([lookup[(j['case_id'],j['position'])] for j in prefix_jobs],causal_hf)
    result['summary']=summary
    (ROOT/'scoring-isolation.json').write_text(json.dumps(result,indent=2))
    print('QWEN_SCORING_ISOLATION_SUMMARY='+json.dumps(summary),flush=True)
    print('WRONG_SHARED_MASS='+json.dumps(wrong_mass),flush=True)

if __name__=='__main__':main()
