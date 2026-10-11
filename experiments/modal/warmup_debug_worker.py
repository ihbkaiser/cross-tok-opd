"""Actual SFT baseline/termination and teacher hidden-head parity diagnostic."""
import ast
import gc
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import sglang

from mp_opd_parity_worker import _stats, _token_ids, _engine_kwargs, _server_info
from answer_verifier import verify_response

OUT=Path('/results');OUT.mkdir(exist_ok=True)
TEMPERATURE=.6
STOP_IDS=[1,107]

def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False))

def release():
    gc.collect();torch.cuda.empty_cache()

def code_block(text):
    blocks=re.findall(r'```(?:python|py|python3)?\s*\n(.*?)```',text,re.S)
    return blocks[0] if blocks else text

def baseline(model_path,arm,cases):
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    ids=[_token_ids(tok.apply_chat_template([{'role':'user','content':c['prompt']}],tokenize=True,add_generation_prompt=True)) for c in cases]
    kwargs=_engine_kwargs(model_path,torch.cuda.get_device_name(0),'default',None)
    kwargs.update(dtype='bfloat16',context_length=4096,random_seed=42)
    engine=sglang.Engine(**kwargs)
    try:
        outputs=engine.generate(input_ids=ids,sampling_params={'temperature':TEMPERATURE,'top_p':.95,
            'max_new_tokens':1536,'ignore_eos':False,'sampling_seed':42},return_logprob=True,logprob_start_len=-1)
        samples=[]
        for i,(case,out) in enumerate(zip(cases,outputs)):
            values=out['meta_info']['output_token_logprobs'];out_ids=list(map(int,out['output_ids']))
            assert [int(v[1]) for v in values]==out_ids
            text=out['text'];finish=out['meta_info'].get('finish_reason');quality={}
            if case['kind']=='code':
                try:
                    tree=ast.parse(code_block(text));quality={'syntax_valid':True,'functions':[n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)]}
                except (SyntaxError,ValueError):quality={'syntax_valid':False,'functions':[]}
            elif case.get('gold'):
                quality={'heuristic_math_pass':verify_response(text,case['gold'],case.get('source','gsm8k'))}
            samples.append({'case':case,'prompt_ids':ids[i],'output_ids':out_ids,'decode_logprobs':[float(v[0]) for v in values],
                'text':text,'finish_reason':finish,'tokens':len(out_ids),'empty_text':not text.strip(),**quality})
        save(arm+'-baseline.json',{'model':model_path,'server_args':_server_info(engine),'samples':samples})
    finally:
        engine.shutdown();del engine;release()

    # Independent eager scoring, including zero decode slots, on exact token IDs.
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
    references=[];decode=[];stop_rows=[]
    with torch.inference_mode():
        for sample in samples[:4]:
            prefix=sample['prompt_ids'];response=sample['output_ids'][:128]
            if not response:continue
            full=torch.tensor([prefix+response],device='cuda')
            logits=model(full,use_cache=False).logits[0]
            selected=logits[len(prefix)-1:len(prefix)+len(response)-1].float()
            labels=torch.tensor(response,device='cuda')
            scores=(selected/TEMPERATURE).log_softmax(-1).gather(-1,labels[:,None]).squeeze(-1).tolist()
            references.extend(scores);decode.extend(sample['decode_logprobs'][:len(response)])
            for generated in sorted(set([0,min(8,len(response)-1),min(32,len(response)-1),len(response)-1])):
                probabilities=logits[len(prefix)-1+generated].float().softmax(-1)
                stop_rows.append({'case_id':sample['case']['id'],'generated_prefix_tokens':generated,
                    'raw_stop_mass':float(probabilities[STOP_IDS].sum()),'next_sampled_id':response[generated]})
    del model;release()
    summary={'arm':arm,'prompts':len(samples),'empty':sum(s['empty_text'] for s in samples),
      'finish_counts':{str(k):sum(s['finish_reason']==k for s in samples) for k in []},
      'mean_tokens':float(np.mean([s['tokens'] for s in samples])),'median_tokens':float(np.median([s['tokens'] for s in samples])),
      'short_le32':sum(s['tokens']<=32 for s in samples),'length_capped':sum(isinstance(s['finish_reason'],dict) and s['finish_reason'].get('type')=='length' for s in samples),
      'code_syntax_valid':sum(s.get('syntax_valid',False) for s in samples),'code_cases':sum(s['case']['kind']=='code' for s in samples),
      'math_heuristic_pass':sum(s.get('heuristic_math_pass',False) for s in samples),
      'decode_vs_hf_eager_tempered':_stats(decode,references),'zero_decode_slots':sum(v==0 for v in decode),
      'stop_mass_rows':stop_rows,'semantic_code_tests_run':False}
    from collections import Counter
    summary['finish_counts']=dict(Counter(json.dumps(s['finish_reason'],sort_keys=True) for s in samples))
    save(arm+'-baseline-summary.json',summary)
    print('BASELINE_SUMMARY_JSON='+json.dumps(summary),flush=True)
    return samples,summary

def teacher_probe(samples, path='/teacher_assets/teacher', student_arm='phi', teacher_name='Phi-4-mini-instruct'):
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    student_tok=AutoTokenizer.from_pretrained(json.loads(Path('/warmup/ready.json').read_text())['models'][student_arm],local_files_only=True)
    records=[]
    for sample in samples[:4]:
        response=student_tok.decode([x for x in sample['output_ids'][:96] if x not in STOP_IDS],skip_special_tokens=False)
        prompt=_token_ids(tok.apply_chat_template([{'role':'user','content':sample['case']['prompt']}],tokenize=True,add_generation_prompt=True))
        content=tok.encode(response,add_special_tokens=False)
        if not content:continue
        records.append({'case_id':sample['case']['id'],'prompt_ids':prompt,'content_ids':content,'full_ids':prompt+content+[tok.eos_token_id]})
    kwargs=_engine_kwargs(path,torch.cuda.get_device_name(0),'default',None)
    kwargs.update(dtype='bfloat16',context_length=16384,chunked_prefill_size=-1,disable_radix_cache=True,enable_return_hidden_states=True,disable_cuda_graph=True)
    engine=sglang.Engine(**kwargs);hidden=[];direct=[]
    try:
        outputs=engine.generate(input_ids=[r['full_ids'] for r in records],sampling_params={'max_new_tokens':0},
            return_hidden_states=True,return_logprob=True,logprob_start_len=0)
        for record,out in zip(records,outputs):
            meta=out['meta_info'];hs=np.asarray(meta['hidden_states'][0]);entries=meta['input_token_logprobs']
            assert hs.ndim==2 and hs.shape[0]==len(record['full_ids']),hs.shape
            assert [int(x[1]) for x in entries]==record['full_ids']
            positions=list(range(len(record['prompt_ids']),len(record['full_ids'])-1))
            hidden.append(hs[np.array(positions)-1].copy())
            direct.append([float(entries[i][0]) for i in positions])
        server_args=_server_info(engine)
    finally:
        engine.shutdown();del engine;release()
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
    reconstructed=[];normalized=[];hf_scores=[];hidden_comparisons=[];rows=[]
    with torch.inference_mode():
        for record,sg_hs,direct_scores in zip(records,hidden,direct):
            full=torch.tensor([record['full_ids']],device='cuda');output=model(full,use_cache=False,output_hidden_states=True)
            positions=torch.arange(len(record['prompt_ids'])-1,len(record['full_ids'])-2,device='cuda')
            labels=torch.tensor(record['content_ids'],device='cuda')
            hf_hidden=output.hidden_states[-1][0,positions]
            hs=torch.tensor(sg_hs,device='cuda',dtype=model.get_output_embeddings().weight.dtype)
            raw=model.get_output_embeddings()(hs).float().log_softmax(-1).gather(-1,labels[:,None]).squeeze(-1)
            corrected=model.get_output_embeddings()(model.model.norm(hs)).float().log_softmax(-1).gather(-1,labels[:,None]).squeeze(-1)
            actual=output.logits[0,positions].float().log_softmax(-1).gather(-1,labels[:,None]).squeeze(-1)
            reconstructed.extend(raw.tolist());normalized.extend(corrected.tolist());hf_scores.extend(actual.tolist())
            delta=(hs.float()-hf_hidden.float())
            hidden_comparisons.append({'case_id':record['case_id'],'rows':len(labels),
              'relative_l2':float(delta.norm()/hf_hidden.float().norm().clamp_min(1e-12)),
              'mean_cosine':float(torch.nn.functional.cosine_similarity(hs.float(),hf_hidden.float()).mean())})
            rows.append({'case_id':record['case_id'],'content_ids':record['content_ids'],'direct_sglang_raw':direct_scores,
              'production_hidden_head_raw':raw.tolist(),'norm_then_head_raw':corrected.tolist(),'hf_eager_raw':actual.tolist()})
    flat_direct=[v for rr in direct for v in rr]
    result={'teacher':teacher_name,'server_args':server_args,
      'hidden_head_vs_direct':_stats(reconstructed,flat_direct),'direct_vs_hf':_stats(flat_direct,hf_scores),
      'hidden_head_vs_hf':_stats(reconstructed,hf_scores),'norm_then_head_vs_direct':_stats(normalized,flat_direct),
      'hidden_state_vs_hf_final':hidden_comparisons,'samples':rows,
      'scope':'Exact same teacher prefix/content token IDs; production hidden API and tied LM head; no Ray/offload/full trainer trajectory.'}
    save('teacher-hidden-head.json',result);print('TEACHER_HIDDEN_HEAD_SUMMARY_JSON='+json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)
    del model;release();return result

def main():
    started=time.monotonic();assets=json.loads(Path('/warmup/ready.json').read_text());inputs=json.loads(Path('/opt/warmup_debug/inputs.json').read_text())
    env={'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'transformers':transformers.__version__,
         'sglang':sglang.__version__,'python':sys.version,'temperature':TEMPERATURE,'top_p':.95,'max_new_tokens':1536,'ignore_eos':False}
    save('asset-manifest.json',assets);save('inputs.json',inputs)
    save('environment.json',env);print('DEBUG_ENV_JSON='+json.dumps(env),flush=True)
    phi,phi_summary=baseline(assets['models']['phi'],'phi',inputs['cases'])
    qwen,qwen_summary=baseline(assets['models']['qwen'],'qwen',inputs['cases'])
    teacher=teacher_probe(phi)
    result={'status':'completed','environment':env,'phi':phi_summary,'qwen':qwen_summary,
       'teacher_hidden_head_vs_direct':teacher['hidden_head_vs_direct'],'teacher_direct_vs_hf':teacher['direct_vs_hf'],
       'elapsed_seconds':time.monotonic()-started,'training_run':False,'benchmark_accuracy_claim':False}
    save('summary.json',result);print('ACTUAL_WARMUP_DEBUG_SUMMARY_JSON='+json.dumps(result),flush=True)

if __name__=='__main__':main()
