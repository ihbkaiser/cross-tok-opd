"""Matched raw generation, independent fixed-prefix scoring, and CPU regrading."""
import ast
import contextlib
from collections import Counter
import gc
import hashlib
import io
import json
from pathlib import Path
import re
import signal
import sys
import time

OUT = Path('/results')


def save(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2, ensure_ascii=False))


def repeat_fraction(ids, n=5):
    grams = [tuple(ids[i:i+n]) for i in range(max(0, len(ids)-n+1))]
    return 1-len(set(grams))/len(grams) if grams else 0.


def code_block(text):
    blocks = re.findall(r'```(?:python|py|python3)?\s*\n(.*?)```', text, re.S)
    return blocks[0] if blocks else text


def inputs():
    config = json.loads(Path('/opt/replay/checkpoint_replay_inputs.json').read_text())
    cases = config['cases']
    manifest = json.loads(Path('/assets/manifest.json').read_text())
    # Reuse pinned held-out test items. Selection is diagnostic, not a full benchmark.
    items = manifest['items']
    for bench in ('math500', 'gsm8k'):
        selected = [x for x in items if x['benchmark']==bench][:8]
        for item in selected:
            cases.append({'id':f"heldout-{bench}-{item['index']}", 'kind':'math',
                          'source':bench,'gold':item['gold'],'prompt':item['prompt']})
    return config, cases


def replay(arm):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    import sglang
    from mp_opd_parity_worker import _token_ids, _engine_kwargs, _server_info, _stats

    started = time.monotonic()
    if arm=='warmup':
        path = json.loads(Path('/warmup/ready.json').read_text())['models']['qwen']
    else:
        ready=Path('/checkpoints/ready.json')
        if ready.exists():
            path=json.loads(ready.read_text())['models'][arm]['model_path']
        else:
            receipts=list(Path('/checkpoints').glob('*/'+arm+'-receipt.json'))
            if len(receipts)!=1:raise RuntimeError('Checkpoint not verified and ready: '+arm)
            path=json.loads(receipts[0].read_text())['model_path']
    config, cases = inputs()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    ids = [_token_ids(tok.apply_chat_template([{'role':'user','content':c['prompt']}],
           tokenize=True, add_generation_prompt=True)) for c in cases]
    kwargs = _engine_kwargs(path, torch.cuda.get_device_name(0), 'default', None)
    kwargs.update(dtype='bfloat16', context_length=8192, random_seed=42,
                  disable_cuda_graph=True, disable_radix_cache=True)
    params = dict(temperature=.6,top_p=.95,top_k=-1,max_new_tokens=4096,
                  ignore_eos=False,sampling_seed=42)
    engine = sglang.Engine(**kwargs)
    try:
        outputs = engine.generate(input_ids=ids, sampling_params=params,
                                  return_logprob=True, logprob_start_len=-1)
        server = _server_info(engine)
        samples = []
        for case, prefix, output in zip(cases, ids, outputs):
            values = output['meta_info']['output_token_logprobs']
            tokens = list(map(int, output['output_ids']))
            if [int(v[1]) for v in values] != tokens:
                raise RuntimeError('Decode logprob/token IDs do not align')
            text=output['text']
            quality={}
            if case['kind']=='code':
                try: ast.parse(code_block(text));quality['syntax_valid']=True
                except (SyntaxError,ValueError): quality['syntax_valid']=False
            samples.append({'case':case,'prompt_ids':prefix,'output_ids':tokens,
                'prompt_ids_sha256':hashlib.sha256(json.dumps(prefix).encode()).hexdigest(),
                'text':text,'tokens':len(tokens),'decode_logprobs':[float(v[0]) for v in values],
                'finish_reason':output['meta_info'].get('finish_reason'),
                'repeat5_fraction':repeat_fraction(tokens),**quality})
        save(arm+'-responses.json', samples)
        print('GENERATION_SAVED='+json.dumps({'arm':arm,'n':len(samples),
           'capped':sum(s['tokens']==4096 for s in samples)}),flush=True)
    finally:
        engine.shutdown();del engine;gc.collect();torch.cuda.empty_cache()

    # Independent Transformers SDPA score of exact SGLang output IDs.
    # Includes short/EOS-bearing outputs and the first two historical math failures.
    model = AutoModelForCausalLM.from_pretrained(path,local_files_only=True,
            dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    picked = [0,2] + [i for i,c in enumerate(cases) if c['id'].startswith('historical-')][:2]
    comparisons=[];flat_hf=[];flat_sg=[];greedy=[];shared=[]
    with torch.inference_mode():
        for idx in picked:
            s=samples[idx];response=s['output_ids'][:128];prefix=s['prompt_ids']
            if not response:continue
            full=torch.tensor([prefix+response],device='cuda')
            logits=model(full,use_cache=False).logits[0,len(prefix)-1:len(prefix)+len(response)-1].float()
            labels=torch.tensor(response,device='cuda')
            score=(logits/.6).log_softmax(-1).gather(-1,labels[:,None]).squeeze(-1).tolist()
            flat_hf.extend(score);flat_sg.extend(s['decode_logprobs'][:len(response)])
            stop_ids=model.generation_config.eos_token_id
            stop_ids=[stop_ids] if isinstance(stop_ids,int) else stop_ids
            comparisons.append({'case_id':s['case']['id'],'tokens':response,'hf_sdpa_tempered':score,
                'sg_tempered':s['decode_logprobs'][:len(response)],
                'stop_ids':stop_ids,'raw_stop_mass':logits.softmax(-1)[:,stop_ids].sum(-1).tolist()})
        # Same frozen prefixes across every checkpoint: healthy code vs math loop.
        for probe in config['frozen_probes']:
            pre=_token_ids(tok.apply_chat_template([{'role':'user','content':probe['prompt']}],
                tokenize=True,add_generation_prompt=True)) + tok.encode(probe['prefix'],add_special_tokens=False)
            logits=model(torch.tensor([pre],device='cuda'),use_cache=False).logits[0,-1].float()
            lp=logits.log_softmax(-1)
            top=logits.topk(8).indices.tolist()
            candidates={}
            for label,text in probe['candidates'].items():
                target=tok.encode(text,add_special_tokens=False)
                if target: candidates[label]={'first_id':target[0],'first_token_logprob':float(lp[target[0]])}
            shared.append({'id':probe['id'],'prefix_ids':pre,'candidates':candidates,
                'top_ids':top,'top_text':[tok.decode([x]) for x in top],
                'raw_stop_mass':float(logits.softmax(-1)[[1,107]].sum())})
        # Greedy cross-backend test uses same tokenized prompts and no sampling.
        greedy_picked=[picked[0],picked[2]]
        for idx in greedy_picked:
            pre=ids[idx];inp=torch.tensor([pre],device='cuda')
            output=model.generate(inp,attention_mask=torch.ones_like(inp),do_sample=False,
                                  max_new_tokens=512,pad_token_id=tok.pad_token_id)
            new=output[0,len(pre):].tolist()
            greedy.append({'case_id':cases[idx]['id'],'output_ids':new,
                           'text':tok.decode(new,skip_special_tokens=True)})
    del model;gc.collect();torch.cuda.empty_cache()
    # SGLang greedy replay in a fresh engine for independent inference parity.
    engine=sglang.Engine(**kwargs)
    try:
        generated=engine.generate(input_ids=[ids[i] for i in greedy_picked],
            sampling_params=dict(temperature=0,max_new_tokens=512,ignore_eos=False))
        for hf,sg in zip(greedy,generated):
            left=hf['output_ids'];right=list(map(int,sg['output_ids']))
            common=next((i for i,(a,b) in enumerate(zip(left,right)) if a!=b),min(len(left),len(right)))
            hf.update(sg_output_ids=right,sg_text=sg['text'],equal_ids=left==right,
                      common_prefix_tokens=common)
    finally: engine.shutdown();del engine;gc.collect();torch.cuda.empty_cache()
    save(arm+'-crossbackend.json',{'fixed_prefix':comparisons,'shared_probes':shared,'greedy':greedy})
    summary={'arm':arm,'model_path':path,'n':len(samples),'sampling':params,'server':server,
       'environment':{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
                      'transformers':transformers.__version__,'sglang':sglang.__version__},
       'finish_counts':dict(Counter(json.dumps(s['finish_reason'],sort_keys=True) for s in samples)),
       'capped':sum(s['tokens']==4096 for s in samples),
       'loop_capped':sum(s['tokens']==4096 and s['repeat5_fraction']>=.75 for s in samples),
       'short_le32':sum(s['tokens']<=32 for s in samples),
       'code_syntax_valid':sum(s.get('syntax_valid',False) for s in samples),
       'mean_tokens':float(np.mean([s['tokens'] for s in samples])),
       'sg_vs_hf_sdpa':_stats(flat_sg,flat_hf),
       'greedy_equal':sum(x['equal_ids'] for x in greedy),'greedy_n':len(greedy),
       'elapsed_seconds':time.monotonic()-started,'scope':config['scope']}
    save(arm+'-summary.json',summary);print('REPLAY_SUMMARY='+json.dumps(summary),flush=True)


def score(only_arm=None):
    sys.path.insert(0,'/assets/mathverify080-v1')
    from math_verify import parse,verify,LatexExtractionConfig,ExprExtractionConfig
    def grade(gold,text):
        target=parse('$'+gold.split('####')[-1].strip()+'$',extraction_config=[LatexExtractionConfig()])
        if not target:raise ValueError('Gold unparseable')
        pred=parse(text,extraction_config=[LatexExtractionConfig(),ExprExtractionConfig()])
        return bool(pred and verify(target,pred))
    assert grade('1/2',r'\boxed{0.5}') and not grade('2',r'\boxed{3}')
    summaries={}
    for source in sorted(OUT.glob('*-responses.json')):
        if only_arm and source.name!=only_arm+'-responses.json':continue
        rows=[]
        for s in json.loads(source.read_text()):
            if s['case']['kind']!='math':continue
            error=None
            warnings=io.StringIO()
            def expired(signum,frame):raise TimeoutError('Per-answer symbolic scoring timeout')
            previous=signal.signal(signal.SIGALRM,expired)
            try:
                signal.alarm(3)
                with contextlib.redirect_stdout(warnings),contextlib.redirect_stderr(warnings):
                    passed=grade(s['case']['gold'],s['text'])
                if 'Timeout' in warnings.getvalue() or 'timed out' in warnings.getvalue():
                    passed=None;error='Internal symbolic scorer timeout; not counted as model failure'
            except Exception as exc:passed=None;error=type(exc).__name__+': '+str(exc)
            finally:
                signal.alarm(0);signal.signal(signal.SIGALRM,previous)
            rows.append({'id':s['case']['id'],'source':s['case'].get('source'),
                         'passed':passed,'error':error,'tokens':s['tokens'],
                         'repeat5_fraction':s['repeat5_fraction'],'scorer_messages':warnings.getvalue()[-2000:]})
        arm=source.name.removesuffix('-responses.json')
        save(arm+'-mathverify.json',rows)
        summaries[arm]={'correct':sum(x['passed'] is True for x in rows),'n':len(rows),
                         'errors':sum(x['passed'] is None for x in rows)}
    # Preserve prior arms when only a newly completed checkpoint is graded.
    for graded in OUT.glob('*-mathverify.json'):
        arm=graded.name.removesuffix('-mathverify.json')
        rows=json.loads(graded.read_text())
        summaries[arm]={'correct':sum(x['passed'] is True for x in rows),'n':len(rows),
                         'errors':sum(x['passed'] is None for x in rows)}
    save('mathverify-summary.json',summaries);print('REPLAY_MATHVERIFY='+json.dumps(summaries),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='score':score(sys.argv[2] if len(sys.argv)>2 and sys.argv[2] else None)
    else:replay(sys.argv[2])
