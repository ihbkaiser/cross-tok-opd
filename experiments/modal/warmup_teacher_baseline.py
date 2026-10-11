"""Native Phi teacher quality control on the shared student diagnostic fixtures."""
import ast
import json
from pathlib import Path
import time

import sglang
import torch
from transformers import AutoTokenizer

from mp_opd_parity_worker import _engine_kwargs, _server_info, _token_ids
from answer_verifier import verify_response
from warmup_debug_worker import code_block


def main():
    root=Path('/results'); path='/teacher_assets/teacher'
    provenance=json.loads(Path('/teacher_assets/assets.json').read_text())
    assert provenance['teacher_revision']=='cfbefacb99257ffa30c83adab238a50856ac3083'
    cases=json.loads(Path('/opt/warmup_debug/inputs.json').read_text())['cases']
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    ids=[_token_ids(tok.apply_chat_template([{'role':'user','content':c['prompt']}],tokenize=True,add_generation_prompt=True)) for c in cases]
    kwargs=_engine_kwargs(path,torch.cuda.get_device_name(0),'default',None)
    kwargs.update(dtype='bfloat16',context_length=16384,random_seed=42)
    engine=sglang.Engine(**kwargs)
    start=time.monotonic()
    try:
        outputs=engine.generate(input_ids=ids,sampling_params={'temperature':.6,'top_p':.95,
            'max_new_tokens':1536,'ignore_eos':False,'sampling_seed':42},return_logprob=True,logprob_start_len=-1)
        samples=[]
        for case,prefix,out in zip(cases,ids,outputs):
            quality={}
            if case['kind']=='code':
                try: ast.parse(code_block(out['text']));quality['syntax_valid']=True
                except SyntaxError: quality['syntax_valid']=False
            else: quality['heuristic_math_pass']=verify_response(out['text'],case['gold'],case['source'])
            samples.append({'case':case,'prompt_ids':prefix,'text':out['text'],'output_ids':out['output_ids'],
                'decode_logprobs':[float(v[0]) for v in out['meta_info']['output_token_logprobs']],
                'tokens':len(out['output_ids']),'finish_reason':out['meta_info'].get('finish_reason'),**quality})
        result={'model':path,'teacher_revision':provenance['teacher_revision'],'samples':samples,
            'server_args':_server_info(engine),'elapsed_seconds':time.monotonic()-start,
            'scope':'Native Phi teacher on the same 16 diagnostic fixtures; not benchmark accuracy'}
        (root/'phi_teacher-baseline.json').write_text(json.dumps(result,indent=2))
        summary={'prompts':len(samples),'empty':sum(not s['text'].strip() for s in samples),
            'mean_tokens':sum(s['tokens'] for s in samples)/len(samples),
            'code_syntax_valid':sum(s.get('syntax_valid',False) for s in samples),
            'math_heuristic_pass':sum(s.get('heuristic_math_pass',False) for s in samples)}
        (root/'phi_teacher-baseline-summary.json').write_text(json.dumps(summary,indent=2))
        print('NATIVE_TEACHER_BASELINE_JSON='+json.dumps(summary),flush=True)
    finally: engine.shutdown()


if __name__=='__main__':main()
