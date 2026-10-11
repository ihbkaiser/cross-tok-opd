"""Paired teacher scoring: same text, AutoTokenizer versus serialized backend IDs."""
import json
from pathlib import Path

import sglang
import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from mp_opd_parity_worker import _engine_kwargs, _server_info, _token_ids
from kdflow.algorithms._mp_opd_atoms import mp_content_ids


def main():
    root=Path('/results');path='/teacher_assets/teacher'
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    serialized=Tokenizer.from_file(path+'/tokenizer.json')
    assets=json.loads(Path('/warmup/ready.json').read_text())
    stu_tok=AutoTokenizer.from_pretrained(assets['models']['phi'],local_files_only=True)
    student=json.loads((root/'phi-baseline.json').read_text())['samples'][:4]
    native=json.loads((root/'phi_teacher-baseline.json').read_text())['samples']
    frozen=json.loads((root/'frozen-credit-probe.json').read_text())['rows']
    jobs=[]
    def add_pair(case,text,kind,student_tokens=None,student_log_score=None):
        prefix=_token_ids(tok.apply_chat_template([{'role':'user','content':case['prompt']}],tokenize=True,add_generation_prompt=True))
        for mode in ('auto','serialized'):
            content=tok.encode(text,add_special_tokens=False) if mode=='auto' else serialized.encode(text,add_special_tokens=False).ids
            assert tok.decode(content,skip_special_tokens=False,clean_up_tokenization_spaces=False)==text
            jobs.append({'case_id':case['id'],'kind':kind,'mode':mode,'prefix':prefix,'content':content,
                'text':text,'full_ids':prefix+content+[tok.eos_token_id],
                'student_tokens':student_tokens,'student_log_score':student_log_score})
    for sample,credit in zip(student,frozen):
        ids,_=mp_content_ids(sample['output_ids'][:96],stu_tok)
        text=stu_tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        add_pair(sample['case'],text,'student-prefix',len(ids),credit['student_log_score_sum'])
    for sample in native:
        add_pair(sample['case'],sample['text'],'native-teacher-answer')
    kwargs=_engine_kwargs(path,torch.cuda.get_device_name(0),'default',None)
    kwargs.update(dtype='bfloat16',context_length=16384,disable_radix_cache=True,
        chunked_prefill_size=-1,disable_cuda_graph=True)
    engine=sglang.Engine(**kwargs)
    try:
        outputs=engine.generate(input_ids=[job['full_ids'] for job in jobs],
            sampling_params={'max_new_tokens':0},return_logprob=True,logprob_start_len=0)
        for job,out in zip(jobs,outputs):
            entries=out['meta_info']['input_token_logprobs']
            assert [int(v[1]) for v in entries]==job['full_ids']
            scores=[float(v[0]) for v in entries[len(job['prefix']):-1]]
            job.update(log_scores=scores,total_log_score=sum(scores),teacher_tokens=len(scores))
            job.pop('full_ids');job.pop('prefix')
        pairs=[]
        for auto,correct in zip(jobs[::2],jobs[1::2]):
            assert auto['text']==correct['text'] and auto['case_id']==correct['case_id']
            row={'case_id':auto['case_id'],'kind':auto['kind'],'auto_teacher_tokens':auto['teacher_tokens'],
                'serialized_teacher_tokens':correct['teacher_tokens'],'auto_log_score':auto['total_log_score'],
                'serialized_log_score':correct['total_log_score'],
                'log_score_gain_same_text':correct['total_log_score']-auto['total_log_score']}
            if auto['student_tokens']:
                row.update(student_tokens=auto['student_tokens'],
                    auto_credit_mean=(auto['total_log_score']-auto['student_log_score'])/auto['student_tokens'],
                    serialized_credit_mean=(correct['total_log_score']-auto['student_log_score'])/auto['student_tokens'])
            pairs.append(row)
        result={'server_args':_server_info(engine),'pairs':pairs,'jobs':jobs,
            'scope':'Same native AutoTokenizer prompt IDs held fixed. Only response token path changes; no model parameter update or collapse trajectory.'}
        (root/'teacher-token-path-scores.json').write_text(json.dumps(result,indent=2))
        print('TEACHER_TOKEN_PATH_PAIRS_JSON='+json.dumps(pairs),flush=True)
    finally:engine.shutdown()


if __name__=='__main__':main()
