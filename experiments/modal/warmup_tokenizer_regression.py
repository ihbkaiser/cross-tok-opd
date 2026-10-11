"""Offline remote regression qualification for the explicit serialized backend loader."""
import json
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def main():
    path='/teacher_assets/teacher';root=Path('/results')
    pipeline=AutoTokenizer.from_pretrained(path,use_fast=True,trust_remote_code=False,local_files_only=True)
    candidate=PreTrainedTokenizerFast.from_pretrained(path,local_files_only=True)
    serialized=Tokenizer.from_file(path+'/tokenizer.json')
    checks=[]
    for text in ['(n',"Let's",'def factorial(n):\n','s.index(char)','\n\n']:
        got=candidate.encode(text,add_special_tokens=False);expected=serialized.encode(text,add_special_tokens=False).ids
        assert got==expected,(text,got,expected)
        assert candidate.decode(got,clean_up_tokenization_spaces=False)==text
        checks.append('encode/decode serialized: '+repr(text))
    for text,tid in [('<|user|>',200021),('<|assistant|>',200019),('<|end|>',200020),('<|endoftext|>',199999)]:
        assert candidate.encode(text,add_special_tokens=False)==[tid]
        checks.append('control id: '+text)
    messages=[{'role':'system','content':'Be precise.'},{'role':'user','content':"Write factorial(n). Let's test it."}]
    rendered=candidate.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    assert rendered==pipeline.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    assert candidate.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)['input_ids']==serialized.encode(rendered,add_special_tokens=False).ids
    checks.append('chat template rendering and serialized prompt ids')
    assert candidate.eos_token_id==pipeline.eos_token_id==199999
    assert candidate.pad_token_id==pipeline.pad_token_id==199999
    checks.append('eos/pad unchanged')
    result={'checks_passed':len(checks),'checks':checks,'candidate_class':type(candidate).__name__,
        'pipeline_class_explicit_use_fast_true':type(pipeline).__name__,
        'pipeline_n_ids':pipeline.encode('(n',add_special_tokens=False),
        'candidate_n_ids':candidate.encode('(n',add_special_tokens=False),
        'candidate_chat_template':rendered,
        'production_code_changed':False,'scope':'Cached tokenizer and runtime only; no production loader patch or training run'}
    (root/'tokenizer-loader-regression.json').write_text(json.dumps(result,indent=2))
    print('TOKENIZER_LOADER_REGRESSION_JSON='+json.dumps(result),flush=True)


if __name__=='__main__':main()
