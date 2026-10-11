"""Audit the loaded tokenizer against the model's pinned tokenizer.json backend."""
import hashlib
import inspect
import json
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import transformers


def main():
    root=Path('/results');path=Path('/teacher_assets/teacher')
    auto=AutoTokenizer.from_pretrained(path,local_files_only=True)
    direct=Tokenizer.from_file(str(path/'tokenizer.json'))
    fast=PreTrainedTokenizerFast.from_pretrained(path,local_files_only=True)
    specimens=['(n',"Let's",'def factorial(n):\n','\n\n','s.index(char)']
    probes=[]
    for text in specimens:
        probes.append({'text':text,'auto':auto.encode(text,add_special_tokens=False),
            'serialized_backend':direct.encode(text,add_special_tokens=False).ids,
            'explicit_fast':fast.encode(text,add_special_tokens=False)})
    samples=json.loads((root/'phi_teacher-baseline.json').read_text())['samples'];rows=[]
    for sample in samples:
        native=sample['output_ids'].copy();matched=sample['finish_reason'].get('matched')
        while native and (native[-1] in auto.all_special_ids or native[-1]==matched):native.pop()
        text=auto.decode(native,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        canonical=auto.encode(text,add_special_tokens=False);serialized=direct.encode(text,add_special_tokens=False).ids
        explicit=fast.encode(text,add_special_tokens=False)
        rows.append({'case_id':sample['case']['id'],'native_tokens':len(native),'auto_tokens':len(canonical),
            'serialized_tokens':len(serialized),'auto_matches_native':canonical==native,
            'serialized_matches_native':serialized==native,'explicit_matches_serialized':explicit==serialized,
            'text_matches_output':text==sample['text'],'native_ids':native,'auto_ids':canonical,'serialized_ids':serialized})
    constructor,start=inspect.getsourcelines(type(auto).__init__)
    source={str(start+i):line.strip() for i,line in enumerate(constructor) if any(k in line for k in ('pre_tokenizer','Split','pattern','regex'))}
    loaded=json.loads(auto.backend_tokenizer.to_str())
    serialized_json=json.loads((path/'tokenizer.json').read_text())
    result={'transformers':transformers.__version__,'auto_class':type(auto).__name__,
        'explicit_class':type(fast).__name__,'constructor_file':inspect.getsourcefile(type(auto)),
        'constructor_excerpt':source,'tokenizer_json_sha256':hashlib.sha256((path/'tokenizer.json').read_bytes()).hexdigest(),
        'auto_pre_tokenizer':loaded['pre_tokenizer'],'serialized_pre_tokenizer':serialized_json['pre_tokenizer'],
        'explicit_pre_tokenizer':json.loads(fast.backend_tokenizer.to_str())['pre_tokenizer'],
        'probes':probes,'rows':rows,'auto_roundtrip_mismatches':sum(not r['auto_matches_native'] for r in rows),
        'serialized_roundtrip_mismatches':sum(not r['serialized_matches_native'] for r in rows),
        'scope':'CPU structural/runtime audit, explicit finished-control stripping; no new model generation'}
    (root/'tokenizer-backend-audit.json').write_text(json.dumps(result,indent=2))
    print('TOKENIZER_BACKEND_AUDIT_JSON='+json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
