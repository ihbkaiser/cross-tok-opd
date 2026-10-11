"""Audit real Qwen-to-Gemma assets; no optimizer or historical efficacy claim."""
import argparse
import hashlib
import json
from pathlib import Path

REVISION='a09a35458c702b33eeacc393d103063234e8bc28'
TEACHER=Path('/teacher_assets/teacher')
OUT=Path('/results')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def save(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2))

def prepare():
    from huggingface_hub import snapshot_download
    snapshot_download('Qwen/Qwen2.5-7B-Instruct',revision=REVISION,local_dir=TEACHER,
        token=False,allow_patterns=['*.json','*.txt','*.safetensors'],max_workers=4)
    manifest={'repo':'Qwen/Qwen2.5-7B-Instruct','revision':REVISION,
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in TEACHER.iterdir() if p.is_file()}}
    save('teacher-manifest.json',manifest)
    audit()

def audit():
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer,PreTrainedTokenizerFast
    import transformers
    assets=json.loads(Path('/warmup/ready.json').read_text())
    samples=json.loads(Path('/prior/qwen-baseline.json').read_text())['samples']
    fixtures=['(n',"Let's",'def factorial(n):\n','\n\n','s.index(char)','1234567890','café naïve Việt Nam','  x = 2\n\treturn x']
    texts=fixtures+[s['text'] for s in samples]+[s['case']['prompt'] for s in samples]
    rows=[]
    for name,path in [('qwen',TEACHER),('gemma',Path(assets['models']['qwen']))]:
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True)
        raw=Tokenizer.from_file(str(path/'tokenizer.json'))
        fast=PreTrainedTokenizerFast.from_pretrained(path,local_files_only=True)
        probes=[{'text':t,'auto':tok.encode(t,add_special_tokens=False),
            'serialized':raw.encode(t,add_special_tokens=False).ids,
            'explicit_fast':fast.encode(t,add_special_tokens=False)} for t in texts]
        rows.append({'name':name,'auto_class':type(tok).__name__,'probes':probes,
            'mismatches':sum(p['auto']!=p['serialized'] for p in probes),
            'pre_tokenizer_equal':json.loads(tok.backend_tokenizer.to_str())['pre_tokenizer']==json.loads(raw.to_str())['pre_tokenizer'],
            'chat_template_equal':tok.chat_template==fast.chat_template,'special_ids_equal':tok.all_special_ids==fast.all_special_ids})
        corpus=Path('/opt/warmup_debug/prompts.parquet')
        if corpus.exists():
            import pyarrow.parquet as pq
            requests=pq.read_table(corpus,columns=['messages']).to_pylist()
            mismatches=[]
            for i,r in enumerate(requests):
                messages=r['messages']
                rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                a=tok.encode(rendered,add_special_tokens=False);b=raw.encode(rendered,add_special_tokens=False).ids
                if a!=b:mismatches.append({'row':i,'auto_tokens':len(a),'serialized_tokens':len(b)})
            rows[-1].update(corpus_rows=len(requests),corpus_sha256=sha(corpus),corpus_mismatches=mismatches,
                auto_pre_tokenizer=json.loads(tok.backend_tokenizer.to_str())['pre_tokenizer'],
                serialized_pre_tokenizer=json.loads(raw.to_str())['pre_tokenizer'])
    tea=AutoTokenizer.from_pretrained(TEACHER,local_files_only=True)
    stu=AutoTokenizer.from_pretrained(assets['models']['qwen'],local_files_only=True)
    sv={k.replace('Ġ','▁'):v for k,v in stu.get_vocab().items()}
    tv={k.replace('Ġ','▁'):v for k,v in tea.get_vocab().items()}
    keys=sorted(set(sv)&set(tv));si=[sv[k] for k in keys];ti=[tv[k] for k in keys]
    sd=stu.batch_decode([[i] for i in si],skip_special_tokens=False,clean_up_tokenization_spaces=False)
    td=tea.batch_decode([[i] for i in ti],skip_special_tokens=False,clean_up_tokenization_spaces=False)
    bad=[{'key':k,'student_id':s,'teacher_id':t,'student_decode':a,'teacher_decode':b} for k,s,t,a,b in zip(keys,si,ti,sd,td) if a!=b]
    save('tokenizer-corpus-audit.json' if Path('/opt/warmup_debug/prompts.parquet').exists() else 'tokenizer-audit.json',{'transformers':transformers.__version__,'rows':rows,
        'raw_key_overlap':len(keys),'decoded_overlap_mismatches':len(bad),'overlap_mismatch_rows':bad,
        'scope':'Same pinned files; real runtime tokenizer encode and SimCT raw-key overlap audit; no training.'})
    print('TOKENIZER_AUDIT_SUMMARY='+json.dumps({'rows':[{k:r[k] for k in ['name','auto_class','mismatches','pre_tokenizer_equal','chat_template_equal','special_ids_equal']} for r in rows],
        'raw_key_overlap':len(keys),'decoded_overlap_mismatches':len(bad),'examples':bad[:8],
        'corpus_checks':[{k:r.get(k) for k in ['name','corpus_rows','corpus_mismatches']} for r in rows]}),flush=True)

def probe():
    from warmup_debug_worker import teacher_probe
    samples=json.loads(Path('/prior/qwen-baseline.json').read_text())['samples']
    teacher_probe(samples,path=str(TEACHER),student_arm='qwen',teacher_name='Qwen2.5-7B-Instruct')

def mapping():
    from collections import Counter,defaultdict
    from transformers import AutoTokenizer
    assets=json.loads(Path('/warmup/ready.json').read_text())
    result={}
    for name,path in [('qwen',TEACHER),('gemma',Path(assets['models']['qwen']))]:
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        grouped=defaultdict(list)
        for key,tid in tok.get_vocab().items():
            grouped[key.replace('Ġ','▁')].append({'raw_key':key,'id':tid,
                'decode':tok.decode([tid],skip_special_tokens=False,clean_up_tokenization_spaces=False)})
        collisions={k:v for k,v in grouped.items() if len(v)>1}
        observations=defaultdict(list)
        for _ in range(30):
            normalized={k.replace('Ġ','▁'):v for k,v in tok.get_vocab().items()}
            for k in collisions:observations[k].append(normalized[k])
        result[name]={'collision_count':len(collisions),'collisions':collisions,
            'winner_counts':{k:dict(Counter(ids)) for k,ids in observations.items()},
            'unstable_keys':[k for k,ids in observations.items() if len(set(ids))>1]}
    save('overlap-key-collisions.json',result)
    print('OVERLAP_COLLISION_SUMMARY='+json.dumps({k:{'collision_count':v['collision_count'],'unstable_keys':v['unstable_keys'],
        'space_candidates':v['collisions'].get('▁'),'space_winner_counts':v['winner_counts'].get('▁')} for k,v in result.items()}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','audit','probe','mapping']);a=p.parse_args()
    globals()[a.stage]()
