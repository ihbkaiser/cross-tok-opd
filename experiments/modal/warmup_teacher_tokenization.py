"""Check native teacher token paths against canonical re-encoding of their text."""
import json
from pathlib import Path
from transformers import AutoTokenizer


def main():
    root=Path('/results');tok=AutoTokenizer.from_pretrained('/teacher_assets/teacher',local_files_only=True)
    baseline=json.loads((root/'phi_teacher-baseline.json').read_text());rows=[]
    for sample in baseline['samples']:
        native=sample['output_ids'].copy()
        while native and native[-1] in tok.all_special_ids: native.pop()
        text=tok.decode(native,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        canonical=tok.encode(text,add_special_tokens=False)
        common=0
        for left,right in zip(native,canonical):
            if left!=right:break
            common+=1
        row={'case_id':sample['case']['id'],'native_tokens':len(native),'canonical_tokens':len(canonical),
            'exact_token_path_match':native==canonical,'common_prefix_tokens':common,
            'text_matches_native_output':text==sample['text'],
            'canonical_text_lossless':tok.decode(canonical,skip_special_tokens=False,clean_up_tokenization_spaces=False)==text,
            'native_first_24':[{'id':v,'piece':tok.decode([v])} for v in native[:24]],
            'canonical_first_24':[{'id':v,'piece':tok.decode([v])} for v in canonical[:24]]}
        rows.append(row)
    result={'scope':'Native Phi samples re-encoded with the exact same cached tokenizer',
        'path_mismatch_samples':sum(not r['exact_token_path_match'] for r in rows),'samples':len(rows),'rows':rows}
    (root/'teacher-tokenization-roundtrip.json').write_text(json.dumps(result,indent=2))
    print('TEACHER_TOKENIZATION_JSON='+json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    for r in rows[:4]: print('TEACHER_TOKENIZATION_ROW='+json.dumps(r),flush=True)


if __name__=='__main__':main()
