"""CPU-only token/text attribution of the qualified teacher scores."""
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    root=Path('/results')
    teacher=json.loads((root/'teacher-hidden-head.json').read_text())
    tok=AutoTokenizer.from_pretrained('/teacher_assets/teacher',local_files_only=True)
    rows=[]
    for sample in teacher['samples']:
        ids=sample['content_ids'];scores=sample['direct_sglang_raw']
        text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        token_rows=[]
        for i,(tid,score) in enumerate(zip(ids,scores)):
            before=tok.decode(ids[:i],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            after=tok.decode(ids[:i+1],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            token_rows.append({'position':i,'token_id':tid,'logprob':score,
                'decoded_piece':after[len(before):] if after.startswith(before) else tok.decode([tid]),
                'context':after[-80:]})
        code_start=text.find('def ')
        header=[row for row in token_rows if len(tok.decode(ids[:row['position']+1]))<=code_start] if code_start>=0 else []
        rows.append({'case_id':sample['case_id'],'text':text,
            'teacher_total_nll':-sum(scores),'code_header_nll':-sum(row['logprob'] for row in header),
            'code_header_token_count':len(header),
            'first_12_teacher_tokens':token_rows[:12],
            'worst_teacher_tokens':sorted(token_rows,key=lambda row:row['logprob'])[:10]})
    (root/'teacher-score-text-readout.json').write_text(json.dumps({'rows':rows,
        'scope':'Text attribution of short qualified Phi teacher scores; no optimization or standalone correctness score'},indent=2))
    print('TEACHER_SCORE_TEXT_JSON='+json.dumps([{k:v for k,v in row.items() if k not in ('text','worst_teacher_tokens','first_12_teacher_tokens')} for row in rows]),flush=True)


if __name__=='__main__':main()
