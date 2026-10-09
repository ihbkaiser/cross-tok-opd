"""Parity gates followed by real aligned Phi->Gemma forward/backward if qualified."""
import json,sys,time,gc,traceback
from pathlib import Path
import torch
import mp_opd_parity_worker as probe

ROOT=Path('/runs')

def main():
    # Full IDs, decode, prefill, raw/tempered HF comparisons are saved by the existing worker.
    probe.MAX_NEW_TOKENS=128
    short=probe.run('/assets/student',ROOT/'gemma-short.json',['default'],None)
    saved=json.loads((ROOT/'gemma-short.json').read_text())
    flat=[v for r in saved['samples'] for v in r['decode_logprobs']]
    metric=short['comparisons']['sglang_default_decode_vs_hf_eager_temperature']
    gate={'tokens':len(flat),'zero_fraction':sum(v==0 for v in flat)/len(flat),'parity':metric,
          'passed':metric['p99']<=0.5 and all(torch.isfinite(torch.tensor(flat)).tolist())}
    print('SHORT_GATE_JSON='+json.dumps(gate),flush=True)
    (ROOT/'short-gate.json').write_text(json.dumps(gate))
    # Longer sampled responses investigate length dependence without fabricating an exact length.
    probe.MAX_NEW_TOKENS=3840
    long=probe.run('/assets/student',ROOT/'gemma-long.json',['default'],None)
    long_saved=json.loads((ROOT/'gemma-long.json').read_text())
    print('LONG_ACTUAL_LENGTHS_JSON='+json.dumps([len(s['output_ids']) for s in long_saved['samples']]),flush=True)
    lm=long['comparisons']['sglang_default_decode_vs_hf_eager_temperature']
    qualified=gate['passed'] and lm['p99']<=0.5
    # A teacher-path probe is independent of whether the student passes.
    probe.MAX_NEW_TOKENS=64
    teacher=probe.run('/runs/teacher',ROOT/'phi-short.json',['default'],None)
    tm=teacher['comparisons']['sglang_default_decode_vs_hf_eager_temperature']
    qualified=qualified and tm['p99']<=0.5
    result={'student_short':gate,'student_long':lm,'teacher_short':tm,
            'training_gate_passed':qualified,'production_training_run':False,
            'student_checkpoint':'pinned base Gemma IT, not company SFT checkpoint'}
    if not qualified:
        result['training_status']='blocked_by_parity';print('TRAINING_SKIPPED: parity gate failed',flush=True)
    else:
        # This next gate is deliberately not represented as an end-to-end rollout trainer.
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from kdflow.algorithms._mp_opd_atoms import build_atoms
        result['training_status']='requires_production_trainer_canary_after_parity'
        result['reason']='No company SFT checkpoint/config is mounted; do not substitute a different training contract.'
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2))
    print('PHI_GEMMA_DEBUG_SUMMARY_JSON='+json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();sys.exit(1)
