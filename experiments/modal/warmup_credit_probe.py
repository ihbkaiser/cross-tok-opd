"""Frozen actual-Phi-SFT credit and stop-logit gradients through production loss.

Uses the exact teacher token scores already qualified against its hidden/head API.
No optimizer update or training-trajectory claim is made by this probe.
"""
import hashlib
import json
import os
from pathlib import Path

os.environ['KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT'] = '1'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kdflow.algorithms.mp_opd import MetaPartitionedOPD
from kdflow.algorithms._mp_opd_atoms import SimCTAtomizer, mp_content_ids
from kdflow.algorithms._mp_opd_credit import AtomCreditTensors, soft_partition_loss
from kdflow.algorithms._mp_opd_grass_span import GrassNoiseEstimator

ROOT = Path('/results')


def main():
    baseline = json.loads((ROOT / 'phi-baseline.json').read_text())
    teacher = json.loads((ROOT / 'teacher-hidden-head.json').read_text())
    assert teacher['hidden_head_vs_direct']['max'] == 0
    assets = json.loads(Path('/warmup/ready.json').read_text())
    path = assets['models']['phi']
    student_tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    teacher_tok = AutoTokenizer.from_pretrained('/teacher_assets/teacher', local_files_only=True)
    atomizer = SimCTAtomizer(student_tok, teacher_tok)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='eager').to('cuda').eval()
    algorithm = MetaPartitionedOPD.__new__(MetaPartitionedOPD)
    algorithm.grass_chunk_source = 'run'
    algorithm.grass_chunk_run_length = 2
    algorithm.grass_geometry = 'exact_head'
    algorithm.grass_softcap = float(model.config.final_logit_softcapping)
    algorithm.grass_head_bias = model.get_output_embeddings().bias is not None
    algorithm.grass_eps_d = 0.
    algorithm.grass_negative_tol_rel = 1e-6
    algorithm.grass_chunk_shadow = False
    algorithm.grass_noise = GrassNoiseEstimator(rho=.99, min_adjacent_pairs=8)
    algorithm.mode = 'grass_chunk'
    rows = []
    for sample, teacher_sample in zip(baseline['samples'][:4], teacher['samples']):
        assert sample['case']['id'] == teacher_sample['case_id']
        student_ids, masked = mp_content_ids(sample['output_ids'][:96], student_tok)
        teacher_ids = teacher_sample['content_ids']
        atoms = atomizer.atomize(student_ids, teacher_ids, sample_id=sample['case']['id'])
        assert atoms.valid, atoms.failure_reason
        assert atoms.covered_student_events == len(student_ids)
        assert atoms.covered_teacher_events == len(teacher_ids)
        prefix = sample['prompt_ids']
        with torch.no_grad():
            output = model(torch.tensor([prefix + student_ids], device='cuda'),
                use_cache=False, output_hidden_states=True)
            logits = output.logits[0, len(prefix)-1:len(prefix)+len(student_ids)-1].detach().clone().requires_grad_(True)
            hidden = output.hidden_states[-1][0, len(prefix)-1:len(prefix)+len(student_ids)-1].detach()
        labels = torch.tensor(student_ids, device='cuda')
        student_scores = logits.float().log_softmax(-1).gather(-1, labels[:,None]).squeeze(-1)
        teacher_scores = torch.tensor(teacher_sample['direct_sglang_raw'], device='cuda')
        l_t = torch.stack([teacher_scores[a.teacher_start:a.teacher_end].sum() for a in atoms.atoms]).detach()
        l_s = torch.stack([student_scores[a.student_start:a.student_end].sum() for a in atoms.atoms])
        weights = torch.tensor([a.student_token_count for a in atoms.atoms], device='cuda', dtype=torch.float32)
        base = l_t - l_s.detach()
        rate = (base / weights).detach()
        credits = AtomCreditTensors(l_t,l_s.detach(),base,weights,rate,-l_s,-student_scores)
        atomic = soft_partition_loss(credits.current_nll, rate)
        grass, metrics = algorithm._grass_chunk_loss(credits, atoms.atoms, student_logits=logits,
            student_labels=labels, student_hidden=hidden, student_ids=student_ids, teacher_ids=teacher_ids)
        grad_atomic = torch.autograd.grad(atomic, logits, retain_graph=True)[0]
        grad_grass = torch.autograd.grad(grass, logits)[0]
        # Descent increases an unobserved stop logit iff its loss gradient is negative.
        def gradient_summary(grad):
            stop = grad[:,[1,107]].float().sum(-1)
            return {'rows':len(student_ids),'stop_logit_descent_increases_fraction':float((stop<0).float().mean()),
                'stop_gradient_mean':float(stop.mean()),'stop_gradient_first_row':float(stop[0]),
                'stop_gradient_min':float(stop.min()),'stop_gradient_max':float(stop.max())}
        row = {'case_id':sample['case']['id'],'student_tokens':len(student_ids),'teacher_tokens':len(teacher_ids),
            'atoms':len(atoms.atoms),'truncated_prefix':len(sample['output_ids'])>96,'masked_sampled_terminal':masked,
            'credit_weighted_mean':float(base.sum()/weights.sum()),'credit_atom_mean':float(rate.mean()),
            'negative_credit_atom_fraction':float((rate<0).float().mean()),'teacher_log_score_sum':float(l_t.sum()),
            'student_log_score_sum':float(l_s.detach().sum()),'atomic':gradient_summary(grad_atomic),
            'grass_run2':gradient_summary(grad_grass),'production_grass_metrics':{k:float(v) for k,v in metrics.items()},
            'first_atom_rate':float(rate[0]),'first_atom_weight':float(weights[0])}
        rows.append(row)
        print('FROZEN_CREDIT_JSON='+json.dumps(row),flush=True)
        del output,logits,hidden,student_scores,credits,grad_atomic,grad_grass
        torch.cuda.empty_cache()
    source = Path('/opt/warmup_debug/kdflow')
    result = {'scope':'Frozen first <=96-token prefixes from four shared diagnostic prompts; exact production atomizer and GRASS-Chunk run2 loss. No parameter update, no historical EMA state, no batch64 trajectory or causal proof of collapse.',
        'teacher_scores_sha256':hashlib.sha256((ROOT/'teacher-hidden-head.json').read_bytes()).hexdigest(),
        'source_sha256':{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*.py'))},
        'rows':rows}
    (ROOT/'frozen-credit-probe.json').write_text(json.dumps(result,indent=2))


if __name__ == '__main__': main()
