#!/usr/bin/env python3
"""Campaign adapters to existing audited train/probe/eval entrypoints."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/evaluation'))
def execute(argv,**kw):subprocess.run([str(x) for x in argv],check=True,**kw)
def main(a):
    work=a.work.resolve(); cfg=json.loads((work/'config.json').read_text())
    host=ROOT/'experiments/runai/python-b200-host.sh'
    trainenv=dict(os.environ,MP_ALGORITHM='mp_opd',MP_STUDENT_PATH=cfg['student'],MP_TEACHER_PATH=cfg['teacher'],MP_DATASET_PATH=cfg['dataset'],MP_MAX_SPAN_LENGTH='2',MP_FIXED_SPAN_LENGTH='2',MP_PARTITION_SEED='43',MP_SEED='43',MP_PREFLIGHT_ONLY='0')
    if a.action.startswith('guide-'):
        execute(['bash',host,ROOT/'experiments/runai/guide_data.py',a.action.removeprefix('guide-'),'--work',work])
    elif a.action in ('train','canary'):
        env=dict(trainenv,MP_RUN_ROOT=str(work/(a.name if a.action=='train' else 'canary-'+a.name)))
        execute(['bash',ROOT/'experiments/runai/run_single_gpu.sh',a.gpu,a.name,0 if a.action=='train' else 5],env=env)
    elif a.action=='prepare-probe':
        argv=['bash',host,ROOT/'experiments/mp_opd/real_oracle.py','prepare','--input',cfg['reference_input'],'--reference-key',cfg['reference_key'],'--groups','32','--seed','20260910','--select-references','4','--eval-references','4','--conflicting-references','exclude','--reference-provenance',cfg['scope'],'--output',work/'probe-data.json']
        for x in cfg['exclude']:argv+=['--exclude-prompts',x]
        execute(argv)
    elif a.action=='probe':
        execute(['bash',host,ROOT/'experiments/mp_opd/real_oracle.py','run','--student',cfg['student'],'--teacher',cfg['teacher'],'--data',work/'probe-data.json','--output',work/a.name,'--adapter-module','model.layers.25.self_attn.q_proj','--device','cuda:0','--rank','4','--virtual-lr',a.lr,'--model-dtype','float32','--max-span','2','--weighting-steps',os.environ.get('PROBE_WEIGHTING_STEPS','1'),'--select-counts','1','4','--max-new-tokens','128','--max-reference-tokens','2048'])
    elif a.action=='adaptive':
        decision=json.loads((work/'decision.json').read_text())
        if decision['branch']=='weighting-sensitivity':
            a.action='probe';a.name='probe-weighting4';a.lr='.1';os.environ['PROBE_WEIGHTING_STEPS']='4';main(a)
        else:
            execute(['bash',ROOT/'experiments/runai/run_single_gpu.sh',a.gpu,'random','50'],env=dict(trainenv,MP_RUN_ROOT=str(work/'random-pilot')))
    elif a.action=='decision':
        # Prequential weighting is NOT a trained partition energy model.
        # Never promote its checkpoint to soft-mode full training.
        reports={}
        for p in work.glob('probe-*/summary.json'):reports[p.parent.name]=json.loads(p.read_text())
        primary=reports.get('probe-1',{});c=primary.get('comparisons',{}).get('atomic',{});ci=c.get('exploratory_group_bootstrap_95pct') or [-1,-1]
        controls={k:primary.get('comparisons',{}).get(k,{}) for k in ('atomic','fixed','skip')}
        signal=primary.get('valid_groups',0)>=24 and primary.get('invalid_groups',99)<=8 and all((v.get('exploratory_group_bootstrap_95pct') or [-1])[0]>0 and v.get('positive_fraction',0)>=.6 for v in controls.values())
        decision={'branch':'weighting-sensitivity' if signal else 'random-pilot', 'gate':'primary lr=.1; >=24 valid groups; <=8 invalid; lower CI>0 and positive fraction>=.6 against EACH atomic/fixed/skip; exploratory only', 'scope':cfg['scope'],'reports':reports,'full_learned_training':'blocked','reason':'No qualified learned-partition training launcher; weighting probe is not interchangeable with partition energy','next':'audit paired guide gains and learned weighting; do not select using benchmark test'}
        (work/'decision.json').write_text(json.dumps(decision,indent=2))
        print(json.dumps(decision))
    elif a.action=='eval-plan':
        import queue_data as D,contract_eval as E
        old=json.loads(Path(cfg['eval_template']).read_text())
        runs=list((work/a.name).glob('*/checkpoint/run-summary.json'))
        if len(runs)!=1:raise ValueError('expected one finished run')
        summary=json.loads(runs[0].read_text())
        if summary['status']!='completed' or summary['optimizer_updates']!=312:raise ValueError('incomplete training')
        cp=runs[0].parent/'step312'
        out=work/('eval-'+a.name);out.mkdir(exist_ok=True)
        old.update(jobs=[dict(id=a.name+'-seed43-312',mode=a.name,step=312,tier=0,checkpoint=E.checkpoint_identity(cp))],source=D.script_hashes(),hours=max(.01,(float(os.environ['CAMPAIGN_DEADLINE'])-time.time())/3600),admit_hours=max(.005,(float(os.environ['CAMPAIGN_DEADLINE'])-time.time())/3600))
        for d in old['data'].values():
            if E.file_hash(d['path'])!=d['sha256']:raise ValueError('eval data changed')
        E.write_new(out/'plan.json',old)
        import eval_queue as Q
        qualification=Q.preflight('/usr/bin/python3.12')
        with Q.locked(out/'state.lock'):
            state=Q.read_state(out,E.file_hash(out/'plan.json'),old)
            state['qualification']=qualification
            Q.atomic_json(out/'state.json',state)
    elif a.action in ('generate','score'):
        plan=work/('eval-'+a.name)/'plan.json'
        argv=[sys.executable,ROOT/'scripts/evaluation/eval_queue.py']
        if a.action=='generate':argv+=['worker','--phase','generate','--gpu',a.gpu,'--concurrency','128','--score-buffer','256']
        else:argv+=['score-spool']
        execute(argv+['--plan',plan,'--internal-code-execution'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--work',type=Path,required=True);p.add_argument('--name',default='');p.add_argument('--gpu',default='0');p.add_argument('--lr',default='.1');main(p.parse_args())
