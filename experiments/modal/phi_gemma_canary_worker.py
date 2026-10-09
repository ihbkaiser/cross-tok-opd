"""Two-update production trainer canary on pinned base models and fixture prompts."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path('/runs')

def main():
    gates={}
    for name in ('gemma-short','gemma-long','phi-short'):
        path=Path('/probes')/(name+'.json')
        data=json.loads(path.read_text())
        metric=data['comparisons']['sglang_default_decode_vs_hf_eager_temperature']
        assert metric['mean']<=0.1 and metric['p99']<=0.5, (name,metric)
        gates[name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'parity':metric}
    tests=['tests/test_simct_paper_scores.py','tests/test_span_ctkd_metrics.py',
           'tests/test_trajectory.py','tests/test_exact_trajectory_integration.py']
    with (ROOT/'preflight.log').open('x') as output:
        checked=subprocess.run([sys.executable,'-m','pytest','-q',*tests],stdout=output,
                               stderr=subprocess.STDOUT,timeout=180)
    print((ROOT/'preflight.log').read_text(),flush=True)
    if checked.returncode:raise RuntimeError('CPU regression preflight failed')
    from datasets import Dataset
    prompts=[
        'Solve carefully: A steel bar is 20 kg heavier than a 90 kg copper bar and weighs twice as much as a tin bar. Find the total mass of 20 bars of each metal.',
        'Write a Python function for the longest strictly increasing subsequence. Explain complexity and include tests.',
    ]
    dataset=Dataset.from_list([{'messages':[{'role':'user','content':prompts[i%2]}]} for i in range(128)])
    dataset.to_parquet(str(ROOT/'fixture-prompts.parquet'))
    opts=dict(num_nodes=1,num_gpus_per_node=1,backend='fsdp2',
        student_name_or_path='/assets/student',teacher_name_or_path='/probes/teacher',
        attn_implementation='sdpa',num_epochs=2,train_batch_size=64,micro_train_batch_size=4,
        learning_rate=5e-7,lr_warmup_ratio=.05,lr_scheduler='cosine_with_min_lr',min_lr=0,
        weight_decay=0.,gradient_checkpointing=True,enable_sleep=True,bf16=True,seed=42,
        save_path='/runs/checkpoint',ckpt_path='/runs/checkpoints',
        train_dataset_path='/runs/fixture-prompts.parquet',input_key='messages',apply_chat_template=True,
        enable_thinking=False,max_samples=128,prompt_max_len=0,max_len=512,preprocess_num_workers=2,
        rollout_num_engines=1,rollout_disable_piecewise_cuda_graph=True,rollout_tp_size=1,
        rollout_mem_fraction_static=.25,rollout_batch_size=64,generate_max_len=128,
        n_samples_per_prompt=1,temperature=.6,top_p=.95,teacher_tp_size=1,teacher_pp_size=1,
        teacher_ep_size=1,teacher_dp_size=1,teacher_mem_fraction_static=.3,teacher_context_length=16384,
        teacher_forward_n_batches=8,kd_algorithm='span_ctkd',kd_loss_fn='rkl',kd_ratio=1.,
        span_score_mode='mean_logprob',exact_token_trajectory=True,enforce_max_sequence_length=True,
        diagnostic_max_updates=2,diagnostic_collapse_gate=True,save_steps=999999,logging_steps=1,
        use_wandb=False)
    command=[sys.executable,'-m','kdflow.cli.train_kd_on_policy']
    for key,value in opts.items():command+=['--'+key,str(value)]
    sources={str(p.relative_to('/opt/debug')):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(Path('/opt/debug/kdflow').rglob('*.py'))}
    (ROOT/'invocation.json').write_text(json.dumps({'gates':gates,'command':command,'source_hashes':sources,
        'scope':'base-model fixture canary, not company SFT reproduction or accuracy experiment'},indent=2))
    subprocess.run(command,check=True)
    summary=json.loads((ROOT/'checkpoint/run-summary.json').read_text())
    assert summary['optimizer_updates']==2 and summary['status']=='completed', summary
    print('PAIRED_CANARY_SUMMARY_JSON='+json.dumps(summary),flush=True)

if __name__=='__main__':main()
