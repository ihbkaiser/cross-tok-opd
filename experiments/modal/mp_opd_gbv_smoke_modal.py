"""Bounded GBV-Span smoke: runtime unit tests, then real `gbv` steps per geometry.

Purpose: does `mp_opd_mode=gbv` execute end to end on the B200 runtime, and does its
telemetry reach the trainer log. **Not a result.** The model pair and the prompt set
exist only to make the path run: a small Qwen2.5 student and a slightly larger Qwen2.5
teacher of the same family (same tokenizer, non-gated, cheap to fetch), plus eight
short questions written here. Nothing about variance reduction, gradient reversals or
benchmark quality can be read off this run.

Why not the staged `google/gemma-2-2b-it` pair: that volume lives in the
`phamvanvuhoan` Modal workspace, which currently has no B200 entitlement, so this
smoke runs under `kieusontung8` and fetches its own weights.

Cost control: two short trainings in one B200 container, two optimizer updates each,
W&B off, no checkpoint writes.
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/opt/overlay')
PYTHON = '/opt/venvs/simct-b200/bin/python'
IMAGE = 'docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f'
RUN = 'mp-opd-gbv-smoke-20261002-r1'
STUDENT = 'Qwen/Qwen2.5-0.5B-Instruct'
TEACHER = 'Qwen/Qwen2.5-1.5B-Instruct'
GEOMETRIES = ('token_count', 'exact_logit')
UPDATES = 2
SAMPLES = 8
MAX_SPAN = 4
BETA = 1.0
# Telemetry the method note names and that the mode must therefore emit.
GBV_KEYS = (
    'mp_opd_gbv_total_cost',
    'mp_opd_gbv_distortion_term',
    'mp_opd_gbv_dof_term',
    'mp_opd_gbv_retained_dof_fraction',
    'mp_opd_gbv_selected_span_count',
    'mp_opd_gbv_selected_span_length_mean',
    'mp_opd_gbv_boundary_strength_mean',
    'mp_opd_gbv_degenerate',
    'mp_opd_gbv_beta',
    'mp_opd_gbv_exact_geometry',
)
PROMPTS = [
    'What is 17*23? Show the arithmetic.',
    'Solve for x: 3x + 7 = 25.',
    'What is the sum of the first ten positive integers?',
    'Write a Python function that reverses a string.',
    'What is the greatest common divisor of 84 and 132?',
    'Simplify (x^2 - 9)/(x - 3).',
    'Write a Python one-liner that counts vowels in a string.',
    'What is 2^10 divided by 8?',
]

app = modal.App(RUN)
image = (
    modal.Image.from_registry(IMAGE)
    .entrypoint([])
    .env({'PYTHONPATH': '/opt/overlay/experiments/modal/vendor:/opt/overlay',
          'PYTHONUNBUFFERED': '1'})
    .add_local_dir(str(ROOT / 'kdflow'), '/opt/overlay/kdflow',
                   ignore=['**/__pycache__/**', '**/*.pyc'])
    .add_local_dir(str(ROOT / 'tests'), '/opt/overlay/tests',
                   ignore=['**/__pycache__/**', '**/*.pyc'])
    # The algorithm registry imports every algorithm, and xtoken needs the vendored
    # aligner from experiments/modal/vendor, so this directory is not optional.
    .add_local_dir(str(ROOT / 'experiments/modal'), '/opt/overlay/experiments/modal',
                   ignore=['**/__pycache__/**', '**/*.pyc'])
)
assets = modal.Volume.from_name(RUN + '-assets', create_if_missing=True)
outputs = modal.Volume.from_name(RUN, create_if_missing=True)


def env(online=False):
    e = dict(os.environ)
    e.update(
        PATH='/opt/venvs/simct-b200/bin:' + e.get('PATH', ''),
        PYTHONPATH='/opt/overlay/experiments/modal/vendor:/opt/overlay',
        HF_HOME='/assets/hf',
        HF_HUB_OFFLINE='0' if online else '1',
        HF_DATASETS_OFFLINE='0' if online else '1',
        TRANSFORMERS_OFFLINE='0' if online else '1',
        TOKENIZERS_PARALLELISM='false',
        RAY_USAGE_STATS_ENABLED='0',
        NCCL_CUMEM_HOST_ENABLE='0',
        OMP_NUM_THREADS='4',
        WANDB_SILENT='true',
        KDFLOW_TRUST_REMOTE_CODE='0',
    )
    libs = ['/usr/local/cuda/lib64', '/usr/local/nvidia/lib64'] + [
        str(p) for p in Path('/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia').glob('*/lib')
    ]
    e['LD_LIBRARY_PATH'] = ':'.join(libs)
    return e


@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={'/runs': outputs})
def unit_tests(commit: str):
    """The new module must pass under the real runtime torch, not only local CPU torch."""
    root = Path('/runs')
    root.mkdir(exist_ok=True)
    e = env()
    e['KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT'] = '1'
    command = [PYTHON, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
               'tests/mp_opd/test_gbv_span.py',
               'tests/mp_opd/test_credit_oracle.py',
               'tests/mp_opd/test_random_partition_min_span.py']
    done = subprocess.run(command, cwd='/opt/overlay', env=e, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1700)
    (root / 'unit-tests.log').write_text(done.stdout)
    outputs.commit()
    tail = done.stdout.strip().splitlines()[-3:]
    print('GBV_UNIT_TAIL=' + json.dumps(tail), flush=True)
    print('GBV_UNIT_JSON=' + json.dumps({'commit': commit, 'returncode': done.returncode,
                                         'passed': done.returncode == 0}), flush=True)
    if done.returncode:
        raise RuntimeError('runtime unit tests failed')


@app.function(image=image, gpu='B200', cpu=16, memory=98304, timeout=3000, retries=0,
              max_containers=1, volumes={'/assets': assets, '/runs': outputs})
def train_smoke(commit: str, geometry: str, updates: int):
    root = Path('/runs')
    root.mkdir(exist_ok=True)
    log_path = root / ('train-' + geometry + '.log')
    if log_path.exists():
        raise RuntimeError('existing attempt requires inspection; no duplicate retry')

    student = Path('/assets/student')
    os.environ['HF_HOME'] = '/assets/hf'
    os.environ['HF_HUB_OFFLINE'] = '0'
    if not (student / 'config.json').is_file():
        student.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(STUDENT, local_dir=str(student))
        assets.commit()
    import pandas as pd
    from huggingface_hub import snapshot_download
    prompts = Path('/assets/smoke-prompts.parquet')
    pd.DataFrame({'messages': [[{'role': 'user', 'content': text}] for text in PROMPTS]}).to_parquet(prompts)
    teacher = Path('/assets/teacher')
    if not (teacher / 'config.json').is_file():
        teacher.mkdir(parents=True, exist_ok=True)
        snapshot_download(TEACHER, local_dir=str(teacher))
        assets.commit()
    print('SMOKE_ASSETS=' + json.dumps({'student': str(student), 'teacher': str(teacher),
                                        'prompts': str(prompts)}), flush=True)

    opts = dict(
        num_nodes=1, num_gpus_per_node=1, backend='fsdp2',
        student_name_or_path=str(student), teacher_name_or_path=str(teacher),
        attn_implementation='sdpa', num_epochs=updates, train_batch_size=SAMPLES,
        micro_train_batch_size=1, learning_rate=1e-6, lr_warmup_ratio=.05,
        lr_scheduler='cosine_with_min_lr', min_lr=0, weight_decay=0.,
        gradient_checkpointing=True, enable_sleep=True, bf16=True, seed=42,
        # Per-geometry output dirs: the trainer fails closed when a checkpoint already
        # exists, so sharing one save_path across geometries would abort the second run.
        save_path='/runs/checkpoint-' + geometry, ckpt_path='/runs/checkpoints-' + geometry,
        train_dataset_path=str(prompts), input_key='messages', apply_chat_template=True,
        enable_thinking=False, max_samples=SAMPLES, prompt_max_len=0, max_len=2048,
        preprocess_num_workers=2, rollout_num_engines=1,
        rollout_disable_piecewise_cuda_graph=True, rollout_tp_size=1,
        rollout_mem_fraction_static=.25, rollout_batch_size=SAMPLES,
        generate_max_len=1024, n_samples_per_prompt=1, temperature=.6, top_p=.95,
        teacher_tp_size=1, teacher_pp_size=1, teacher_ep_size=1, teacher_dp_size=1,
        teacher_mem_fraction_static=.3, teacher_context_length=4096,
        teacher_forward_n_batches=8, kd_algorithm='mp_opd', mp_opd_mode='gbv',
        mp_opd_max_span_length=MAX_SPAN, mp_opd_gbv_beta=BETA,
        mp_opd_gbv_geometry=geometry, kd_loss_fn='rkl', kd_ratio=1.,
        span_score_mode='mean_logprob', exact_token_trajectory=True,
        enforce_max_sequence_length=True, diagnostic_max_updates=updates,
        diagnostic_collapse_gate=False, save_steps=1000, logging_steps=1,
        use_wandb=False,
    )
    command = [PYTHON, '-m', 'kdflow.cli.train_kd_on_policy']
    for key, value in opts.items():
        command += ['--' + key, str(value)]
    (root / ('invocation-' + geometry + '.json')).write_text(json.dumps(dict(
        purpose='gbv code-path smoke, not a result', source_commit=commit, geometry=geometry,
        teacher=TEACHER, student=str(student), updates=updates, samples=SAMPLES,
        max_span_length=MAX_SPAN, beta=BETA, command=command), indent=2))
    outputs.commit()

    process = None
    try:
        with log_path.open('x') as handle:
            process = subprocess.Popen(command, cwd='/opt/overlay', env=env(online=True),
                                       stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            started = time.monotonic()
            while process.poll() is None:
                outputs.commit()
                if time.monotonic() - started > 2400:
                    raise TimeoutError('smoke time cap')
                time.sleep(15)
        text = log_path.read_text()
        if process.returncode:
            print('SMOKE_LOG_TAIL=' + json.dumps(text.strip().splitlines()[-15:]), flush=True)
            raise RuntimeError('training failed rc=' + str(process.returncode))
        seen = {}
        for key in GBV_KEYS:
            found = re.findall(re.escape(key) + r':\s*([-\d.eE+]+)', text)
            if found:
                seen[key] = float(found[-1])
        missing = [key for key in GBV_KEYS if key not in seen]
        summary = {'commit': commit, 'geometry': geometry, 'updates': updates,
                   'metrics_seen': seen, 'missing': missing,
                   'note': 'code-path smoke only; no efficacy claim'}
        (root / ('gbv-smoke-' + geometry + '.json')).write_text(json.dumps(summary, indent=2))
        outputs.commit()
        print('GBV_SMOKE_JSON=' + json.dumps(summary), flush=True)
        if missing:
            raise RuntimeError('gbv telemetry missing: ' + ','.join(missing))
        return summary
    finally:
        if process and process.poll() is None:
            import signal
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        outputs.commit()


@app.local_entrypoint()
def main(updates: int = UPDATES, geometries: str = ','.join(GEOMETRIES)):
    dirty = subprocess.check_output(
        ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError('Commit tracked changes first')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    print('SMOKE_SOURCE_COMMIT=' + commit, flush=True)
    unit_tests.remote(commit)
    results = []
    for geometry in [g.strip() for g in geometries.split(',') if g.strip()]:
        if geometry not in GEOMETRIES:
            raise ValueError('unknown geometry ' + geometry)
        results.append(train_smoke.remote(commit, geometry, updates))
    print('GBV_SMOKE_ALL=' + json.dumps(results), flush=True)
