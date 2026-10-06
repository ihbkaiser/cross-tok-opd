"""Bounded cross-atom credit smoke: runtime unit tests, then real `kernel` steps.

Purpose: prove that the credit-transform interface of
`_mp_opd_credit_transform.py` executes end to end on the B200 runtime, that Atomic
is numerically unchanged by the refactor, and that every documented credit operator
emits its telemetry into the trainer log. **Not a result.**

The model pair and prompt set exist only to make the path run: a small Qwen2.5
student and a slightly larger Qwen2.5 teacher of the same family (same tokenizer,
non-gated, cheap to fetch) plus eight short questions written here. Nothing about
efficacy, directionality or benchmark quality can be read off two optimizer steps
on a toy pair. The directional question belongs to the branch probe of the
specification, which this run does not replace.

Gates this smoke is allowed to answer:

1. does the new module import and pass under the runtime torch (not only CPU torch);
2. does `mp_opd_mode=kernel` execute the full rollout -> atomize -> score ->
   credit-transform -> NLL path on a real GPU;
3. does `MP_OPD_CREDIT_IDENTITY_CHECK=1` hold on real atoms, i.e. does the identity
   operator reproduce the historical Atomic pooled loss on a real microbatch;
4. does each operator's telemetry reach the log, and are the identity arms exactly
   the Atomic arms.

Cost control: six short trainings in one B200 container, two optimizer updates each,
W&B off, no checkpoint writes beyond what the trainer requires.
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
RUN = 'mp-opd-credit-kernel-smoke-20261006-r1'
STUDENT = 'Qwen/Qwen2.5-0.5B-Instruct'
TEACHER = 'Qwen/Qwen2.5-1.5B-Instruct'
UPDATES = 2
SAMPLES = 8
MAX_SPAN = 4

# Pre-registered conditions. `atomic` and `kernel-identity` are the same operator
# reached through two modes; the three neighbor arms are the Experiment C controls;
# `kernel-causal-h3` exercises the multi-offset kernel branch.
CONDITIONS = (
    {'tag': 'atomic', 'mode': 'atomic', 'transform': 'identity', 'identity_check': True},
    {'tag': 'kernel-identity', 'mode': 'kernel', 'transform': 'identity', 'identity_check': True},
    {'tag': 'kernel-forward1', 'mode': 'kernel', 'transform': 'forward', 'lam': 0.25},
    {'tag': 'kernel-backward1', 'mode': 'kernel', 'transform': 'backward', 'lam': 0.25},
    {'tag': 'kernel-shuffle1', 'mode': 'kernel', 'transform': 'shuffle', 'lam': 0.25},
    {
        'tag': 'kernel-causal-h3',
        'mode': 'kernel',
        'transform': 'causal_kernel',
        'horizon': 3,
        'kernel': 'uniform',
        'direction': 'forward',
    },
)
CONDITION_TAGS = tuple(condition['tag'] for condition in CONDITIONS)

# Telemetry the credit operator must emit, from section 11 of the specification.
CREDIT_KEYS = (
    'mp_opd_credit_transform_code',
    'mp_opd_credit_valid_atom_count',
    'mp_opd_credit_mean_atomic',
    'mp_opd_credit_std_atomic',
    'mp_opd_credit_rms_atomic',
    'mp_opd_credit_mean_effective',
    'mp_opd_credit_std_effective',
    'mp_opd_credit_rms_effective',
    'mp_opd_credit_corr_effective_atomic',
    'mp_opd_credit_mean_abs_delta',
    'mp_opd_credit_transfer_fraction',
    'mp_opd_credit_sign_flip_fraction',
    'mp_opd_credit_neighbor_product_mean',
    'mp_opd_credit_neighbor_sign_agreement',
    'mp_opd_credit_neighbor_pair_count',
    'mp_opd_credit_lambda',
    'mp_opd_credit_horizon',
)
IDENTITY_EXTRA_KEYS = (
    'mp_opd_credit_identity_abs_diff',
    'mp_opd_credit_identity_rel_diff',
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


def env(online=False, identity_check=False):
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
    if identity_check:
        # Fail-closed regression: the identity operator must reproduce the historical
        # Atomic pooled loss on the real microbatch, not only on synthetic credits.
        e['MP_OPD_CREDIT_IDENTITY_CHECK'] = '1'
    libs = ['/usr/local/cuda/lib64', '/usr/local/nvidia/lib64'] + [
        str(p) for p in Path('/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia').glob('*/lib')
    ]
    e['LD_LIBRARY_PATH'] = ':'.join(libs)
    return e


@app.function(image=image, cpu=8, memory=32768, timeout=2400, volumes={'/runs': outputs})
def unit_tests(commit: str):
    """The new module must pass under the real runtime torch, not only local CPU torch."""
    root = Path('/runs')
    root.mkdir(exist_ok=True)
    e = env()
    command = [PYTHON, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
               'tests/mp_opd/test_credit_transform.py',
               'tests/mp_opd/test_gbv_span.py',
               'tests/mp_opd/test_credit_oracle.py',
               'tests/mp_opd/test_random_partition_min_span.py',
               'tests/mp_opd/test_atoms.py',
               'tests/mp_opd/test_training_diagnostics.py']
    done = subprocess.run(command, cwd='/opt/overlay', env=e, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=2300)
    (root / 'unit-tests.log').write_text(done.stdout)
    outputs.commit()
    tail = done.stdout.strip().splitlines()[-3:]
    print('CREDIT_UNIT_TAIL=' + json.dumps(tail), flush=True)
    print('CREDIT_UNIT_JSON=' + json.dumps({'commit': commit, 'returncode': done.returncode,
                                            'passed': done.returncode == 0}), flush=True)
    if done.returncode:
        raise RuntimeError('runtime unit tests failed')


def _training_options(condition, updates, student, teacher, prompts):
    """The shared recipe, with only the credit-operator knobs varying per condition."""
    options = dict(
        num_nodes=1, num_gpus_per_node=1, backend='fsdp2',
        student_name_or_path=str(student), teacher_name_or_path=str(teacher),
        attn_implementation='sdpa', num_epochs=updates, train_batch_size=SAMPLES,
        micro_train_batch_size=1, learning_rate=1e-6, lr_warmup_ratio=.05,
        lr_scheduler='cosine_with_min_lr', min_lr=0, weight_decay=0.,
        gradient_checkpointing=True, enable_sleep=True, bf16=True, seed=42,
        # Per-condition output dirs: the trainer fails closed when a checkpoint already
        # exists, so sharing one save_path would abort every run after the first.
        save_path='/runs/checkpoint-' + condition['tag'],
        ckpt_path='/runs/checkpoints-' + condition['tag'],
        train_dataset_path=str(prompts), input_key='messages', apply_chat_template=True,
        enable_thinking=False, max_samples=SAMPLES, prompt_max_len=0, max_len=2048,
        preprocess_num_workers=2, rollout_num_engines=1,
        rollout_disable_piecewise_cuda_graph=True, rollout_tp_size=1,
        rollout_mem_fraction_static=.25, rollout_batch_size=SAMPLES,
        generate_max_len=1024, n_samples_per_prompt=1, temperature=.6, top_p=.95,
        teacher_tp_size=1, teacher_pp_size=1, teacher_ep_size=1, teacher_dp_size=1,
        teacher_mem_fraction_static=.3, teacher_context_length=4096,
        teacher_forward_n_batches=8, kd_algorithm='mp_opd', kd_loss_fn='rkl', kd_ratio=1.,
        span_score_mode='mean_logprob', exact_token_trajectory=True,
        enforce_max_sequence_length=True, diagnostic_max_updates=updates,
        diagnostic_collapse_gate=False, save_steps=1000, logging_steps=1,
        use_wandb=False,
        mp_opd_mode=condition['mode'],
        mp_opd_max_span_length=MAX_SPAN,
        mp_opd_credit_transform=condition['transform'],
        mp_opd_credit_lambda=condition.get('lam', 0.25),
        mp_opd_credit_horizon=condition.get('horizon', 2),
        mp_opd_credit_kernel=condition.get('kernel', 'uniform'),
        mp_opd_credit_direction=condition.get('direction', 'forward'),
        mp_opd_credit_shuffle_seed=condition.get('shuffle_seed', 43),
        mp_opd_credit_convex=True,
    )
    return options


@app.function(image=image, gpu='B200', cpu=8, memory=65536, timeout=2400, retries=0,
              max_containers=1, volumes={'/assets': assets, '/runs': outputs})
def credit_path_probe(commit: str, samples: int = 8):
    """Run the credit path on a real B200 with real atoms, without the rollout engine.

    Why not just train: at this revision every on-policy rollout with
    ``temperature > 0`` aborts inside the SGLang engine with
    ``engine returned a zero sampled behavior log-probability (p=1)``. That is the
    documented upstream 0.5.11 defect (``SGLANG_ZERO_LOGPROB_BUG.md``,
    ``LONGCAP_PARITY_PROBE.md`` section 17) and the team decision is to keep the
    guard closed rather than relax it. The engine is therefore not evidence about
    this change, and this probe replaces it: the same atomizer, the same
    ``build_atom_credits``, the same ``production_credit_batch`` and the same
    operators, on real logits from a real GPU forward pass, with the historical
    pooled Atomic loss computed alongside as the regression oracle.

    What this can show: the credit operators run on real atoms, the identity
    operator reproduces the historical Atomic loss, and telemetry is finite. What
    it cannot show: anything about rollout, training dynamics or utility.
    """
    root = Path('/runs')
    root.mkdir(exist_ok=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kdflow.algorithms._mp_opd_atoms import SimCTAtomizer
    from kdflow.algorithms._mp_opd_credit import build_atom_credits, hard_partition_loss
    from kdflow.algorithms._mp_opd_credit_transform import (
        CREDIT_TRANSFORM_CHOICES,
        build_credit_transform,
        production_credit_batch,
    )

    def legacy_atomic_loss(current_nll, base, weight):
        """The historical Atomic objective: the singleton partition of the pooled loss."""
        partition = tuple((index, index + 1) for index in range(current_nll.numel()))
        return hard_partition_loss(current_nll, base, weight, partition)

    pairs = [
        ('What is 17*23? Show the arithmetic.', ' 17 * 23 = 391, so the answer is 391.'),
        ('Solve for x: 3x + 7 = 25.', ' 3x = 18, so x = 6.'),
        ('What is the sum of the first ten positive integers?', ' The sum is 55.'),
        ('Write a Python function that reverses a string.',
         ' def reverse(s):\n    return s[::-1]'),
        ('What is the greatest common divisor of 84 and 132?', ' The answer is 12.'),
        ('Simplify (x^2 - 9)/(x - 3).', ' It simplifies to x + 3.'),
        ('Write a Python one-liner that counts vowels in a string.',
         " sum(c in 'aeiou' for c in s)"),
        ('What is 2^10 divided by 8?', ' 1024 / 8 = 128.'),
    ][:samples]

    device = 'cuda'
    student_tok = AutoTokenizer.from_pretrained('/assets/student')
    teacher_tok = AutoTokenizer.from_pretrained('/assets/teacher')
    student = AutoModelForCausalLM.from_pretrained(
        '/assets/student', torch_dtype=torch.bfloat16, attn_implementation='sdpa'
    ).to(device).eval()
    teacher = AutoModelForCausalLM.from_pretrained(
        '/assets/teacher', torch_dtype=torch.bfloat16, attn_implementation='sdpa'
    ).to(device).eval()
    atomizer = SimCTAtomizer(student_tok, teacher_tok)

    operators = list(CREDIT_TRANSFORM_CHOICES)
    per_operator = {name: [] for name in operators}
    per_operator['legacy_atomic'] = []
    atom_report = []
    invalid = 0
    with torch.no_grad():
        for index, (prompt, response) in enumerate(pairs):
            row = {}
            for tag, tokenizer, model in (('student', student_tok, student),
                                          ('teacher', teacher_tok, teacher)):
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                response_ids = tokenizer.encode(response, add_special_tokens=False)
                eos = tokenizer.eos_token_id
                if eos is not None and (not response_ids or response_ids[-1] != eos):
                    response_ids = response_ids + [eos]
                ids = prompt_ids + response_ids
                logits = model(torch.tensor([ids], device=device)).logits[0]
                # Rows that predict a response token: the last prompt row predicts the
                # first response token, and the final row predicts nothing.
                first = len(prompt_ids) - 1
                last = len(ids) - 1
                row[tag] = (logits[first:last].float(),
                            torch.tensor(ids[first + 1:last + 1], device=device),
                            response_ids)
            stu_logits, stu_labels, stu_response = row['student']
            tea_logits, tea_labels, tea_response = row['teacher']
            atomized = atomizer.atomize(stu_response, tea_response, sample_id='probe-%d' % index)
            if not atomized.valid:
                invalid += 1
                print('PROBE_ATOMIZE_FAIL=' + json.dumps({'index': index,
                                                          'reason': atomized.failure_reason}))
                continue
            atoms = atomized.atoms
            credits = build_atom_credits(atoms, stu_logits, stu_labels, tea_logits, tea_labels)
            atom_report.append({
                'index': index, 'atoms': len(atoms),
                'one_to_one': sum(atom.boundary_type == 'one_to_one' for atom in atoms),
                'student_tokens': int(credits.weight.sum().item()),
                'masked_student_eos': atomized.masked_student_eos,
                'w_mean': float(credits.weight.mean().item()),
                'base_mean': float(credits.base_credit.mean().item()),
                'base_std': float(credits.base_credit.std(unbiased=False).item()),
                'rate_mean': float(credits.rate.mean().item()),
                'rate_std': float(credits.rate.std(unbiased=False).item()),
                'rate_min': float(credits.rate.min().item()),
                'rate_max': float(credits.rate.max().item()),
            })
            batch = production_credit_batch(credits.rate, credits.base_credit, credits.weight)
            legacy = float(
                legacy_atomic_loss(credits.current_nll, credits.base_credit, credits.weight)
                .detach()
                .item()
            )
            per_operator['legacy_atomic'].append({'loss': legacy})
            for name in operators:
                if name == 'external':
                    # The external operator is Experiment B's offline augmentation; give
                    # it the local directional signal the branch probe would supply.
                    direction = (credits.rate - credits.rate.mean()).detach()
                    transform = build_credit_transform(name, alpha=0.25, scale_match='rms')
                    operator_batch = production_credit_batch(
                        credits.rate, credits.base_credit, credits.weight,
                        {'future_advantage': direction})
                else:
                    transform = build_credit_transform(
                        name, lam=0.25, horizon=3, kernel='uniform', direction='forward',
                        shuffle_seed=43)
                    operator_batch = batch
                output = transform(operator_batch, training=True)
                effective = output.effective_credit
                loss = float((effective * credits.current_nll).sum().detach().item())
                per_operator[name].append({
                    'loss': loss,
                    'relative_drift_vs_legacy': abs(loss - legacy) / (1.0 + abs(legacy)),
                    'finite': all(bool(torch.isfinite(value).all().item())
                                  for value in output.diagnostics.values()),
                    'rms_atomic': float(output.diagnostics['mp_opd_credit_rms_atomic'].item()),
                    'rms_effective': float(output.diagnostics['mp_opd_credit_rms_effective'].item()),
                    'mean_abs_delta': float(output.diagnostics['mp_opd_credit_mean_abs_delta'].item()),
                    'transfer_fraction': float(
                        output.diagnostics['mp_opd_credit_transfer_fraction'].item()),
                    'corr': float(
                        output.diagnostics['mp_opd_credit_corr_effective_atomic'].item()),
                })

    summary = {'commit': commit, 'samples': len(pairs), 'invalid_samples': invalid,
               'atoms': atom_report, 'operators': {}}
    failures = []
    for name, rows in per_operator.items():
        if not rows:
            continue
        summary['operators'][name] = {
            'mean_loss': sum(row['loss'] for row in rows) / len(rows),
            'max_relative_drift': max(
                row['relative_drift_vs_legacy'] for row in rows) if name != 'legacy_atomic' else 0.0,
            'max_rms_ratio': max(
                row['rms_effective'] / max(row['rms_atomic'], 1e-12)
                for row in rows) if name != 'legacy_atomic' else 1.0,
            'min_transfer_fraction': min(
                row['transfer_fraction'] for row in rows) if name != 'legacy_atomic' else 0.0,
            'all_finite': all(row.get('finite', True) for row in rows),
        }
    for name, stats in summary['operators'].items():
        if name in ('identity', 'legacy_atomic'):
            if stats['max_relative_drift'] > 1e-4:
                failures.append('%s drift %.3g > 1e-4' % (name, stats['max_relative_drift']))
        else:
            if not stats['all_finite']:
                failures.append('%s emitted a non-finite metric' % name)
            if stats['min_transfer_fraction'] <= 0.0:
                failures.append('%s never moved a credit' % name)
            if stats['max_rms_ratio'] > 1.05:
                failures.append('%s inflated the credit scale (%.3g)'
                                % (name, stats['max_rms_ratio']))
    summary['failures'] = failures
    (root / 'credit-path-probe.json').write_text(json.dumps(summary, indent=2))
    outputs.commit()
    print('CREDIT_PATH_ATOMS=' + json.dumps(atom_report), flush=True)
    print('CREDIT_PATH_JSON=' + json.dumps(summary), flush=True)
    if failures:
        raise RuntimeError('credit path probe failed: ' + '; '.join(failures))


@app.function(image=image, gpu='B200', cpu=16, memory=98304, timeout=3600, retries=0,
              max_containers=1, volumes={'/assets': assets, '/runs': outputs})
def train_smoke(commit: str, condition_json: str, updates: int):
    condition = json.loads(condition_json)
    tag = condition['tag']
    root = Path('/runs')
    root.mkdir(exist_ok=True)
    log_path = root / ('train-' + tag + '.log')
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

    options = _training_options(condition, updates, student, teacher, prompts)
    command = [PYTHON, '-m', 'kdflow.cli.train_kd_on_policy']
    for key, value in options.items():
        command += ['--' + key, str(value)]
    (root / ('invocation-' + tag + '.json')).write_text(json.dumps(dict(
        purpose='cross-atom credit code-path smoke, not a result', source_commit=commit,
        condition=condition, teacher=TEACHER, student=str(student), updates=updates,
        samples=SAMPLES, command=command), indent=2))
    outputs.commit()

    process = None
    try:
        with log_path.open('x') as handle:
            process = subprocess.Popen(
                command, cwd='/opt/overlay',
                env=env(online=True, identity_check=bool(condition.get('identity_check'))),
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            started = time.monotonic()
            while process.poll() is None:
                outputs.commit()
                if time.monotonic() - started > 3000:
                    raise TimeoutError('smoke time cap')
                time.sleep(15)
        text = log_path.read_text()
        if process.returncode:
            print('SMOKE_LOG_TAIL=' + json.dumps(text.strip().splitlines()[-20:]), flush=True)
            raise RuntimeError('training failed rc=' + str(process.returncode))
        seen = {}
        for key in CREDIT_KEYS:
            found = re.findall(re.escape(key) + r':\s*([-\d.eE+]+)', text)
            if found:
                seen[key] = float(found[-1])
        required = list(CREDIT_KEYS)
        if condition.get('identity_check'):
            required += list(IDENTITY_EXTRA_KEYS)
            for key in IDENTITY_EXTRA_KEYS:
                found = re.findall(re.escape(key) + r':\s*([-\d.eE+]+)', text)
                if found:
                    seen[key] = float(found[-1])
        missing = [key for key in required if key not in seen]
        summary = {'commit': commit, 'tag': tag, 'condition': condition, 'updates': updates,
                   'metrics_seen': seen, 'missing': missing,
                   'note': 'code-path smoke only; no efficacy claim'}
        (root / ('credit-smoke-' + tag + '.json')).write_text(json.dumps(summary, indent=2))
        outputs.commit()
        print('CREDIT_SMOKE_JSON=' + json.dumps(summary), flush=True)
        if missing:
            raise RuntimeError('credit telemetry missing for ' + tag + ': ' + ','.join(missing))
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


def cross_check(summaries):
    """Local consistency gates over the returned telemetry.

    These are *code-path* gates, not scientific ones: they say the operators were
    wired to the intended atom credits, and they say nothing about utility. The
    hard identity gate already ran inside the container and would have aborted the
    run; here it is only re-read from the log.
    """
    by_tag = {summary['tag']: summary['metrics_seen'] for summary in summaries}
    checks = []
    for tag in ('atomic', 'kernel-identity'):
        seen = by_tag[tag]
        checks.append((f'{tag}: identity leaves every atom credit unchanged',
                       seen['mp_opd_credit_mean_abs_delta'] == 0.0))
        checks.append((f'{tag}: identity transfer_fraction is zero',
                       seen['mp_opd_credit_transfer_fraction'] == 0.0))
        checks.append((f'{tag}: identity correlation with atomic is 1',
                       seen['mp_opd_credit_corr_effective_atomic'] >= 0.999))
        checks.append((f'{tag}: historical Atomic loss reproduced (relative drift < 1e-5)',
                       seen['mp_opd_credit_identity_rel_diff'] < 1e-5))
    atomic_mean = by_tag['atomic']['mp_opd_credit_mean_effective']
    kernel_identity_mean = by_tag['kernel-identity']['mp_opd_credit_mean_effective']
    checks.append(('mode kernel + identity equals mode atomic on mean credit',
                   abs(atomic_mean - kernel_identity_mean) < 1e-4))
    for tag in ('kernel-forward1', 'kernel-backward1', 'kernel-causal-h3'):
        checks.append((f'{tag}: credit actually moved',
                       by_tag[tag]['mp_opd_credit_transfer_fraction'] > 0.5))
    checks.append(('kernel-causal-h3 reports its horizon',
                   by_tag['kernel-causal-h3']['mp_opd_credit_horizon'] == 3.0))
    checks.append(('kernel-forward1 reports the registered lambda',
                   by_tag['kernel-forward1']['mp_opd_credit_lambda'] == 0.25))
    checks.append(('the shuffle control keeps the credit marginal intact',
                   abs(by_tag['kernel-shuffle1']['mp_opd_credit_rms_effective']
                       - by_tag['kernel-shuffle1']['mp_opd_credit_rms_atomic'])
                   <= 0.35 * by_tag['kernel-shuffle1']['mp_opd_credit_rms_atomic']))
    for tag in CONDITION_TAGS:
        seen = by_tag[tag]
        checks.append((f'{tag}: no credit-scale inflation',
                       seen['mp_opd_credit_rms_effective'] <= 1.001 * seen['mp_opd_credit_rms_atomic']))
    return checks


@app.local_entrypoint()
def main(
    updates: int = UPDATES,
    conditions: str = ','.join(CONDITION_TAGS),
    phases: str = 'units,probe',
):
    """Run the selected phases: ``units`` (CPU runtime tests), ``probe`` (B200 credit
    path without the engine), ``train`` (B200 rollout; blocked while the SGLang
    0.5.11 zero-log-probability defect is open -- see ``credit_path_probe``).
    """
    # This checkout lives on an NTFS path that two different git builds read: the
    # Windows one with core.autocrlf=true (which created the working tree) and the
    # WSL one with no autocrlf at all. Without the explicit flag the WSL check calls
    # every CRLF file modified and the smoke refuses to start for a reason that is
    # not a dirty tree. The flag makes the guard mean the same thing in both.
    git = ['git', '-c', 'core.autocrlf=true']
    dirty = subprocess.check_output(
        git + ['status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError('Commit tracked changes first:\n' + dirty)
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    print('SMOKE_SOURCE_COMMIT=' + commit, flush=True)
    selected_phases = [item.strip() for item in phases.split(',') if item.strip()]
    unknown_phases = [item for item in selected_phases if item not in {'units', 'probe', 'train'}]
    if unknown_phases:
        raise ValueError('unknown phases ' + ','.join(unknown_phases))
    if 'units' in selected_phases:
        unit_tests.remote(commit)
    if 'probe' in selected_phases:
        credit_path_probe.remote(commit)
    if 'train' not in selected_phases:
        print('CREDIT_SMOKE_PHASES=' + json.dumps(selected_phases), flush=True)
        return
    selected = [item.strip() for item in conditions.split(',') if item.strip()]
    unknown = [tag for tag in selected if tag not in CONDITION_TAGS]
    if unknown:
        raise ValueError('unknown conditions ' + ','.join(unknown))
    summaries = []
    for condition in CONDITIONS:
        if condition['tag'] not in selected:
            continue
        summaries.append(train_smoke.remote(commit, json.dumps(condition), updates))
    checks = cross_check(summaries)
    for label, passed in checks:
        print('CREDIT_CROSS_CHECK ' + ('PASS' if passed else 'FAIL') + ' ' + label, flush=True)
    failed = [label for label, passed in checks if not passed]
    print('CREDIT_SMOKE_ALL=' + json.dumps(summaries), flush=True)
    if failed:
        raise RuntimeError('cross-checks failed: ' + '; '.join(failed))
    print('CREDIT_SMOKE_PASS=' + json.dumps({'commit': commit, 'conditions': selected,
                                             'updates': updates}), flush=True)
