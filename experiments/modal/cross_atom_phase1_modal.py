"""Phase 1 sampler validation (brief steps D/E) on the B200 image.

Purpose: prove the standalone HF frozen-policy sampler is reproducible, reports finite
log-probabilities for every sampled token, honours the campaign's sampling config, and
produces continuations the production atomizer accepts. **This is not the branch probe
and not a result**: no prefix sampling, no branching, no utility measurement.

Asset honesty: the campaign's fine-tuned Gemma-2 student lives on company storage and is
not reachable from Modal, so this validation runs on the staged public base
``google/gemma-2-2b-it``. That is sufficient for step E, which is about the *sampler's*
properties (determinism, logprob contract, stop logic, atomizability) and not about the
policy's behaviour. Phase 1 itself still needs the real checkpoints; passing
``--student-dir`` swaps them in with no code change.

Config parity comes from the campaign's own record: the launch config of the September
qwen7b->gemma2 run is mounted and read, rather than re-typing temperature and top-p.
"""
import json
import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/opt/overlay')
PYTHON = '/opt/venvs/simct-b200/bin/python'
IMAGE = 'docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f'
RUN = 'cross-atom-phase1-20261006-r1'
STAGED_VOLUME = 'simct-qwen7b-gemma2-assets-20260915'
STAGED_MOUNT = '/staged'
STUDENT_DIR = STAGED_MOUNT + '/student'
CAMPAIGN_VOLUME = 'simct-qwen7b-gemma2-e2e-20260915-r6'
CAMPAIGN_MOUNT = '/campaign'
CAMPAIGN_LAUNCH_CONFIG = ('qwen7b-gemma2-soft-offload-1update-20260915-final3/'
                          'launch-config.json')
TEACHER_TOKENIZER_DIR = '/assets/teacher'

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
    .add_local_dir(str(ROOT / 'experiments/modal'), '/opt/overlay/experiments/modal',
                   ignore=['**/__pycache__/**', '**/*.pyc'])
    .add_local_dir(str(ROOT / 'experiments/mp_opd'), '/opt/overlay/experiments/mp_opd',
                   ignore=['**/__pycache__/**', '**/*.pyc'])
)
assets = modal.Volume.from_name('cross-atom-credit-assets', create_if_missing=True)
staged = modal.Volume.from_name(STAGED_VOLUME, create_if_missing=False)
campaign = modal.Volume.from_name(CAMPAIGN_VOLUME, create_if_missing=False)
outputs = modal.Volume.from_name(RUN, create_if_missing=True)


@app.function(image=image, cpu=8, memory=32768, timeout=1800,
              volumes={'/assets': assets, STAGED_MOUNT: staged, '/runs': outputs})
def sampler_unit_tests(commit: str, student_dir: str = ''):
    """The sampler tests, including the one that needs the real tokenizers."""
    root = Path('/runs')
    root.mkdir(exist_ok=True)
    e = dict(os.environ)
    e.update(PATH='/opt/venvs/simct-b200/bin:' + e.get('PATH', ''),
             PYTHONPATH='/opt/overlay/experiments/modal/vendor:/opt/overlay',
             TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8')
    e['CA_STUDENT_DIR'] = student_dir or STUDENT_DIR
    e['CA_TEACHER_DIR'] = TEACHER_TOKENIZER_DIR
    command = [PYTHON, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
               'tests/mp_opd/test_hf_frozen_sampler.py']
    done = subprocess.run(command, cwd='/opt/overlay', env=e, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1700)
    (root / 'sampler-tests.log').write_text(done.stdout)
    outputs.commit()
    print('PHASE1_SAMPLER_TESTS=' + json.dumps(
        {'returncode': done.returncode, 'tail': done.stdout.strip().splitlines()[-3:]}),
        flush=True)
    if done.returncode:
        raise RuntimeError('sampler unit tests failed')


@app.function(image=image, gpu='B200', cpu=16, memory=131072, timeout=3600, retries=0,
              max_containers=1,
              volumes={'/assets': assets, STAGED_MOUNT: staged, CAMPAIGN_MOUNT: campaign,
                       '/runs': outputs})
def sampler_validation(commit: str, student_dir: str = '', max_new_tokens: int = 256,
                       samples: int = 4):
    os.environ['HF_HOME'] = '/assets/hf'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from experiments.mp_opd.hf_frozen_sampler import (
        HFFrozenPolicySampler,
        sampling_config_from_launch_config,
        terminal_token_ids,
    )
    from kdflow.algorithms._mp_opd_atoms import SimCTAtomizer

    root = Path('/runs')
    root.mkdir(exist_ok=True)
    resolved_student = (student_dir or STUDENT_DIR).strip() or STUDENT_DIR
    launch_config_path = Path(CAMPAIGN_MOUNT) / CAMPAIGN_LAUNCH_CONFIG
    campaign_options = json.loads(launch_config_path.read_text())
    campaign_options = campaign_options.get('options', campaign_options)

    student_tok = AutoTokenizer.from_pretrained(resolved_student)
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER_TOKENIZER_DIR)
    stop_ids = terminal_token_ids(student_tok)
    config = sampling_config_from_launch_config(
        launch_config_path, stop_token_ids=stop_ids)
    # Config parity is a claim, so check it explicitly against the campaign's record.
    parity = {
        'campaign_temperature': campaign_options['temperature'],
        'campaign_top_p': campaign_options['top_p'],
        'campaign_generate_max_len': campaign_options['generate_max_len'],
        'campaign_max_len': campaign_options['max_len'],
        'sampler_temperature': config.temperature,
        'sampler_top_p': config.top_p,
        'sampler_max_new_tokens': config.max_new_tokens,
        'sampler_max_context_tokens': config.max_context_tokens,
        'temperature_matches': config.temperature == float(campaign_options['temperature']),
        'top_p_matches': config.top_p == float(campaign_options['top_p']),
        'max_new_tokens_matches':
            config.max_new_tokens == int(campaign_options['generate_max_len']),
        'max_context_matches': config.max_context_tokens == int(campaign_options['max_len']),
        'stop_token_ids': list(stop_ids),
    }

    student = AutoModelForCausalLM.from_pretrained(
        resolved_student, torch_dtype=torch.bfloat16, attn_implementation='sdpa'
    ).to('cuda').eval()
    bounded = type(config)(temperature=config.temperature, top_p=config.top_p,
                           top_k=config.top_k, max_new_tokens=int(max_new_tokens),
                           stop_strings=config.stop_strings,
                           stop_token_ids=config.stop_token_ids,
                           max_context_tokens=config.max_context_tokens)
    engine = HFFrozenPolicySampler(student, student_tok, bounded,
                                   student_revision=resolved_student)
    full_engine = HFFrozenPolicySampler(student, student_tok, config,
                                        student_revision=resolved_student)

    prompts = [
        'Natalia sold clips to 48 friends in April, and then she sold half as many clips '
        'in May. How many clips did Natalia sell altogether in April and May?',
        'Simplify the expression (x^2 - 9)/(x - 3) for x != 3.',
        'Write a Python function that returns the greatest common divisor of two integers.',
        'What does the word "idempotent" mean in software engineering?',
    ][:samples]

    def chat_prompt(text):
        return student_tok.apply_chat_template(
            [{'role': 'user', 'content': text}], tokenize=False,
            add_generation_prompt=True)

    rollouts = []
    failures = []
    for index, prompt in enumerate(prompts):
        prompt_ids = student_tok.encode(chat_prompt(prompt), add_special_tokens=False)
        first = engine.sample(prompt_ids=prompt_ids, seed=1000 + index)
        again = engine.sample(prompt_ids=prompt_ids, seed=1000 + index)
        record = {
            'index': index,
            'prompt_tokens': len(prompt_ids),
            'generated_tokens': first.generated_tokens,
            'finished_reason': first.finished_reason,
            'deterministic': first.token_ids == again.token_ids
            and first.logprobs == again.logprobs,
            'lengths_align': len(first.token_ids) == len(first.logprobs),
            'all_logprobs_finite': all(torch.isfinite(torch.tensor(value)).item()
                                       for value in first.logprobs),
            'no_zero_logprob': all(value != 0.0 for value in first.logprobs),
            'max_logprob': max(first.logprobs) if first.logprobs else None,
            'min_logprob': min(first.logprobs) if first.logprobs else None,
            'text_head': first.text[:160],
        }
        # Every sampled token must be reproducible from the recorded logprob: re-running
        # the same seed must give the same value, and a different seed a different draw.
        other = engine.sample(prompt_ids=prompt_ids, seed=2000 + index)
        record['seed_changes_the_draw'] = other.token_ids != first.token_ids
        content = list(first.content_ids)
        text = student_tok.decode(content, skip_special_tokens=True)
        stu_ids = student_tok.encode(text, add_special_tokens=False)
        tea_ids = teacher_tok.encode(text, add_special_tokens=False)
        atomized = SimCTAtomizer(student_tok, teacher_tok).atomize(
            stu_ids, tea_ids, sample_id='phase1-sampler-%d' % index)
        record['atomization'] = 'OK' if atomized.valid else 'FAILED'
        record['atomization_failure'] = atomized.failure_reason
        if atomized.valid:
            weights = [atom.student_token_count for atom in atomized.atoms]
            record['atoms'] = len(weights)
            record['atoms_w_gt_1'] = sum(1 for value in weights if value > 1)
        for key in ('deterministic', 'lengths_align', 'all_logprobs_finite',
                    'no_zero_logprob'):
            if not record[key]:
                failures.append('sample %d: %s' % (index, key))
        if record['atomization'] != 'OK':
            failures.append('sample %d: atomization %s'
                            % (index, record['atomization_failure']))
        rollouts.append(record)

    # One draw under the campaign's *full* config, to show the unbounded recipe runs.
    full_prompt_ids = student_tok.encode(chat_prompt(prompts[0]), add_special_tokens=False)
    full_rollout = full_engine.sample(prompt_ids=full_prompt_ids, seed=4242)
    lengths = [row['generated_tokens'] for row in rollouts]
    summary = {
        'commit': commit,
        'profile': 'lhtu05',
        'phase': '1-sampler-validation',
        'student_dir': resolved_student,
        'student_is_staged_public_base': resolved_student == STUDENT_DIR,
        'student_revision_note': ('staged public base google/gemma-2-2b-it; the campaign '
                                  'SFT checkpoint is on company storage and is not '
                                  'reachable from Modal'),
        'config_parity': parity,
        'bounded_max_new_tokens': int(max_new_tokens),
        'rollouts': rollouts,
        'length_distribution': {
            'n': len(lengths), 'min': min(lengths), 'max': max(lengths),
            'mean': sum(lengths) / max(len(lengths), 1),
            'finished_reasons': sorted({row['finished_reason'] for row in rollouts}),
        },
        'full_config_rollout': {
            'max_new_tokens': config.max_new_tokens,
            'generated_tokens': full_rollout.generated_tokens,
            'finished_reason': full_rollout.finished_reason,
            'all_logprobs_finite': all(
                torch.isfinite(torch.tensor(value)).item()
                for value in full_rollout.logprobs),
            'no_zero_logprob': all(value != 0.0 for value in full_rollout.logprobs),
        },
        'failures': failures,
        'note': ('sampler validation only: no prefix sampling, no branching, no utility '
                 'measurement, and no statement about the policy'),
    }
    (root / 'cross_atom_phase1_sampler_validation.json').write_text(json.dumps(summary, indent=2))
    outputs.commit()
    print('PHASE1_SAMPLER_JSON=' + json.dumps(
        {key: value for key, value in summary.items() if key != 'rollouts'}), flush=True)
    if failures:
        raise RuntimeError('sampler validation failed: ' + '; '.join(failures))
    return summary


@app.local_entrypoint()
def main(phases: str = 'tests,validate', student_dir: str = '', max_new_tokens: int = 256,
         samples: int = 4):
    git = ['git', '-c', 'core.autocrlf=true']
    dirty = subprocess.check_output(
        git + ['status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError('Commit tracked changes first:\n' + dirty)
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    print('PHASE1_SOURCE_COMMIT=' + commit, flush=True)
    selected = [item.strip() for item in phases.split(',') if item.strip()]
    unknown = [item for item in selected if item not in {'tests', 'validate'}]
    if unknown:
        raise ValueError('unknown phases ' + ','.join(unknown))
    if 'tests' in selected:
        sampler_unit_tests.remote(commit, student_dir)
    if 'validate' in selected:
        result = sampler_validation.remote(commit, student_dir, max_new_tokens, samples)
        print('PHASE1_SAMPLER_VERDICT=' + json.dumps(
            {'failures': result['failures'],
             'config_parity': result['config_parity'],
             'length_distribution': result['length_distribution']}), flush=True)
    print('PHASE1_PHASES=' + json.dumps(selected), flush=True)
