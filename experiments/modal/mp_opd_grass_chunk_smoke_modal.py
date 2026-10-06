"""Bounded GRASS-Chunk smoke: runtime unit tests, then real `grass_chunk` steps.

Purpose: does `mp_opd_mode=grass_chunk` execute end to end on the B200 runtime, and
does its telemetry reach the trainer log. **Not a result.** The model pair and the
prompt set exist only to make the path run: a small Qwen2.5 student and a slightly
larger Qwen2.5 teacher of the same family, plus eight short questions written here.
Nothing about credit quality, shrinkage strength or benchmark performance can be
read off this run.

Two chunk sources are exercised, and the difference between them is the point:

* `xtoken` - the native path. The upstream `TokenAligner` is built from the two
  tokenizers and returns per-token chunk ids, which are projected onto atoms by
  `atom_chunk_ids_from_tokens`. **Caveat:** the student and teacher here share a
  tokenizer, so the alignment is near-identity and the chunks are not the ones a
  real cross-tokenizer pair would produce. This proves the plumbing, not the
  chunk quality.
* `run` - the fixed-run baseline, which needs no projection and is the honest
  label for what it is.

`xtoken_projection_path` must point at a file whose sha256 matches the argument,
because `_build_grass_aligner` re-verifies the digest. `TokenAligner.align()` never
loads that file - it aligns the token strings, and only the xtoken *loss* reads the
projection - so a clearly named placeholder is enough to exercise the chunk path.
It is a placeholder for a plumbing test and must never be used to train.

Cost control: two short trainings in one B200 container, two optimizer updates each,
W&B off, no checkpoint writes, models cached in a Modal volume across both runs.
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
RUN = 'mp-opd-grass-chunk-smoke-20260920-r1'
STUDENT = 'Qwen/Qwen2.5-0.5B-Instruct'
TEACHER = 'Qwen/Qwen2.5-1.5B-Instruct'
# 'xtoken' first: it is the method's intended chunk source and the one whose code
# path has never executed before. 'run' is the baseline it must be compared against.
SOURCES = ('xtoken', 'run')
RUN_LENGTH = 2
GEOMETRY = 'exact_head'
UPDATES = 2
SAMPLES = 8
SIGMA_RHO = 0.99
SIGMA_MIN_PAIRS = 8

PLACEHOLDER_NOTE = (
    'plumbing-test placeholder for GRASS-Chunk: TokenAligner.align() aligns token '
    'strings and never reads this file; only the xtoken loss does. Generated inside '
    'the container and digested there, so it is not a pinned artifact. Not a usable '
    'projection and never a substitute for one.'
)

# Keys emitted unconditionally by grass_chunk_metrics(), the fixed telemetry block in
# _grass_chunk_loss, chunk_boundary_diagnostics() and the shadow block. A missing one
# is a wiring fault; a zero-valued one is a legitimate measurement.
REQUIRED_KEYS = (
    'mp_opd_grass_chunk_count',
    'mp_opd_grass_chunk_source',
    'mp_opd_grass_chunk_length_mean',
    'mp_opd_grass_chunk_length_max',
    'mp_opd_grass_chunk_singleton_fraction',
    'mp_opd_grass_chunk_non_singleton_fraction',
    'mp_opd_grass_chunk_alpha_mean',
    'mp_opd_grass_chunk_alpha_median',
    'mp_opd_grass_chunk_alpha_atomic_fraction',
    'mp_opd_grass_chunk_alpha_hard_fraction',
    'mp_opd_grass_chunk_alpha_clip_low_fraction',
    'mp_opd_grass_chunk_alpha_clip_high_fraction',
    'mp_opd_grass_chunk_distortion_mean',
    'mp_opd_grass_chunk_variance_mean',
    'mp_opd_grass_chunk_gamma_mean',
    'mp_opd_grass_chunk_sigma2',
    'mp_opd_grass_chunk_degenerate',
    'mp_opd_grass_chunk_exact_geometry',
    'mp_opd_grass_chunk_gram_symmetry_error',
    'mp_opd_grass_chunk_atom_token_count',
    'mp_opd_grass_chunk_sure_gain_mean',
    'mp_opd_grass_chunk_sure_gain_median',
    'mp_opd_grass_chunk_sure_gain_positive_fraction',
    'mp_opd_grass_chunk_pathology_negative_d_fraction',
    'mp_opd_grass_chunk_pathology_negative_v_fraction',
    'mp_opd_grass_chunk_head_update_energy_atomic',
    'mp_opd_grass_chunk_head_update_energy_shrunk',
    'mp_opd_grass_chunk_head_update_cosine',
    'mp_opd_grass_chunk_credit_relative_l2_change',
    'mp_opd_grass_chunk_credit_sign_change_fraction',
    'mp_opd_grass_chunk_conservation_error',
    'mp_opd_grass_chunk_mapping_straddling_atoms',
    'mp_opd_grass_chunk_mapping_unaligned_atoms',
    'mp_opd_grass_chunk_mapping_noncontiguous_splits',
    'mp_opd_grass_chunk_gram_diagonal_mean',
    'mp_opd_grass_chunk_sigma_batch',
    'mp_opd_grass_chunk_sigma_batch_variance',
    'mp_opd_grass_chunk_noise_valid_pairs',
    'mp_opd_grass_chunk_noise_pairs_excluded',
    'mp_opd_grass_chunk_noise_z_abs_mean',
    'mp_opd_grass_chunk_noise_z_abs_p99',
    'mp_opd_grass_chunk_softcap',
    'mp_opd_grass_chunk_head_bias',
    'mp_opd_grass_chunk_boundary_adjacent_pairs_within',
    'mp_opd_grass_chunk_boundary_adjacent_pairs_cross',
    'mp_opd_grass_chunk_boundary_within_abs_diff_mean',
    'mp_opd_grass_chunk_boundary_cross_abs_diff_mean',
    'mp_opd_grass_chunk_boundary_within_normalized_diff_mean',
    'mp_opd_grass_chunk_boundary_cross_normalized_diff_mean',
    'mp_opd_grass_chunk_boundary_within_gradient_cosine_mean',
    'mp_opd_grass_chunk_boundary_cross_gradient_cosine_mean',
    'mp_opd_grass_chunk_shadow_hard_chunk_head_cosine_to_grass',
    'mp_opd_grass_chunk_shadow_atomic_head_cosine_to_grass',
    'mp_opd_grass_chunk_shadow_hard_chunk_credit_relative_l2_change',
)
# Emitted only when the response actually contains a chunk of that length, or a
# non-singleton chunk to sample. Reported, never required.
OPTIONAL_KEYS = (
    'mp_opd_grass_chunk_length_1_fraction',
    'mp_opd_grass_chunk_length_2_fraction',
    'mp_opd_grass_chunk_length_3_fraction',
    'mp_opd_grass_chunk_length_4_fraction',
    'mp_opd_grass_chunk_length_8_fraction',
    'mp_opd_grass_chunk_length_16_fraction',
    'mp_opd_grass_chunk_gram_sampled_chunks',
    'mp_opd_grass_chunk_gram_sampled_min_eigenvalue',
    'mp_opd_grass_chunk_gram_sampled_negative_fraction',
    'mp_opd_grass_chunk_gram_neighbour_cosine_mean',
    'mp_opd_grass_chunk_gram_neighbour_cosine_negative_fraction',
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
    # aligner, so this directory is not optional for GRASS-Chunk's native source.
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
        # The Gram must not be computed under TF32: at ~1e-3 relative error a PSD
        # head Gram starts producing materially negative D_c.
        TORCH_CUDA_MATRIX_ALLOW_TF32='0',
    )
    libs = ['/usr/local/cuda/lib64', '/usr/local/nvidia/lib64'] + [
        str(p) for p in Path('/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia').glob('*/lib')
    ]
    e['LD_LIBRARY_PATH'] = ':'.join(libs)
    return e


@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={'/runs': outputs})
def unit_tests(commit: str):
    """Run the suite on the real runtime torch, not only local CPU torch.

    Two runs, deliberately. The *gate* is the GRASS-Chunk suite plus the shared core
    it is built on - the head Gram, the noise estimator, the shrink, the hard
    pooling and the conservation check - because a regression in any of those is a
    regression here. The selector-only GRASS-DP tests (candidate search, margins,
    pathology counters over candidate spans) are not on any code path GRASS-Chunk
    reaches, so they run in the second pass and are reported, not gated. They are
    still executed and still written to the log: nothing about the full-suite result
    is hidden, it simply does not decide whether this feature's smoke passes.
    """
    root = Path('/runs')
    root.mkdir(exist_ok=True)
    e = env()
    e['KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT'] = '1'
    shared = (
        'atom_head_gram or weighted_gram or row_chunking or token_weight or '
        'softcap_factor or gram_survives or noise_estimator or loss_gradient or '
        'conservation or pooled_credits or chunk_span_ids'
    )
    # `-k` would apply to every file in a single invocation, so the gate runs in two
    # steps: the GRASS-Chunk suite and the other modes unfiltered, then the shared
    # core of test_grass_span.py selected by name.
    gate = [PYTHON, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
            'tests/mp_opd/test_grass_chunk.py',
            'tests/mp_opd/test_gbv_span.py',
            'tests/mp_opd/test_credit_oracle.py',
            'tests/mp_opd/test_random_partition_min_span.py']
    gate_shared = [PYTHON, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                   'tests/mp_opd/test_grass_span.py', '-k', shared]
    full = [PYTHON, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
            'tests/mp_opd', '--continue-on-collection-errors']

    def run(command):
        return subprocess.run(command, cwd='/opt/overlay', env=e, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1700)

    main_result = run(gate)
    shared_result = run(gate_shared)
    full_result = run(full)
    (root / 'unit-tests.log').write_text(
        '===== GATE: grass_chunk and the modes that share its machinery =====\n'
        + main_result.stdout
        + '\n===== GATE: shared core of test_grass_span =====\n'
        + shared_result.stdout
        + '\n===== RECORD ONLY: full tests/mp_opd (not gating) =====\n'
        + full_result.stdout
    )
    outputs.commit()
    for name, result in (('gate', main_result), ('shared', shared_result)):
        tail = result.stdout.strip().splitlines()[-3:]
        print(f'GRASS_CHUNK_UNIT_{name.upper()}_TAIL=' + json.dumps(tail), flush=True)
    full_tail = full_result.stdout.strip().splitlines()[-3:]
    print('GRASS_CHUNK_FULL_SUITE_TAIL=' + json.dumps(full_tail), flush=True)
    passed = main_result.returncode == 0 and shared_result.returncode == 0
    print('GRASS_CHUNK_UNIT_JSON=' + json.dumps({'commit': commit, 'passed': passed,
                                                 'full_suite_passed': full_result.returncode == 0}),
          flush=True)
    if not passed:
        raise RuntimeError('runtime unit tests failed')


def _metrics(text, keys):
    found = {}
    for key in keys:
        hits = re.findall(re.escape(key) + r':\s*([-\d.eE+]+)', text)
        if hits:
            found[key] = float(hits[-1])
    return found


def _ensure_placeholder(path: Path) -> str:
    """Create the placeholder projection and return its sha256.

    Generated in the container rather than committed: it is not an artifact anyone
    should reuse, and a real file on disk is enough to satisfy the digest check
    that ``_build_grass_aligner`` performs.
    """
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        script = (
            'import torch;'
            f'torch.save({{"indices": torch.zeros((4,4), dtype=torch.long),'
            ' "likelihoods": torch.zeros((4,4), dtype=torch.float32),'
            # A single closing brace, and no f-prefix: this line is not an f-string,
            # so `}}` would reach the interpreter as two closing braces and leave
            # torch.save with an unclosed dict argument.
            ' "__note__": "GRASS-Chunk plumbing placeholder; not a usable projection"}, '
            f'"{path}")'
        )
        done = subprocess.run([PYTHON, '-c', script], capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError('placeholder write failed: ' + done.stderr[-500:])
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


@app.function(image=image, gpu='B200', cpu=16, memory=98304, timeout=3000, retries=0,
               max_containers=1, volumes={'/assets': assets, '/runs': outputs})
def train_smoke(commit: str, source: str, geometry: str, updates: int):
    root = Path('/runs')
    root.mkdir(exist_ok=True)
    log_path = root / ('train-' + source + '-' + geometry + '.log')
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
    prompts = Path('/assets/smoke-prompts.parquet')
    pd.DataFrame({'messages': [[{'role': 'user', 'content': text}] for text in PROMPTS]}).to_parquet(prompts)
    teacher = Path('/assets/teacher')
    if not (teacher / 'config.json').is_file():
        teacher.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(TEACHER, local_dir=str(teacher))
        assets.commit()
    print('SMOKE_ASSETS=' + json.dumps({'student': str(student), 'teacher': str(teacher),
                                        'prompts': str(prompts)}), flush=True)

    projection = Path('/assets/xtoken_projection_smoke_placeholder.pt')
    placeholder_sha = _ensure_placeholder(projection)
    assets.commit()

    opts = dict(
        num_nodes=1, num_gpus_per_node=1, backend='fsdp2',
        student_name_or_path=str(student), teacher_name_or_path=str(teacher),
        attn_implementation='sdpa', num_epochs=updates, train_batch_size=SAMPLES,
        micro_train_batch_size=1, learning_rate=1e-6, lr_warmup_ratio=.05,
        lr_scheduler='cosine_with_min_lr', min_lr=0, weight_decay=0.,
        gradient_checkpointing=True, enable_sleep=True, bf16=True, seed=42,
        # Per-config output dirs: the trainer fails closed when a checkpoint already
        # exists, so sharing one save_path across runs would abort the second one.
        save_path='/runs/checkpoint-' + source + '-' + geometry,
        ckpt_path='/runs/checkpoints-' + source + '-' + geometry,
        train_dataset_path=str(prompts), input_key='messages', apply_chat_template=True,
        enable_thinking=False, max_samples=SAMPLES, prompt_max_len=0, max_len=2048,
        preprocess_num_workers=2, rollout_num_engines=1,
        rollout_disable_piecewise_cuda_graph=True, rollout_tp_size=1,
        rollout_mem_fraction_static=.25, rollout_batch_size=SAMPLES,
        generate_max_len=1024, n_samples_per_prompt=1, temperature=.6, top_p=.95,
        teacher_tp_size=1, teacher_pp_size=1, teacher_ep_size=1, teacher_dp_size=1,
        teacher_mem_fraction_static=.3, teacher_context_length=4096,
        teacher_forward_n_batches=8, kd_algorithm='mp_opd', mp_opd_mode='grass_chunk',
        mp_opd_grass_geometry=geometry, mp_opd_grass_sigma_rho=SIGMA_RHO,
        mp_opd_grass_sigma_min_pairs=SIGMA_MIN_PAIRS,
        mp_opd_grass_chunk_source=source, mp_opd_grass_chunk_run_length=RUN_LENGTH,
        mp_opd_grass_chunk_straddle='singleton', mp_opd_grass_chunk_shadow=True,
        xtoken_projection_path=str(projection), xtoken_projection_sha256=placeholder_sha,
        xtoken_max_comb_len=4,
        kd_loss_fn='rkl', kd_ratio=1.,
        span_score_mode='mean_logprob', exact_token_trajectory=True,
        enforce_max_sequence_length=True, diagnostic_max_updates=updates,
        diagnostic_collapse_gate=False, save_steps=1000, logging_steps=1,
        use_wandb=False,
    )
    command = [PYTHON, '-m', 'kdflow.cli.train_kd_on_policy']
    for key, value in opts.items():
        command += ['--' + key, str(value)]
    (root / ('invocation-' + source + '-' + geometry + '.json')).write_text(json.dumps(dict(
        purpose='grass_chunk code-path smoke, not a result', source_commit=commit,
        chunk_source=source, geometry=geometry, teacher=TEACHER, student=str(student),
        updates=updates, samples=SAMPLES, run_length=RUN_LENGTH, sigma_rho=SIGMA_RHO,
        sigma_min_pairs=SIGMA_MIN_PAIRS, shadow=True,
        projection_placeholder_note=PLACEHOLDER_NOTE, projection_placeholder_sha=placeholder_sha,
        caveat=('student and teacher share a tokenizer, so the xtoken chunks here are '
                'near-identity and say nothing about real cross-tokenizer chunk quality'),
        command=command), indent=2))
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
        seen = _metrics(text, REQUIRED_KEYS + OPTIONAL_KEYS)
        missing = [key for key in REQUIRED_KEYS if key not in seen]
        summary = {'commit': commit, 'chunk_source': source, 'geometry': geometry,
                   'updates': updates, 'metrics_seen': seen, 'missing': missing,
                   'optional_absent': [key for key in OPTIONAL_KEYS if key not in seen],
                   'note': 'code-path smoke only; no efficacy claim'}
        (root / ('grass-chunk-smoke-' + source + '-' + geometry + '.json')).write_text(
            json.dumps(summary, indent=2))
        outputs.commit()
        print('GRASS_CHUNK_SMOKE_JSON=' + json.dumps(summary), flush=True)
        if missing:
            raise RuntimeError('grass_chunk telemetry missing: ' + ','.join(missing))
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
def main(updates: int = UPDATES, sources: str = ','.join(SOURCES), geometry: str = GEOMETRY):
    # core.autocrlf=true is forced because this repository has no .gitattributes:
    # the tree was checked out with CRLF by a Windows git whose *global* config sets
    # it, so a git run under a different HOME (WSL) sees every file as modified and
    # the gate below would refuse to launch on 531 phantom diffs. Forcing it makes
    # the check mean the same thing from either host.
    git = ['git', '-c', 'core.autocrlf=true']
    dirty = subprocess.check_output(
        git + ['status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError('Commit tracked changes first:\n' + dirty[:2000])
    commit = subprocess.check_output(
        git + ['rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    print('SMOKE_SOURCE_COMMIT=' + commit, flush=True)
    unit_tests.remote(commit)
    results = []
    for source in [s.strip() for s in sources.split(',') if s.strip()]:
        if source not in SOURCES:
            raise ValueError('unknown chunk source ' + source)
        results.append(train_smoke.remote(commit, source, geometry, updates))
    print('GRASS_CHUNK_SMOKE_ALL=' + json.dumps(results), flush=True)
