"""Phase 0.5: real cross-tokenizer identity probe on Gemma-2 <-> Qwen2.5.

Purpose (brief section 4): the Phase 0 GPU probe validated the credit path on a
same-tokenizer Qwen pair where every atom had ``w_i = 1``. Before any scientific
branch experiment, the identity operator must be shown to reproduce the historical
pooled Atomic loss on the *actual research tokenizer mismatch*, where ``w_i > 1``
atoms really exist. This is an engineering equivalence test, not a quality
experiment: nothing here says any credit operator is better or worse.

Assets, and why they are what they are (brief section 4.1 asked for the campaign's
fine-tuned Gemma-2 student):

* teacher = ``Qwen/Qwen2.5-7B-Instruct`` at the revision the campaign manifest pins,
  which is the campaign teacher;
* student = ``google/gemma-2-2b-it`` **public base**, from the transfer volume that
  the campaign itself staged, whose own manifest says
  ``"student_lineage": "public-base; not company qwen-gemma-sft checkpoint"``.

The campaign's fine-tuned student checkpoint lives only on company storage and is not
on the Hub, so it is not reachable from Modal. None of the five gate conditions in
section 4.4 depends on the student's weights: they depend on the tokenizer pair
(Gemma SentencePiece 256k vs Qwen byte-BPE) and on the code path. A fine-tune does not
change the tokenizer, so the mismatch under test is the real one. Every number this
probe produces is therefore labelled with that substitution.

Run from WSL with profile ``lhtu05``:
    bash tmp/run_phase05.sh
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/opt/overlay')
PYTHON = '/opt/venvs/simct-b200/bin/python'
IMAGE = 'docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f'
RUN = 'cross-atom-phase05-20261006-r1'
STAGED_VOLUME = 'simct-qwen7b-gemma2-assets-20260915'
STAGED_MOUNT = '/staged'
STUDENT_DIR = STAGED_MOUNT + '/student'
# The staged student is the public base, so the campaign SFT checkpoint (and, for
# Phase 1, the mid/step312 training checkpoints) has to be pointed at explicitly once
# it is reachable. Everything below resolves through this override rather than
# hard-coding the staged path, so swapping in the real checkpoint is one argument.
STUDENT_OVERRIDE = ''
TEACHER_REPO = 'Qwen/Qwen2.5-7B-Instruct'
# Revision pinned by the campaign's own model-manifest.json, so the teacher is the
# same weights the campaign used rather than whatever main resolves to today.
TEACHER_REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
TEACHER_TOKENIZER_FILES = (
    'config.json',
    'generation_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'vocab.json',
    'merges.txt',
    'special_tokens_map.json',
    'added_tokens.json',
)
# Pre-registered gate thresholds, fixed before the run so the numbers cannot pick them.
ATOMIZATION_SUCCESS_FLOOR = 0.5
IDENTITY_RELATIVE_TOLERANCE = 1e-5

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
    # The fixed Phase 0.5 text corpus lives here and is read by both this harness and
    # the tokenizer-level tests, so there is exactly one copy of it.
    .add_local_dir(str(ROOT / 'experiments/mp_opd'), '/opt/overlay/experiments/mp_opd',
                   ignore=['**/__pycache__/**', '**/*.pyc'])
)
assets = modal.Volume.from_name('cross-atom-credit-assets', create_if_missing=True)
staged = modal.Volume.from_name(STAGED_VOLUME, create_if_missing=False)
outputs = modal.Volume.from_name(RUN, create_if_missing=True)

TEXT_CORPUS = ROOT / 'experiments/mp_opd/cross_atom_phase05_texts.json'


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
        OMP_NUM_THREADS='8',
        WANDB_SILENT='true',
        KDFLOW_TRUST_REMOTE_CODE='0',
    )
    libs = ['/usr/local/cuda/lib64', '/usr/local/nvidia/lib64'] + [
        str(p) for p in Path('/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia').glob('*/lib')
    ]
    e['LD_LIBRARY_PATH'] = ':'.join(libs)
    return e


def ensure_teacher(tokenizer_only: bool) -> dict:
    """Materialise the campaign teacher, cheaply when only the tokenizer is needed.

    The file list comes from the pinned revision rather than from a hard-coded list,
    because a repo does not necessarily contain every optional tokenizer file (the
    first attempt 404'd on ``special_tokens_map.json``).

    Offline flags must be cleared *before* huggingface_hub is first imported:
    ``transformers`` pulls it in, and it freezes ``HF_HUB_OFFLINE`` into a module
    constant at import time, so setting the variable later has no effect.
    """
    os.environ['HF_HOME'] = '/assets/hf'
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['HF_DATASETS_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    teacher = Path('/assets/teacher')
    teacher.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    import huggingface_hub.constants as hf_constants

    if getattr(hf_constants, 'HF_HUB_OFFLINE', False):
        hf_constants.HF_HUB_OFFLINE = False

    available = sorted(HfApi().list_repo_files(TEACHER_REPO, revision=TEACHER_REVISION))
    tokenizer_files = [name for name in available if name in TEACHER_TOKENIZER_FILES]
    weights = [name for name in available
               if name.endswith('.safetensors') or name.endswith('.safetensors.index.json')]
    for name in tokenizer_files:
        if not (teacher / name).is_file():
            hf_hub_download(TEACHER_REPO, name, revision=TEACHER_REVISION,
                            local_dir=str(teacher))
    if not tokenizer_only and not (teacher / 'model.safetensors.index.json').is_file():
        if not weights:
            raise RuntimeError('no safetensors in %s@%s' % (TEACHER_REPO, TEACHER_REVISION))
        snapshot_download(TEACHER_REPO, revision=TEACHER_REVISION, local_dir=str(teacher),
                          allow_patterns=weights)
    assets.commit()
    return {'teacher_repo': TEACHER_REPO, 'teacher_revision': TEACHER_REVISION,
            'repo_files': available, 'tokenizer_files_used': tokenizer_files,
            'weight_files': weights}


def resolve_student_dir(override: str) -> str:
    """Where the student weights come from: the caller's path, or the staged base."""
    chosen = (override or STUDENT_OVERRIDE or '').strip()
    return chosen or STUDENT_DIR


def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def student_lineage(student_dir: str) -> dict:
    """Provenance for the student, including the campaign's own label for it."""
    manifest_path = Path(STAGED_MOUNT) / 'model-manifest.json'
    declared = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        declared = {
            'campaign_manifest_student_repo': manifest.get('student', {}).get('repo'),
            'campaign_manifest_student_revision': manifest.get('student', {}).get('revision'),
            'campaign_manifest_teacher_repo': manifest.get('teacher', {}).get('repo'),
            'campaign_manifest_teacher_revision': manifest.get('teacher', {}).get('revision'),
            'campaign_manifest_student_lineage': manifest.get('student_lineage'),
            'campaign_manifest_image': manifest.get('image'),
        }
    student = Path(student_dir)
    local = {}
    for name in ('config.json', 'model.safetensors.index.json', 'tokenizer.model',
                 'tokenizer.json', 'tokenizer_config.json'):
        candidate = student / name
        if candidate.is_file():
            local[name] = {'bytes': candidate.stat().st_size,
                           'sha256': file_digest(candidate)}
    total = sum(p.stat().st_size for p in student.glob('*.safetensors'))
    local['safetensors_total_bytes'] = total
    return {'declared': declared, 'student_dir': student_dir, 'staged_files': local,
            'is_staged_public_base': os.path.realpath(student_dir)
            == os.path.realpath(STUDENT_DIR)}


@app.function(image=image, cpu=8, memory=32768, timeout=2400,
              volumes={'/assets': assets, STAGED_MOUNT: staged, '/runs': outputs})
def tokenizer_tests(commit: str, student_dir: str = ''):
    """Cross-tokenizer assertions that need no weights: tokenizers plus real text."""
    root = Path('/runs')
    root.mkdir(exist_ok=True)
    resolved = resolve_student_dir(student_dir)
    ensure_teacher(tokenizer_only=True)
    e = env(online=False)
    e['CA_STUDENT_DIR'] = resolved
    e['CA_TEACHER_DIR'] = '/assets/teacher'
    command = [PYTHON, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
               'tests/mp_opd/test_cross_atom_phase05.py',
               'tests/mp_opd/test_credit_transform.py']
    done = subprocess.run(command, cwd='/opt/overlay', env=e, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=2300)
    (root / 'tokenizer-tests.log').write_text(done.stdout)
    outputs.commit()
    tail = done.stdout.strip().splitlines()[-3:]
    print('PHASE05_TOKENIZER_TAIL=' + json.dumps(tail), flush=True)
    print('PHASE05_TOKENIZER_JSON=' + json.dumps(
        {'commit': commit, 'returncode': done.returncode, 'passed': done.returncode == 0}),
        flush=True)
    if done.returncode:
        raise RuntimeError('phase 0.5 tokenizer tests failed')


@app.function(image=image, gpu='B200', cpu=16, memory=131072, timeout=3600, retries=0,
              max_containers=1,
              volumes={'/assets': assets, STAGED_MOUNT: staged, '/runs': outputs})
def cross_tokenizer_probe(commit: str, student_dir: str = ''):
    """Identity vs historical Atomic on the real Gemma<->Qwen tokenizer mismatch."""
    # Before any import that pulls in huggingface_hub: it caches the offline flags.
    os.environ['HF_HOME'] = '/assets/hf'
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['HF_DATASETS_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kdflow.algorithms._mp_opd_atoms import SimCTAtomizer
    from kdflow.algorithms._mp_opd_credit import build_atom_credits, hard_partition_loss
    from kdflow.algorithms._mp_opd_credit_transform import (
        build_credit_transform,
        production_credit_batch,
    )

    root = Path('/runs')
    root.mkdir(exist_ok=True)
    resolved_student = resolve_student_dir(student_dir)
    teacher_files = ensure_teacher(tokenizer_only=False)
    corpus = json.loads(Path('/opt/overlay/experiments/mp_opd/cross_atom_phase05_texts.json').read_text())

    lineage = student_lineage(resolved_student)
    print('PHASE05_LINEAGE=' + json.dumps(lineage), flush=True)

    device = 'cuda'
    student_tok = AutoTokenizer.from_pretrained(resolved_student)
    teacher_tok = AutoTokenizer.from_pretrained('/assets/teacher')
    student = AutoModelForCausalLM.from_pretrained(
        resolved_student, torch_dtype=torch.bfloat16, attn_implementation='sdpa'
    ).to(device).eval()
    teacher = AutoModelForCausalLM.from_pretrained(
        '/assets/teacher', torch_dtype=torch.bfloat16, attn_implementation='sdpa'
    ).to(device).eval()
    atomizer = SimCTAtomizer(student_tok, teacher_tok)
    print('PHASE05_MODELS=' + json.dumps({
        'student': {'type': student.config.model_type, 'vocab': student.config.vocab_size},
        'teacher': {'type': teacher.config.model_type, 'vocab': teacher.config.vocab_size},
        'student_tokens_per_teacher_token': None,
    }), flush=True)

    def prompt_text(tokenizer, prompt):
        try:
            return tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}], tokenize=False,
                add_generation_prompt=True)
        except Exception:
            return prompt

    def legacy_atomic_loss(current_nll, base, weight):
        partition = tuple((index, index + 1) for index in range(current_nll.numel()))
        return hard_partition_loss(current_nll, base, weight, partition)

    per_sample = []
    failures = {}
    w_histogram = {}
    rows_with_w_gt_1 = 0
    atoms_with_w_gt_1 = 0
    total_atoms = 0
    max_w = 0
    drifts = []
    boundary_checks = []
    with torch.no_grad():
        for item in corpus['items']:
            record = {'id': item['id'], 'task': item['task']}
            try:
                sides = {}
                for tag, tokenizer, model in (('student', student_tok, student),
                                              ('teacher', teacher_tok, teacher)):
                    prompt_ids = tokenizer.encode(prompt_text(tokenizer, item['prompt']),
                                                  add_special_tokens=False)
                    response_ids = tokenizer.encode(item['response'], add_special_tokens=False)
                    eos = tokenizer.eos_token_id
                    if eos is not None and (not response_ids or response_ids[-1] != eos):
                        response_ids = response_ids + [eos]
                    ids = prompt_ids + response_ids
                    logits = model(torch.tensor([ids], device=device)).logits[0]
                    first = len(prompt_ids) - 1
                    last = len(ids) - 1
                    sides[tag] = {
                        'prompt_tokens': len(prompt_ids),
                        'response_ids': response_ids,
                        'logits': logits[first:last].float(),
                        'labels': torch.tensor(ids[first + 1:last + 1], device=device),
                    }
                atomized = atomizer.atomize(sides['student']['response_ids'],
                                            sides['teacher']['response_ids'],
                                            sample_id=item['id'])
                record['student_response_tokens'] = len(sides['student']['response_ids'])
                record['teacher_response_tokens'] = len(sides['teacher']['response_ids'])
                if not atomized.valid:
                    record['atomization'] = 'FAILED'
                    record['failure_reason'] = atomized.failure_reason
                    failures[atomized.failure_reason] = failures.get(atomized.failure_reason, 0) + 1
                    per_sample.append(record)
                    continue
                atoms = atomized.atoms
                credits = build_atom_credits(atoms, sides['student']['logits'],
                                             sides['student']['labels'],
                                             sides['teacher']['logits'],
                                             sides['teacher']['labels'])
                weight = credits.weight
                n = len(atoms)
                total_atoms += n
                w_int = [int(value) for value in weight.tolist()]
                for value in w_int:
                    w_histogram[str(value)] = w_histogram.get(str(value), 0) + 1
                over = sum(1 for value in w_int if value > 1)
                atoms_with_w_gt_1 += over
                rows_with_w_gt_1 += 1 if over else 0
                max_w = max(max_w, max(w_int))
                record.update({
                    'atomization': 'OK',
                    'atoms': n,
                    'one_to_one_atoms': sum(a.boundary_type == 'one_to_one' for a in atoms),
                    'multi_token_atoms': sum(a.boundary_type == 'multi_token' for a in atoms),
                    'student_token_count': int(weight.sum().item()),
                    'teacher_token_count': sum(a.teacher_token_count for a in atoms),
                    'w_min': min(w_int), 'w_max': max(w_int),
                    'w_mean': float(weight.mean().item()),
                    'w_atoms_gt_1': over,
                    'w_fraction_gt_1': over / n,
                    'terminal_masked_student': atomized.masked_student_eos,
                    'terminal_masked_teacher': atomized.masked_teacher_eos,
                    'covered_student_events': atomized.covered_student_events,
                    'covered_teacher_events': atomized.covered_teacher_events,
                    'rate_mean': float(credits.rate.mean().item()),
                    'rate_std': float(credits.rate.std(unbiased=False).item()),
                    'rate_min': float(credits.rate.min().item()),
                    'rate_max': float(credits.rate.max().item()),
                })
                # Terminal handling unchanged: the atoms plus the masked terminal EOS
                # must account for every token the tokenizer produced.
                record['terminal_accounting_ok'] = bool(
                    atomized.covered_student_events + atomized.masked_student_eos
                    == len(sides['student']['response_ids'])
                    and atomized.covered_teacher_events + atomized.masked_teacher_eos
                    == len(sides['teacher']['response_ids']))
                batch = production_credit_batch(credits.rate, credits.base_credit, credits.weight)
                legacy = float(legacy_atomic_loss(credits.current_nll, credits.base_credit,
                                                  credits.weight).detach().item())
                identity_out = build_credit_transform('identity')(batch, training=True)
                identity = float((identity_out.effective_credit * credits.current_nll)
                                 .sum().detach().item())
                epsilon = 1e-12
                drift = abs(identity - legacy) / max(abs(legacy), epsilon)
                drifts.append(drift)
                record['identity_loss'] = identity
                record['legacy_atomic_loss'] = legacy
                record['relative_drift'] = drift
                record['identity_bitwise_equal_to_rate'] = bool(
                    torch.equal(identity_out.effective_credit, credits.rate))
                record['telemetry_finite'] = all(
                    bool(torch.isfinite(value).all().item())
                    for value in identity_out.diagnostics.values())
                # Mask/boundary behaviour on real atoms: a forward neighbour operator
                # must leave the last atom at its own atomic credit (boundary fallback)
                # and must never move credit onto a token outside the atoms.
                forward_out = build_credit_transform('forward', lam=0.25)(batch, training=True)
                applied = forward_out.effective_credit
                record['last_atom_boundary_fallback'] = bool(
                    abs(float(applied[-1]) - float(credits.rate[-1])) <= 1e-6)
                record['forward_transfer_fraction'] = float(
                    forward_out.diagnostics['mp_opd_credit_transfer_fraction'].item())
                boundary_checks.append({
                    'id': item['id'],
                    'last_atom_fallback': record['last_atom_boundary_fallback'],
                    'terminal_accounting_ok': record['terminal_accounting_ok'],
                })
            except Exception as error:  # noqa: BLE001 - recorded, never silently swallowed
                record['atomization'] = 'ERROR'
                record['failure_reason'] = '%s: %s' % (type(error).__name__, error)
                failures[record['failure_reason']] = failures.get(record['failure_reason'], 0) + 1
            per_sample.append(record)

    atomized_rows = [row for row in per_sample if row.get('atomization') == 'OK']
    success_fraction = len(atomized_rows) / max(len(per_sample), 1)
    w_distribution = {
        'histogram': dict(sorted(w_histogram.items(), key=lambda kv: int(kv[0]))),
        'max_w': max_w,
        'atoms_with_w_gt_1': atoms_with_w_gt_1,
        'samples_with_w_gt_1': rows_with_w_gt_1,
        'fraction_atoms_w_gt_1': (atoms_with_w_gt_1 / total_atoms) if total_atoms else 0.0,
    }
    checks = {
        'has_w_gt_1_atom': atoms_with_w_gt_1 > 0,
        'atomization_success_fraction': success_fraction,
        'atomization_meets_floor': success_fraction >= ATOMIZATION_SUCCESS_FLOOR,
        'max_relative_drift': max(drifts) if drifts else None,
        'identity_within_tolerance': bool(drifts) and max(drifts) <= IDENTITY_RELATIVE_TOLERANCE,
        'identical_credit_bitwise': all(row.get('identity_bitwise_equal_to_rate', False)
                                        for row in atomized_rows),
        'terminal_accounting_ok': all(row.get('terminal_accounting_ok', False)
                                      for row in atomized_rows),
        'boundary_fallback_ok': all(row.get('last_atom_boundary_fallback', False)
                                    for row in atomized_rows),
        'telemetry_all_finite': all(row.get('telemetry_finite', False) for row in atomized_rows),
    }
    gate_keys = ('has_w_gt_1_atom', 'atomization_meets_floor', 'identity_within_tolerance',
                 'identical_credit_bitwise', 'terminal_accounting_ok', 'boundary_fallback_ok',
                 'telemetry_all_finite')
    gate_passed = all(checks[key] for key in gate_keys)
    summary = {
        'commit': commit,
        'profile': 'lhtu05',
        'phase': '0.5',
        'pre_registered_thresholds': {
            'atomization_success_floor': ATOMIZATION_SUCCESS_FLOOR,
            'identity_relative_tolerance': IDENTITY_RELATIVE_TOLERANCE,
        },
        'pair': {
            'student_dir': resolved_student,
            'student_is_staged_public_base': lineage['is_staged_public_base'],
            'student_repo_declared': lineage['declared'].get('campaign_manifest_student_repo'),
            'student_revision_declared': lineage['declared'].get('campaign_manifest_student_revision'),
            'student_lineage_declared': lineage['declared'].get('campaign_manifest_student_lineage'),
            'student_is_campaign_checkpoint': not lineage['is_staged_public_base'],
            'teacher_repo': TEACHER_REPO,
            'teacher_revision': TEACHER_REVISION,
            'teacher_is_campaign_teacher': True,
        },
        'teacher_files': teacher_files,
        'corpus': {'items': len(per_sample), 'atomized': len(atomized_rows)},
        'w_distribution': w_distribution,
        'failures': failures,
        'checks': checks,
        'gate': 'PASS' if gate_passed else 'FAIL',
        'per_sample': per_sample,
        'note': ('engineering equivalence test on fixed hand-written text; no rollout, '
                 'no optimizer step, and no statement about operator quality'),
    }
    (root / 'cross_atom_phase05_cross_tokenizer_probe.json').write_text(
        json.dumps(summary, indent=2))
    outputs.commit()
    print('PHASE05_SUMMARY=' + json.dumps({key: value for key, value in summary.items()
                                           if key != 'per_sample'}), flush=True)
    if not gate_passed:
        failed = [key for key in gate_keys if not checks[key]]
        raise RuntimeError('phase 0.5 gate FAILED: ' + ','.join(failed))
    return summary


@app.local_entrypoint()
def main(phases: str = 'tokens,probe', student_dir: str = ''):
    git = ['git', '-c', 'core.autocrlf=true']
    dirty = subprocess.check_output(
        git + ['status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError('Commit tracked changes first:\n' + dirty)
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    print('PHASE05_SOURCE_COMMIT=' + commit, flush=True)
    print('PHASE05_STUDENT_DIR=' + resolve_student_dir(student_dir), flush=True)
    selected = [item.strip() for item in phases.split(',') if item.strip()]
    unknown = [item for item in selected if item not in {'tokens', 'probe'}]
    if unknown:
        raise ValueError('unknown phases ' + ','.join(unknown))
    if 'tokens' in selected:
        tokenizer_tests.remote(commit, student_dir)
    if 'probe' in selected:
        result = cross_tokenizer_probe.remote(commit, student_dir)
        print('PHASE05_GATE=' + json.dumps({'gate': result['gate'],
                                            'checks': result['checks'],
                                            'w_distribution': result['w_distribution']}),
              flush=True)
    print('PHASE05_PHASES=' + json.dumps(selected), flush=True)
