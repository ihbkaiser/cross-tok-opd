#!/usr/bin/env python3
"""Run the cross-atom credit path on one GPU, with real atoms and real logits.

Node-side counterpart of ``experiments/modal/mp_opd_credit_kernel_smoke_modal.py``
(``--phases probe``). The same atomizer, the same ``build_atom_credits``, the same
``production_credit_batch``, the same operators, and the same historical pooled Atomic
loss as the regression oracle -- but every path comes from the command line, so it runs
on a company node instead of a Modal container, and the texts come either from a JSON
file or from the frozen student policy itself (``experiments/mp_opd/hf_frozen_sampler``,
which samples in PyTorch because the SGLang engine path is blocked upstream).

What this can show: the operators run on real atoms of a real checkpoint, ``identity``
reproduces the historical Atomic loss, telemetry stays finite, and a cross-tokenizer
pair really produces ``w_i > 1`` atoms. What it cannot show: anything about rollout,
training dynamics or utility.

Fail-closed: a drift above ``--identity-tol``, a non-finite metric, an operator that
never moves a credit, or a credit-scale inflation exits non-zero with the reason printed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_MARKER = Path('kdflow') / 'algorithms' / '_mp_opd_credit_transform.py'


def put_repo_on_path(explicit: str | None) -> Path:
    """Put the SimCT checkout on ``sys.path`` and return its root."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    here = Path(__file__).resolve()
    candidates.extend(here.parents)
    candidates.append(Path(os.environ['MP_SRC']) if os.environ.get('MP_SRC') else Path('.'))
    for root in candidates:
        try:
            if (root / REPO_MARKER).is_file():
                text = str(root.resolve())
                if text not in sys.path:
                    sys.path.insert(0, text)
                return root.resolve()
        except OSError:
            continue
    raise SystemExit(
        'PROBE_SETUP_FAIL: cannot locate the SimCT checkout; pass --repo-root <path> '
        'or export MP_SRC=<path> (expected %s under the checkout root)' % REPO_MARKER
    )


def git_revision(root: Path) -> str:
    try:
        out = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return ''
    return out.stdout.strip() if out.returncode == 0 else ''


def read_texts(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text())
    samples: list[dict[str, str]] = []
    for row in payload:
        if isinstance(row, dict):
            prompt, response = row.get('prompt'), row.get('response')
        elif isinstance(row, (list, tuple)) and len(row) == 2:
            prompt, response = row
        else:
            raise SystemExit('PROBE_SETUP_FAIL: unsupported texts record: %r' % (row,))
        if not prompt or not response:
            raise SystemExit('PROBE_SETUP_FAIL: texts record needs prompt and response')
        samples.append({'prompt': str(prompt), 'response': str(response)})
    return samples


def as_chat_cell(value: Any) -> list | None:
    """Return the chat turns of a prompt cell, or ``None`` if it is plain text.

    ``pandas.read_parquet`` hands back a column of Python lists as numpy arrays, so an
    ``isinstance(value, list)`` check alone silently mis-routes chat rows to ``str()``.
    """
    if isinstance(value, str):
        return None
    if hasattr(value, 'tolist'):  # numpy array of dicts
        value = value.tolist()
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def chat_turns(value: Any) -> list[dict[str, str]]:
    """Normalise a prompt cell to OpenAI turns.

    The repo converter is preferred so the probe renders prompts the same way training
    does. ``kdflow.datasets.utils`` pulls in torch and ``datasets`` at import time, so
    when that stack is absent the probe falls back to a minimal normaliser and says so
    instead of silently rendering something different.
    """
    cell = as_chat_cell(value)
    if cell is None:
        return [{'role': 'user', 'content': str(value)}]
    try:
        from kdflow.datasets.utils import convert_to_openai_messages
    except ImportError as exc:
        print('PROBE_PROMPT_CONVERTER: local (%s)' % exc.__class__.__name__, flush=True)
        turns = [{'role': turn.get('role', 'user'), 'content': turn.get('content', '')}
                 for turn in cell if isinstance(turn, dict)]
        if not turns:
            raise SystemExit('PROBE_SETUP_FAIL: chat cell holds no role/content turns')
        return turns
    return list(convert_to_openai_messages(cell))


def render_chat_prompt(value: Any, chat_template_fn: Any) -> str:
    """Render a chat cell the way ``prompts_dataset._build_prompt`` does.

    The campaign prompt files store ``messages`` (OpenAI turns), not a plain prompt
    string, so the probe must go through the same converter and the same
    ``tokenize=False, add_generation_prompt=True`` call instead of guessing a layout.
    """
    chat = chat_turns(value)
    # A prompt cell must not carry the gold answer: anything after the last user turn is
    # dropped, which is what ``sft_dataset.py`` does with ``messages[:-1]`` when it renders
    # the prompt half of a training pair.
    while len(chat) > 1 and chat[-1].get('role') == 'assistant':
        chat = chat[:-1]
        print('PROBE_PROMPT_DROP_ASSISTANT: kept %d turns' % len(chat), flush=True)
    for kwargs in ({'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False},
                   {'tokenize': False, 'add_generation_prompt': True}):
        try:
            return str(chat_template_fn(chat, **kwargs))
        except TypeError:
            continue
    raise SystemExit('PROBE_SETUP_FAIL: the student chat template rejected both '
                     'enable_thinking variants')


def read_prompts(path: Path, column: str | None,
                 chat_template_fn: Any = None) -> tuple[list[str], str]:
    try:
        import pandas as pd
        frame = pd.read_parquet(path)
        columns = list(frame.columns)
    except ImportError:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        columns = list(table.column_names)
        frame = table.to_pandas()
    name = column or next((c for c in columns if c in ('prompt', 'question', 'text', 'input')), None)
    if name is None and 'messages' in columns:
        name = 'messages'
    if name is None:
        raise SystemExit('PROBE_SETUP_FAIL: pass --prompt-column; columns are %s' % columns)
    mode = 'plain'
    prompts: list[str] = []
    for value in frame[name].tolist():
        cell = as_chat_cell(value)
        if cell is None:
            prompts.append(str(value))
            continue
        if chat_template_fn is None:
            raise SystemExit('PROBE_SETUP_FAIL: column %s holds chat turns but the '
                             'student tokenizer exposes no chat template' % name)
        prompts.append(render_chat_prompt(cell, chat_template_fn))
        mode = 'chat_template'
    print('PROBE_PROMPTS: column=%s rows=%d mode=%s' % (name, len(prompts), mode), flush=True)
    return prompts, mode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo-root', default=None,
                        help='SimCT checkout root; falls back to this file parents or $MP_SRC')
    parser.add_argument('--student', required=True, help='student checkpoint directory')
    parser.add_argument('--student-tokenizer', default=None,
                        help='tokenizer directory; defaults to --student')
    parser.add_argument('--teacher', required=True, help='teacher model directory')
    parser.add_argument('--teacher-tokenizer', default=None,
                        help='tokenizer directory; defaults to --teacher')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--texts', default=None, help='JSON file of {prompt, response} pairs')
    source.add_argument('--prompts', default=None, help='parquet file of prompts only')
    parser.add_argument('--prompt-column', default=None,
                        help='prompt column name; auto-detected when omitted')
    parser.add_argument('--launch-config', default=None,
                        help='campaign launch-config.json, read for sampling parity (--prompts); '
                             'when given it wins over --temperature/--top-p/--max-context-tokens')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--max-context-tokens', type=int, default=4096)
    parser.add_argument('--response-tokens', type=int, default=256,
                        help='tokens to sample per prompt when --prompts is used (0 = no sampling)')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--samples', type=int, default=8, help='max samples to probe')
    parser.add_argument('--max-response-tokens', type=int, default=1024,
                        help='skip a sample whose response exceeds this many tokens')
    parser.add_argument('--lam', type=float, default=0.25)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--kernel', default='uniform')
    parser.add_argument('--direction', default='forward')
    parser.add_argument('--shuffle-seed', type=int, default=43)
    parser.add_argument('--external-alpha', type=float, default=0.25)
    parser.add_argument('--external-scale-match', default='rms')
    parser.add_argument('--identity-tol', type=float, default=1e-5,
                        help='max relative drift of identity vs the pooled Atomic oracle')
    parser.add_argument('--rms-tol', type=float, default=1.05)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--attn', default='sdpa', choices=('sdpa', 'eager'))
    parser.add_argument('--out', default='credit-path-probe.json')
    args = parser.parse_args(argv)

    root = put_repo_on_path(args.repo_root)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ``kdflow.algorithms.__init__`` imports every algorithm module unless this flag is
    # set, and ``xtoken.py`` needs ``xtoken_upstream_token_aligner``, which is vendored
    # only for the Modal images. The unit tests get this from ``tests/mp_opd/conftest.py``
    #; a script run does not. Set it the same way ``experiments/mp_opd/real_oracle.py``
    # and ``toy_oracle.py`` do, before the first ``kdflow`` import.
    os.environ.setdefault('KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT', '1')

    from kdflow.algorithms._mp_opd_atoms import SimCTAtomizer
    from kdflow.algorithms._mp_opd_credit import build_atom_credits, hard_partition_loss
    from kdflow.algorithms._mp_opd_credit_transform import (
        CREDIT_TRANSFORM_CHOICES,
        build_credit_transform,
        production_credit_batch,
    )

    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
             'float32': torch.float32}[args.dtype]

    student_path = Path(args.student)
    teacher_path = Path(args.teacher)
    if not student_path.is_dir():
        raise SystemExit('PROBE_SETUP_FAIL: --student is not a directory: %s' % student_path)
    if not teacher_path.is_dir():
        raise SystemExit('PROBE_SETUP_FAIL: --teacher is not a directory: %s' % teacher_path)

    student_tok = AutoTokenizer.from_pretrained(args.student_tokenizer or str(student_path))
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_tokenizer or str(teacher_path))

    def load_model(path: str):
        """``torch_dtype`` is the 4.x name and was renamed to ``dtype`` in 5.x.

        The 4.x name is tried first because on some versions ``dtype`` is swallowed
        as a config kwarg instead of raising, which would silently load fp32.
        """
        try:
            return AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=dtype, attn_implementation=args.attn)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(
                path, dtype=dtype, attn_implementation=args.attn)

    student = load_model(str(student_path)).to(args.device).eval()
    teacher = load_model(str(teacher_path)).to(args.device).eval()
    atomizer = SimCTAtomizer(student_tok, teacher_tok)

    if args.texts:
        samples = read_texts(Path(args.texts))[:args.samples]
        source_note = 'texts:%s' % args.texts
    else:
        prompts, prompt_mode = read_prompts(Path(args.prompts), args.prompt_column,
                                             getattr(student_tok, 'apply_chat_template', None))
        prompts = prompts[:args.samples]
        if args.response_tokens < 1:
            raise SystemExit('PROBE_SETUP_FAIL: --response-tokens must be >= 1 with --prompts')
        from experiments.mp_opd.hf_frozen_sampler import (
            HFFrozenPolicySampler,
            SamplingConfig,
            sampling_config_from_launch_config,
            terminal_token_ids,
        )
        # The policy that generates is the student, so the stop set must be the
        # student's own turn terminators. Taking the teacher's eos here hands the
        # sampler an id that is an ordinary token in the student vocabulary, so
        # rollouts never stop at the turn boundary and keep emitting turn markers,
        # which the atomizer then rejects as unsupported added tokens. The repo
        # already defines this set for the sampler, so reuse it.
        stops = list(terminal_token_ids(student_tok))
        print('PROBE_STOP_TOKENS: %s (student policy)' % stops, flush=True)
        campaign = (sampling_config_from_launch_config(
            args.launch_config, stop_token_ids=tuple(stops))
            if args.launch_config else SamplingConfig())
        config = SamplingConfig(
            temperature=campaign.temperature if args.launch_config else args.temperature,
            top_p=campaign.top_p if args.launch_config else args.top_p,
            top_k=campaign.top_k if args.launch_config else args.top_k,
            max_new_tokens=args.response_tokens,
            stop_strings=campaign.stop_strings,
            stop_token_ids=campaign.stop_token_ids,
            max_context_tokens=(campaign.max_context_tokens if args.launch_config
                                else args.max_context_tokens),
        )
        sampler = HFFrozenPolicySampler(student, student_tok, config,
                                        student_revision=git_revision(root), device=args.device)
        samples = []
        for index, prompt in enumerate(prompts):
            try:
                rollout = sampler.sample(prompt, seed=args.seed + index)
            except ValueError as exc:  # prompt too long for the context window
                print('PROBE_SAMPLE_SKIP index=%d reason=%s' % (index, exc), flush=True)
                continue
            text = rollout.text.strip()
            if not text:
                print('PROBE_SAMPLE_SKIP index=%d reason=empty_continuation' % index, flush=True)
                continue
            samples.append({'prompt': prompt, 'response': text})
        source_note = 'prompts:%s seed=%d prompt_render=%s campaign=%s probe=%s' % (
            args.prompts, args.seed, prompt_mode, campaign.as_dict(), config.as_dict())

    device_name = (torch.cuda.get_device_name(args.device) if args.device.startswith('cuda')
                   else 'cpu')
    free_bytes, total_bytes = (torch.cuda.mem_get_info(args.device)
                               if args.device.startswith('cuda') else (0, 0))
    print('PROBE_ENV: host=%s gpu_visible=%s device=%s (%s) free_gib=%.1f/%.1f torch=%s '
          'transformers=%s attn=%s dtype=%s'
          % (platform.node(), os.environ.get('CUDA_VISIBLE_DEVICES', '<all>'), args.device,
             device_name, free_bytes / 2 ** 30, total_bytes / 2 ** 30, torch.__version__,
             __import__('transformers').__version__, args.attn, args.dtype), flush=True)
    print('PROBE_ENV: repo=%s rev=%s' % (root, git_revision(root) or '<not a git checkout>'),
          flush=True)
    print('PROBE_ENV: student=%s teacher=%s source=%s' % (student_path, teacher_path, source_note),
          flush=True)
    print('PROBE_ENV: student_vocab=%d teacher_vocab=%d samples=%d'
          % (len(student_tok), len(teacher_tok), len(samples)), flush=True)

    def legacy_atomic_loss(current_nll, base, weight):
        """The historical Atomic objective: the singleton partition of the pooled loss."""
        partition = tuple((index, index + 1) for index in range(current_nll.numel()))
        return hard_partition_loss(current_nll, base, weight, partition)

    operators = list(CREDIT_TRANSFORM_CHOICES)
    per_operator: dict[str, list[dict[str, Any]]] = {name: [] for name in operators}
    per_operator['legacy_atomic'] = []
    atom_report: list[dict[str, Any]] = []
    invalid = 0
    skipped_long = 0

    with torch.no_grad():
        for index, sample in enumerate(samples):
            prompt, response = sample['prompt'], sample['response']
            row: dict[str, Any] = {}
            for tag, tokenizer, model in (('student', student_tok, student),
                                          ('teacher', teacher_tok, teacher)):
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                response_ids = tokenizer.encode(response, add_special_tokens=False)
                eos = tokenizer.eos_token_id
                if eos is not None and (not response_ids or response_ids[-1] != eos):
                    response_ids = response_ids + [eos]
                if len(response_ids) > args.max_response_tokens:
                    skipped_long += 1
                    row[tag] = None
                    break
                ids = prompt_ids + response_ids
                logits = model(torch.tensor([ids], device=args.device)).logits[0]
                # Rows that predict a response token: the last prompt row predicts the
                # first response token, and the final row predicts nothing.
                first = len(prompt_ids) - 1
                last = len(ids) - 1
                row[tag] = (logits[first:last].float(),
                            torch.tensor(ids[first + 1:last + 1], device=args.device),
                            response_ids)
            if row.get('student') is None or row.get('teacher') is None:
                print('PROBE_SAMPLE_SKIP index=%d reason=response_too_long' % index, flush=True)
                continue
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
            weight = credits.weight
            atom_report.append({
                'index': index,
                'atoms': len(atoms),
                'one_to_one': sum(atom.boundary_type == 'one_to_one' for atom in atoms),
                'w_gt_1': int((weight > 1).sum().item()),
                'student_tokens': int(weight.sum().item()),
                'masked_student_eos': atomized.masked_student_eos,
                'w_mean': float(weight.mean().item()),
                'w_max': float(weight.max().item()),
                'base_mean': float(credits.base_credit.mean().item()),
                'base_std': float(credits.base_credit.std(unbiased=False).item()),
                'rate_mean': float(credits.rate.mean().item()),
                'rate_std': float(credits.rate.std(unbiased=False).item()),
                'rate_min': float(credits.rate.min().item()),
                'rate_max': float(credits.rate.max().item()),
            })
            batch = production_credit_batch(credits.rate, credits.base_credit, weight)
            legacy = float(
                legacy_atomic_loss(credits.current_nll, credits.base_credit, weight)
                .detach().item())
            per_operator['legacy_atomic'].append({'loss': legacy})
            for name in operators:
                if name == 'external':
                    # The external operator is the offline augmentation arm; give it the
                    # local directional signal the branch probe would supply instead.
                    direction = (credits.rate - credits.rate.mean()).detach()
                    transform = build_credit_transform(name, alpha=args.external_alpha,
                                                      scale_match=args.external_scale_match)
                    operator_batch = production_credit_batch(
                        credits.rate, credits.base_credit, weight, {'future_advantage': direction})
                else:
                    transform = build_credit_transform(
                        name, lam=args.lam, horizon=args.horizon, kernel=args.kernel,
                        direction=args.direction, shuffle_seed=args.shuffle_seed)
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
                    'corr': float(output.diagnostics['mp_opd_credit_corr_effective_atomic'].item()),
                })

    summary: dict[str, Any] = {
        'repo_rev': git_revision(root),
        'student': str(student_path),
        'teacher': str(teacher_path),
        'source': source_note,
        'samples': len(samples),
        'atomized': len(atom_report),
        'invalid_samples': invalid,
        'skipped_too_long': skipped_long,
        'parameters': {'lam': args.lam, 'horizon': args.horizon, 'kernel': args.kernel,
                       'direction': args.direction, 'shuffle_seed': args.shuffle_seed,
                       'external_alpha': args.external_alpha,
                       'external_scale_match': args.external_scale_match,
                       'identity_tol': args.identity_tol, 'rms_tol': args.rms_tol},
        'atoms': atom_report,
        'operators': {},
    }
    failures: list[str] = []
    for name, rows in per_operator.items():
        if not rows:
            continue
        summary['operators'][name] = {
            'mean_loss': sum(row['loss'] for row in rows) / len(rows),
            'max_relative_drift': (max(row['relative_drift_vs_legacy'] for row in rows)
                                   if name != 'legacy_atomic' else 0.0),
            'max_rms_ratio': (max(row['rms_effective'] / max(row['rms_atomic'], 1e-12)
                                  for row in rows) if name != 'legacy_atomic' else 1.0),
            'min_transfer_fraction': (min(row['transfer_fraction'] for row in rows)
                                      if name != 'legacy_atomic' else 0.0),
            'all_finite': all(row.get('finite', True) for row in rows),
        }
    for name, stats in summary['operators'].items():
        if name in ('identity', 'legacy_atomic'):
            if stats['max_relative_drift'] > args.identity_tol:
                failures.append('%s drift %.3g > %.3g' % (name, stats['max_relative_drift'],
                                                          args.identity_tol))
        else:
            if not stats['all_finite']:
                failures.append('%s emitted a non-finite metric' % name)
            if stats['min_transfer_fraction'] <= 0.0:
                failures.append('%s never moved a credit' % name)
            if stats['max_rms_ratio'] > args.rms_tol:
                failures.append('%s inflated the credit scale (%.3g)' % (name, stats['max_rms_ratio']))
    if not summary['atomized']:
        failures.append('no sample atomized; the probe proved nothing')
    summary['failures'] = failures

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print('PROBE_ARTIFACT=%s' % out.resolve(), flush=True)
    for name, stats in summary['operators'].items():
        print('PROBE_OPERATOR %-14s mean_loss=%12.6f drift=%.3g rms_ratio=%.4f min_transfer=%.3f '
              'finite=%s' % (name, stats['mean_loss'], stats['max_relative_drift'],
                              stats['max_rms_ratio'], stats['min_transfer_fraction'],
                              stats['all_finite']), flush=True)
    print('CREDIT_PATH_JSON=' + json.dumps({k: v for k, v in summary.items() if k != 'atoms'}),
          flush=True)
    print('PROBE_FAILURES=' + json.dumps(failures), flush=True)
    if failures:
        print('PROBE_VERDICT=FAIL', flush=True)
        return 3
    print('PROBE_VERDICT=PASS', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())