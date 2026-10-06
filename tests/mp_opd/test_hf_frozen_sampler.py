"""HF frozen-policy sampler: determinism, logprob contract, filters and stops.

Brief section 17 lists six sampler tests. All of them run without a GPU and without
transformers except the last, which needs the real Gemma/Qwen tokenizers and therefore
skips explicitly when the fixture is absent.

The stubs are deliberate: they make the *expected* numbers computable by hand, which is
the only way to test that the recorded log-probability has the engine's semantics rather
than the sampler's own renormalised one.
"""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.mp_opd.hf_frozen_sampler import (  # noqa: E402
    FINISHED_CONTEXT_LIMIT,
    FINISHED_EOS,
    FINISHED_MAX_NEW_TOKENS,
    FINISHED_STOP_STRING,
    HFFrozenPolicySampler,
    SamplingConfig,
    filter_logits,
    sampling_config_from_launch_config,
    terminal_token_ids,
)

VOCAB = 256
EOS = 0


class StubTokenizer:
    """One token per character: encode and decode are exact inverses over ASCII."""

    def __init__(self, eos_token_id=EOS, added=None, vocab_size=VOCAB):
        self.eos_token_id = eos_token_id
        self._added = dict(added or {})
        self.vocab_size = vocab_size

    def get_added_vocab(self):
        return dict(self._added)

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return ''.join(chr(int(value)) for value in ids)


class StubModel:
    """Logits depend only on the last input token, so cached and uncached agree."""

    def __init__(self, logits_fn, with_cache=False):
        self.logits_fn = logits_fn
        self.with_cache = with_cache
        self.calls = 0

    def __call__(self, input_ids, use_cache=False, past_key_values=None):
        self.calls += 1
        last = int(input_ids[0, -1].item())
        logits = self.logits_fn(last).view(1, 1, VOCAB)
        cache = (self.calls,) if self.with_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cache)


def peaked(last, *, winner=None, sharpness=6.0, seed_offset=0):
    """A distribution with a clear winner and a finite tail."""
    generator = torch.Generator().manual_seed(1000 + last + seed_offset)
    logits = torch.randn(VOCAB, generator=generator) * 0.5
    if winner is None:
        winner = (last + 1) % VOCAB
    logits[winner] += sharpness
    return logits


def sampler(model=None, tokenizer=None, **config_kwargs):
    config = SamplingConfig(**config_kwargs)
    model = model or StubModel(peaked)
    tokenizer = tokenizer or StubTokenizer()
    return HFFrozenPolicySampler(model, tokenizer, config, student_revision='stub')


# ---------------------------------------------------------------------------
# 17.1 determinism
# ---------------------------------------------------------------------------


def test_sampling_is_deterministic_under_a_fixed_seed():
    first = sampler(max_new_tokens=12).sample('abc', seed=7)
    again = sampler(max_new_tokens=12).sample('abc', seed=7)
    assert first.token_ids == again.token_ids
    assert first.logprobs == again.logprobs


def test_different_seeds_diverge_when_the_distribution_is_wide():
    # A sharp stub would mask the RNG entirely, so this uses a flat one on purpose.
    def flat(last):
        return torch.zeros(VOCAB)

    engine = sampler(model=StubModel(flat), top_p=1.0, max_new_tokens=12)
    runs = {engine.sample('abc', seed=seed).token_ids for seed in range(12)}
    assert len(runs) > 1


def test_cached_and_uncached_paths_produce_the_same_tokens():
    plain = sampler(model=StubModel(peaked, with_cache=False), max_new_tokens=10)
    cached = sampler(model=StubModel(peaked, with_cache=True), max_new_tokens=10)
    assert plain.sample('abc', seed=11).token_ids == cached.sample('abc', seed=11).token_ids


def test_branch_draws_use_independent_seeded_streams():
    engine = sampler(max_new_tokens=8)
    branches = engine.sample_branches([1, 2, 3], 4, base_seed=100)
    assert len(branches) == 4
    assert [branch.seed for branch in branches] == [100, 101, 102, 103]
    for index, branch in enumerate(branches):
        direct = engine.sample(prompt_ids=[1, 2, 3], seed=100 + index)
        assert branch.token_ids == direct.token_ids


# ---------------------------------------------------------------------------
# 17.2 / 17.3 alignment and finiteness
# ---------------------------------------------------------------------------


def test_token_and_logprob_lengths_align_and_are_all_finite():
    rollout = sampler(max_new_tokens=16).sample('hello', seed=3)
    assert len(rollout.token_ids) == len(rollout.logprobs)
    assert rollout.generated_tokens == len(rollout.token_ids)
    assert all(math.isfinite(value) for value in rollout.logprobs)
    assert all(value < 0.0 for value in rollout.logprobs)
    assert len(rollout.content_ids) <= len(rollout.token_ids)


def test_a_sampled_token_never_reports_a_zero_log_probability():
    """The exact failure mode of the engine path cannot occur here."""
    for seed in range(30):
        rollout = sampler(max_new_tokens=24).sample('q', seed=seed)
        assert all(value != 0.0 for value in rollout.logprobs)


# ---------------------------------------------------------------------------
# The log-probability contract: unfiltered, temperature-scaled, engine-shaped
# ---------------------------------------------------------------------------


def test_recorded_logprob_is_the_unfiltered_temperature_scaled_value():
    temperature = 0.6
    model = StubModel(peaked)
    rollout = sampler(model=model, temperature=temperature, max_new_tokens=1).sample('a', seed=5)
    last_prompt_token = ord('a') % VOCAB
    scaled = peaked(last_prompt_token) / temperature
    expected = float(torch.log_softmax(scaled, dim=-1)[rollout.token_ids[0]])
    assert rollout.logprobs[0] == pytest.approx(expected, rel=1e-6)


def test_top_p_narrows_the_draw_but_does_not_change_the_reported_logprob():
    temperature = 0.6
    wide = sampler(model=StubModel(peaked), temperature=temperature,
                   top_p=1.0, max_new_tokens=1).sample('a', seed=5)
    narrow = sampler(model=StubModel(peaked), temperature=temperature,
                     top_p=0.01, max_new_tokens=1).sample('a', seed=5)
    scaled = peaked(ord('a') % VOCAB) / temperature
    unfiltered = float(torch.log_softmax(scaled, dim=-1)[narrow.token_ids[0]])
    assert narrow.logprobs[0] == pytest.approx(unfiltered, rel=1e-6)
    # A tiny nucleus must pick the mode, so the two draws differ from each other.
    assert narrow.token_ids[0] != wide.token_ids[0] or narrow.token_ids[0] == int(
        torch.argmax(scaled).item())


def test_top_k_one_is_the_argmax():
    rollout = sampler(model=StubModel(peaked), top_k=1, max_new_tokens=6).sample('a', seed=1)
    assert all(token == (ord('a') + 1) % VOCAB for token in rollout.token_ids[:1])


def test_temperature_is_honoured():
    def two_way(last):
        # A deep tail on purpose: with a shallow one the 254 tail tokens collectively
        # dominate a hot softmax and the test would measure the tail, not the two modes.
        logits = torch.full((VOCAB,), -60.0)
        logits[10] = 0.0
        logits[11] = 0.4
        return logits

    cold = sampler(model=StubModel(two_way), temperature=0.05, max_new_tokens=1)
    hot = sampler(model=StubModel(two_way), temperature=5.0, max_new_tokens=1)
    cold_tokens = {cold.sample('a', seed=seed).token_ids[0] for seed in range(120)}
    hot_tokens = {hot.sample('a', seed=seed).token_ids[0] for seed in range(120)}
    # Token 11 has the higher logit, so a cold temperature always takes it; a hot one
    # flattens the two and both appear.
    assert cold_tokens == {11}
    assert hot_tokens == {10, 11}


def test_filter_logits_keeps_at_least_one_token():
    logits = torch.full((8,), float('-inf'))
    logits[3] = 0.0
    filtered = filter_logits(logits, SamplingConfig(top_p=0.01))
    assert int(torch.isfinite(filtered).sum()) == 1
    assert int(torch.argmax(filtered)) == 3


# ---------------------------------------------------------------------------
# 17.5 stop / EOS logic
# ---------------------------------------------------------------------------


def test_terminal_token_ends_the_rollout_and_is_kept_but_not_content():
    def to_eos(last):
        logits = torch.full((VOCAB,), -20.0)
        logits[EOS if last != EOS else 5] = 0.0
        return logits

    rollout = sampler(model=StubModel(to_eos), max_new_tokens=10).sample('a', seed=2)
    assert rollout.finished_reason == FINISHED_EOS
    assert rollout.token_ids[-1] == EOS
    assert EOS not in rollout.content_ids
    assert len(rollout.content_ids) == len(rollout.token_ids) - 1


def test_stop_string_ends_the_rollout():
    def to_b(last):
        logits = torch.full((VOCAB,), -20.0)
        logits[ord('b') % VOCAB] = 0.0
        return logits

    rollout = sampler(model=StubModel(to_b), stop_strings=('b',),
                      max_new_tokens=10).sample('a', seed=2)
    assert rollout.finished_reason == FINISHED_STOP_STRING
    assert rollout.text.endswith('b')


def test_max_new_tokens_and_context_limit_are_reported():
    capped = sampler(max_new_tokens=5).sample('a', seed=1)
    assert capped.finished_reason == FINISHED_MAX_NEW_TOKENS
    assert capped.generated_tokens == 5
    limited = sampler(max_new_tokens=50, max_context_tokens=4).sample('abc', seed=1)
    assert limited.finished_reason == FINISHED_CONTEXT_LIMIT
    assert len(limited.prompt_ids) + limited.generated_tokens == 4


def test_terminal_token_ids_accepts_int_and_list_eos():
    assert terminal_token_ids(StubTokenizer(eos_token_id=5)) == (5,)
    assert terminal_token_ids(StubTokenizer(eos_token_id=[1, 107])) == (1, 107)
    with_turn = StubTokenizer(added={'<end_of_turn>': 106})
    assert terminal_token_ids(with_turn) == (EOS, 106)


def test_configured_stop_token_ids_override_the_derived_ones():
    engine = sampler(stop_token_ids=(7,))
    assert engine.stop_token_ids == (7,)


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_sampling_config_from_launch_config_reads_the_campaign_recipe(tmp_path):
    path = tmp_path / 'launch-config.json'
    path.write_text(json.dumps({
        'source_commit': 'deadbeef',
        'options': {'temperature': 0.6, 'top_p': 0.95, 'generate_max_len': 1024,
                    'max_len': 4096, 'micro_train_batch_size': 4},
    }))
    config = sampling_config_from_launch_config(path, stop_token_ids=(1, 107))
    assert config.temperature == 0.6
    assert config.top_p == 0.95
    assert config.max_new_tokens == 1024
    assert config.max_context_tokens == 4096
    assert config.stop_token_ids == (1, 107)


def test_launch_config_without_sampling_keys_is_rejected(tmp_path):
    path = tmp_path / 'launch-config.json'
    path.write_text(json.dumps({'options': {'temperature': 0.6}}))
    with pytest.raises(ValueError):
        sampling_config_from_launch_config(path)


@pytest.mark.parametrize('kwargs', [
    {'temperature': 0.0},
    {'temperature': -1.0},
    {'top_p': 0.0},
    {'top_p': 1.5},
    {'top_k': -1},
    {'max_new_tokens': 0},
    {'max_context_tokens': 0},
])
def test_invalid_sampling_configs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SamplingConfig(**kwargs)


def test_prompt_argument_contract():
    engine = sampler(max_new_tokens=2)
    with pytest.raises(ValueError):
        engine.sample()
    with pytest.raises(ValueError):
        engine.sample('a', prompt_ids=[1])
    with pytest.raises(ValueError):
        engine.sample(prompt_ids=[])
    with pytest.raises(ValueError):
        engine.sample_branches([1, 2], 0, base_seed=0)


# ---------------------------------------------------------------------------
# 17.6 the continuation must be atomizable by the production atomizer
# ---------------------------------------------------------------------------


def test_sampled_continuation_is_atomizable_by_the_production_atomizer():
    import os
    student_dir = os.environ.get('CA_STUDENT_DIR')
    teacher_dir = os.environ.get('CA_TEACHER_DIR')
    if not (student_dir and teacher_dir):
        pytest.skip('CA_STUDENT_DIR / CA_TEACHER_DIR are not set: no real tokenizer fixture')
    transformers = pytest.importorskip('transformers')
    from kdflow.algorithms._mp_opd_atoms import SimCTAtomizer

    student = transformers.AutoTokenizer.from_pretrained(student_dir)
    teacher = transformers.AutoTokenizer.from_pretrained(teacher_dir)

    # Drive the stub so it emits the token ids of a real sentence under the student
    # tokenizer. Then the sampled continuation is genuine text for this tokenizer and the
    # production atomizer can be run on it exactly as the trainer would.
    sentence = 'The answer is 391, because 17 times 23 equals 391.'
    sentence_ids = student.encode(sentence, add_special_tokens=False)
    assert len(sentence_ids) >= 4
    cursor = {'step': 0}

    def next_sentence_token(last):
        logits = torch.full((student.vocab_size,), -20.0)
        logits[sentence_ids[cursor['step'] % len(sentence_ids)]] = 0.0
        cursor['step'] += 1
        return logits

    engine = HFFrozenPolicySampler(
        StubModel(next_sentence_token, with_cache=False), student,
        SamplingConfig(temperature=0.6, top_p=0.95, top_k=1,
                       max_new_tokens=len(sentence_ids)),
        student_revision='stub')
    rollout = engine.sample(prompt_ids=student.encode('What is 17*23?',
                                                      add_special_tokens=False), seed=4)
    assert rollout.token_ids == tuple(sentence_ids)
    text = student.decode(list(rollout.content_ids), skip_special_tokens=True)
    stu_ids = student.encode(text, add_special_tokens=False)
    tea_ids = teacher.encode(text, add_special_tokens=False)
    assert stu_ids and tea_ids
    result = SimCTAtomizer(student, teacher).atomize(stu_ids, tea_ids, sample_id='sampler-test')
    assert result.valid, result.failure_reason
    assert sum(atom.student_token_count for atom in result.atoms) <= len(stu_ids)
