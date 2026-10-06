"""Standalone frozen-policy sampler for the cross-atom branch probe (Phase 1).

Why this exists: the campaign's rollout path runs through SGLang 0.5.11, which returns
the unfilled ``0.0`` sentinel for a fraction of sampled tokens whenever
``temperature > 0`` (see ``SGLANG_ZERO_LOGPROB_BUG.md``). Phase 1 is an offline
diagnostic, not a training run, so it samples the frozen old student directly in
PyTorch instead of going through that engine. The brief forbids relaxing the engine's
guard and forbids falling back to greedy decoding, so neither is done here.

Contract this sampler keeps, because Phase 1's numbers depend on it:

* **Log-probability semantics match the engine's**, not the sampler's internals. SGLang
  reports ``log_softmax(logits / T)[sampled_token]`` computed on the *unfiltered*
  temperature-scaled distribution (that is exactly what the trainer's parity guard
  assumes when it divides the trainer logits by the rollout temperature). So the
  recorded ``logprob`` is that quantity, while the *draw* comes from the top-p/top-k
  filtered distribution. Recording the filtered probability instead would silently
  change the meaning of every ``b_i`` computed downstream.
* **Every recorded log-probability is finite by construction.** It is read out of the
  same ``log_softmax`` the draw came from, so there is no unfilled slot to misread —
  the failure mode of the engine path cannot occur here.
* **Determinism.** The draw uses an explicit CPU ``torch.Generator`` seeded per call,
  so a fixed seed reproduces the exact token sequence on any device. Sampling on CPU
  costs one small transfer per token and buys device-independent reproducibility.
* **No trainer, no optimizer, no parameter mutation.** The model is only ever called
  under ``torch.inference_mode()``.

Terminal handling mirrors the production contract: the sampled stop token is *kept* in
``token_ids`` (so the caller re-atomizes exactly what the trainer would see) and also
exposed separately as ``content_ids`` with terminal controls stripped, which is the
same rule ``mp_content_ids`` applies inside the atomizer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

FINISHED_EOS = 'eos_token'
FINISHED_STOP_STRING = 'stop_string'
FINISHED_MAX_NEW_TOKENS = 'max_new_tokens'
FINISHED_CONTEXT_LIMIT = 'context_limit'

# Chat templates for the pairs in this campaign end a turn with one of these; they are
# terminal controls, not response content, and must be stripped like EOS.
DEFAULT_TERMINAL_TOKENS = ('<end_of_turn>', '<|im_end|>', '<|endoftext|>', '<end_of_turn>')


def terminal_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """EOS plus the template's end-of-turn control, as integer ids.

    ``tokenizer.eos_token_id`` is an int for some tokenizers and a list for others
    (Gemma-2 ships ``[1, 107]``), so both shapes are accepted.
    """
    ids: list[int] = []
    eos = getattr(tokenizer, 'eos_token_id', None)
    if isinstance(eos, (list, tuple)):
        ids.extend(int(value) for value in eos)
    elif eos is not None:
        ids.append(int(eos))
    added = getattr(tokenizer, 'get_added_vocab', lambda: {})()
    for name in DEFAULT_TERMINAL_TOKENS:
        value = added.get(name)
        if value is not None:
            ids.append(int(value))
    seen: list[int] = []
    for value in ids:
        if value not in seen:
            seen.append(value)
    return tuple(seen)


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling knobs, defaulted to the campaign's recorded recipe."""

    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 0
    max_new_tokens: int = 1024
    stop_strings: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    max_context_tokens: int = 4096

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('temperature must be positive and finite')
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError('top_p must be in (0, 1]')
        if self.top_k < 0:
            raise ValueError('top_k must be non-negative (0 disables it)')
        if self.max_new_tokens < 1:
            raise ValueError('max_new_tokens must be at least 1')
        if self.max_context_tokens < 1:
            raise ValueError('max_context_tokens must be at least 1')

    def as_dict(self) -> dict[str, Any]:
        return {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'max_new_tokens': self.max_new_tokens,
            'stop_strings': list(self.stop_strings),
            'stop_token_ids': list(self.stop_token_ids),
            'max_context_tokens': self.max_context_tokens,
        }


def sampling_config_from_launch_config(path: str | Path, *, stop_token_ids: Sequence[int] = (),
                                       stop_strings: Sequence[str] = ()) -> SamplingConfig:
    """Read the sampling knobs out of a campaign ``launch-config.json``.

    Config parity is a claim Phase 1 has to be able to make, so the values come from the
    campaign's own record instead of being re-typed: ``temperature``, ``top_p``,
    ``generate_max_len`` and ``max_len``.
    """
    payload = json.loads(Path(path).read_text())
    options = payload.get('options', payload)
    missing = [key for key in ('temperature', 'top_p', 'generate_max_len', 'max_len')
               if key not in options]
    if missing:
        raise ValueError('launch config is missing ' + ','.join(missing))
    return SamplingConfig(
        temperature=float(options['temperature']),
        top_p=float(options['top_p']),
        max_new_tokens=int(options['generate_max_len']),
        max_context_tokens=int(options['max_len']),
        stop_token_ids=tuple(int(value) for value in stop_token_ids),
        stop_strings=tuple(str(value) for value in stop_strings),
    )


@dataclass(frozen=True)
class FrozenRollout:
    """One sampled continuation of the frozen policy, with its provenance."""

    prompt_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    content_ids: tuple[int, ...]
    text: str
    finished_reason: str
    config: SamplingConfig
    seed: int
    student_revision: str = ''
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.logprobs):
            raise ValueError('token_ids and logprobs must be the same length')

    @property
    def generated_tokens(self) -> int:
        return len(self.token_ids)


def filter_logits(logits: torch.Tensor, config: SamplingConfig) -> torch.Tensor:
    """Apply top-k then top-p to *temperature-scaled* logits, in place of a copy.

    The nucleus keeps the smallest set of tokens whose cumulative probability reaches
    ``top_p`` and always keeps at least one token, which is the same rule the engine's
    top-p kernel implements.
    """
    filtered = logits.clone()
    if config.top_k > 0 and config.top_k < filtered.numel():
        threshold = torch.topk(filtered, config.top_k).values[-1]
        filtered = torch.where(filtered < threshold,
                               torch.full_like(filtered, float('-inf')), filtered)
    if config.top_p < 1.0:
        ordered, order = torch.sort(filtered, descending=True)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        drop = cumulative > config.top_p
        # Keep the token that crosses the threshold, drop everything after it.
        drop[1:] = drop[:-1].clone()
        drop[0] = False
        filtered[order[drop]] = float('-inf')
    return filtered


class HFFrozenPolicySampler:
    """Sample continuations from a frozen student with finite, engine-shaped logprobs."""

    def __init__(self, model: Any, tokenizer: Any, config: SamplingConfig,
                 *, student_revision: str = '', device: str | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.student_revision = student_revision
        self.device = device or getattr(model, 'device', None) or 'cpu'
        self._stop_ids = tuple(config.stop_token_ids) or terminal_token_ids(tokenizer)

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        return self._stop_ids

    def _decode(self, ids: Sequence[int]) -> str:
        try:
            return self.tokenizer.decode(list(ids), skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
        except TypeError:
            return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def _encode(self, text: str) -> list[int]:
        try:
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(self.tokenizer(text)['input_ids'])

    def sample(self, prompt: str | None = None, *, prompt_ids: Sequence[int] | None = None,
               seed: int = 0) -> FrozenRollout:
        if (prompt is None) == (prompt_ids is None):
            raise ValueError('pass exactly one of prompt or prompt_ids')
        prompt_tokens = self._encode(prompt) if prompt_ids is None else [int(v) for v in prompt_ids]
        if not prompt_tokens:
            raise ValueError('prompt must encode to at least one token')
        if len(prompt_tokens) >= self.config.max_context_tokens:
            raise ValueError('prompt already fills the context window')

        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        ids = list(prompt_tokens)
        generated: list[int] = []
        logprobs: list[float] = []
        finished_reason = FINISHED_MAX_NEW_TOKENS
        cache: Any = None
        cache_supported: bool | None = None

        with torch.inference_mode():
            for _ in range(self.config.max_new_tokens):
                if len(ids) >= self.config.max_context_tokens:
                    finished_reason = FINISHED_CONTEXT_LIMIT
                    break
                if cache_supported is None or not cache_supported:
                    feed = torch.tensor([ids], device=self.device)
                else:
                    feed = torch.tensor([[ids[-1]]], device=self.device)
                if cache_supported is None:
                    output = self.model(feed, use_cache=True)
                    cache = getattr(output, 'past_key_values', None)
                    cache_supported = cache is not None
                elif cache_supported:
                    output = self.model(feed, past_key_values=cache, use_cache=True)
                    cache = getattr(output, 'past_key_values', None)
                else:
                    output = self.model(feed)
                logits = output.logits[0, -1].float()
                if not torch.isfinite(logits).all():
                    raise RuntimeError('the frozen policy produced non-finite logits')

                scaled = logits / self.config.temperature
                # Engine contract: the reported logprob is the temperature-scaled,
                # *unfiltered* log-softmax of the token that was actually drawn.
                report_logprob = torch.log_softmax(scaled, dim=-1)
                draw_logits = filter_logits(scaled, self.config)
                draw_probs = torch.softmax(draw_logits, dim=-1)
                if not torch.isfinite(draw_probs).all() or float(draw_probs.sum()) <= 0.0:
                    raise RuntimeError('sampling distribution is not a valid distribution')
                draw_probs = draw_probs / draw_probs.sum()
                token = int(torch.multinomial(draw_probs.cpu(), 1, generator=generator).item())
                value = float(report_logprob[token].item())
                if not math.isfinite(value):
                    raise RuntimeError('sampled token has a non-finite log-probability')

                ids.append(token)
                generated.append(token)
                logprobs.append(value)

                if token in self._stop_ids:
                    finished_reason = FINISHED_EOS
                    break
                if self.config.stop_strings:
                    text = self._decode(generated)
                    if any(marker in text for marker in self.config.stop_strings):
                        finished_reason = FINISHED_STOP_STRING
                        break

        content = [token for token in generated if token not in self._stop_ids]
        return FrozenRollout(
            prompt_ids=tuple(prompt_tokens),
            token_ids=tuple(generated),
            logprobs=tuple(logprobs),
            content_ids=tuple(content),
            text=self._decode(content),
            finished_reason=finished_reason,
            config=self.config,
            seed=int(seed),
            student_revision=self.student_revision,
            extra={'stop_token_ids': list(self._stop_ids),
                   'terminated_by_terminal_token': finished_reason == FINISHED_EOS},
        )

    def sample_branches(self, prompt_ids: Sequence[int], count: int, *,
                        base_seed: int) -> list[FrozenRollout]:
        """``count`` independent draws from the same prefix, one RNG stream each.

        Independence is structural: every branch gets its own seed derived from
        ``base_seed`` and its own generator, so branch ``m`` never consumes randomness
        that belongs to branch ``m + 1``.
        """
        if count < 1:
            raise ValueError('count must be at least 1')
        return [self.sample(prompt_ids=list(prompt_ids), seed=base_seed + index)
                for index in range(count)]
