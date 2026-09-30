"""Lossless SimCT atom construction for MP-OPD.

This module deliberately does not reuse SpanCTKD's character-length alignment:
MP-OPD needs byte-identical, response-only atomic units and must fail closed
when that contract cannot be established.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional, Sequence


class AtomizationFailure(str, Enum):
    EMPTY_RESPONSE = "empty_response"
    EMPTY_DECODE = "empty_decode"
    REPLACEMENT_CHARACTER = "replacement_character"
    NORMALIZATION_MISMATCH = "normalization_mismatch"
    UNSUPPORTED_ADDED_TOKEN = "unsupported_added_token"
    UNALIGNED_SUFFIX = "unaligned_suffix"


@dataclass(frozen=True)
class MPAtom:
    sample_id: str
    student_start: int
    student_end: int
    teacher_start: int
    teacher_end: int
    byte_start: int
    byte_end: int
    student_token_count: int
    teacher_token_count: int
    valid: bool = True
    failure_reason: Optional[str] = None
    boundary_type: str = "one_to_one"
    lT: Optional[float] = None
    lS_old: Optional[float] = None
    b: Optional[float] = None
    w: Optional[int] = None
    r: Optional[float] = None

    def with_credit(self, *, lT: float, lS_old: float, b: float, r: float) -> "MPAtom":
        return replace(self, lT=lT, lS_old=lS_old, b=b, w=self.student_token_count, r=r)


@dataclass(frozen=True)
class AtomizationResult:
    atoms: tuple[MPAtom, ...]
    valid: bool
    failure_reason: Optional[str]
    covered_student_events: int
    covered_teacher_events: int
    masked_student_eos: int = 0
    masked_teacher_eos: int = 0


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(
            list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    except TypeError:
        return tokenizer.decode(list(ids), skip_special_tokens=False)


def _added_ids(tokenizer: Any) -> set[int]:
    getter = getattr(tokenizer, "get_added_vocab", None)
    if getter is None:
        return set()
    special = getattr(tokenizer, "all_special_ids", None)
    if special is not None:
        return {int(value) for value in special}
    return {int(value) for value in getter().values()}


def mp_content_ids(ids, tokenizer):
    """Strip terminal controls from a copy; preserve sampled IDs and logprobs."""
    terminal = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        terminal.add(int(eos))
    added = getattr(tokenizer, "get_added_vocab", lambda: {})()
    eot = added.get("<end_of_turn>")
    if eot is not None and int(eot) in set(getattr(tokenizer, "all_special_ids", ())):
        terminal.add(int(eot))
    content = [int(x) for x in ids]
    while content and content[-1] in terminal:
        content.pop()
    return content, len(ids) - len(content)


class SimCTAtomizer:
    """Construct minimal synchronized response segments with exact byte checks.

    EOS is masked by default. It is not decoded into response text because two
    tokenizers need not share a lossless representation of the stopping event.
    Padding must already have been removed by the caller's loss mask.
    """

    def __init__(self, student_tokenizer: Any, teacher_tokenizer: Any, eos_policy: str = "mask"):
        if eos_policy != "mask":
            raise ValueError("MP-OPD v0 supports only eos_policy='mask'")
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.eos_policy = eos_policy

    @staticmethod
    def _mask_terminal_eos(ids: list[int], eos_id: Optional[int]) -> tuple[list[int], int]:
        count = 0
        if eos_id is not None:
            while ids and ids[-1] == int(eos_id):
                ids.pop()
                count += 1
        return ids, count

    def atomize(
        self,
        student_label_ids: Sequence[int],
        teacher_label_ids: Sequence[int],
        *,
        sample_id: str,
    ) -> AtomizationResult:
        stu_ids, stu_eos = mp_content_ids(student_label_ids, self.student_tokenizer)
        tea_ids, tea_eos = mp_content_ids(teacher_label_ids, self.teacher_tokenizer)
        if not stu_ids or not tea_ids:
            return AtomizationResult((), False, AtomizationFailure.EMPTY_RESPONSE.value, 0, 0, stu_eos, tea_eos)

        stu_added = _added_ids(self.student_tokenizer)
        tea_added = _added_ids(self.teacher_tokenizer)
        if any(x in stu_added for x in stu_ids) or any(x in tea_added for x in tea_ids):
            return AtomizationResult(
                (), False, AtomizationFailure.UNSUPPORTED_ADDED_TOKEN.value, 0, 0, stu_eos, tea_eos
            )

        stu_prefix = [_decode(self.student_tokenizer, stu_ids[:i]) for i in range(1, len(stu_ids) + 1)]
        tea_prefix = [_decode(self.teacher_tokenizer, tea_ids[:i]) for i in range(1, len(tea_ids) + 1)]
        # Byte-BPE tokenizers can emit U+FFFD for an *intermediate* prefix that
        # ends inside a multi-byte UTF-8 scalar, then recover losslessly after
        # the next token. Such a prefix is not a legal synchronized boundary,
        # but it must not invalidate the complete response. Only the complete
        # decode is required to be replacement-free; transient prefixes are
        # filtered from the boundary maps below.
        if "\ufffd" in stu_prefix[-1] or "\ufffd" in tea_prefix[-1]:
            return AtomizationResult(
                (), False, AtomizationFailure.REPLACEMENT_CHARACTER.value, 0, 0, stu_eos, tea_eos
            )
        if not stu_prefix[-1] or not tea_prefix[-1]:
            return AtomizationResult((), False, AtomizationFailure.EMPTY_DECODE.value, 0, 0, stu_eos, tea_eos)
        if stu_prefix[-1].encode("utf-8") != tea_prefix[-1].encode("utf-8"):
            return AtomizationResult(
                (), False, AtomizationFailure.NORMALIZATION_MISMATCH.value, 0, 0, stu_eos, tea_eos
            )

        full_bytes = stu_prefix[-1].encode("utf-8")

        def boundary_map(prefixes: Sequence[str]) -> dict[int, int]:
            boundaries: dict[int, int] = {0: 0}
            for token_end, value in enumerate(prefixes, start=1):
                if not value or "\ufffd" in value:
                    continue
                candidate = value.encode("utf-8")
                if full_bytes.startswith(candidate):
                    # Keep the latest token index for a byte offset so any
                    # zero-width decoder artifacts remain inside an atom.
                    boundaries[len(candidate)] = token_end
            return boundaries

        stu_boundaries = boundary_map(stu_prefix)
        tea_boundaries = boundary_map(tea_prefix)
        common_offsets = sorted(set(stu_boundaries) & set(tea_boundaries))
        if not common_offsets or common_offsets[-1] != len(full_bytes):
            return AtomizationResult(
                (), False, AtomizationFailure.UNALIGNED_SUFFIX.value, 0, 0, stu_eos, tea_eos
            )

        atoms: list[MPAtom] = []
        for byte_start, byte_end in zip(common_offsets, common_offsets[1:]):
            prev_si, si = stu_boundaries[byte_start], stu_boundaries[byte_end]
            prev_ti, ti = tea_boundaries[byte_start], tea_boundaries[byte_end]
            stu_count = si - prev_si
            tea_count = ti - prev_ti
            if byte_end <= byte_start or stu_count <= 0 or tea_count <= 0:
                return AtomizationResult(
                    (), False, AtomizationFailure.EMPTY_DECODE.value, 0, 0, stu_eos, tea_eos
                )
            atoms.append(
                MPAtom(
                    sample_id=str(sample_id),
                    student_start=prev_si,
                    student_end=si,
                    teacher_start=prev_ti,
                    teacher_end=ti,
                    byte_start=byte_start,
                    byte_end=byte_end,
                    student_token_count=stu_count,
                    teacher_token_count=tea_count,
                    boundary_type="one_to_one" if stu_count == tea_count == 1 else "multi_token",
                )
            )
        return AtomizationResult(
            tuple(atoms), True, None, len(stu_ids), len(tea_ids), stu_eos, tea_eos
        )
