"""Deterministic SimCT vocabulary coordinates with equal emitted UTF-8 bytes.

Partial Unicode tokens and context-dependent decoder emissions are excluded.
This is a conservative shared vocabulary, not a full distribution projection.
"""
import hashlib
import json


def _decode(tok, ids):
    try:
        return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return tok.decode(ids)


def emitted_vocab(tok):
    special = set(getattr(tok, "all_special_ids", ()))
    eos = getattr(tok, "eos_token_id", None)
    if eos is not None:
        special.add(eos)
    anchor = None
    encoder = getattr(tok, "encode", None)
    if callable(encoder):
        ids = encoder("x", add_special_tokens=False)
        if ids:
            anchor = ids
    prefix = _decode(tok, anchor) if anchor else None
    buckets = {}
    excluded = 0
    for tid in sorted(set(tok.get_vocab().values())):
        if tid in special:
            continue
        text = _decode(tok, [tid])
        if not text or "\ufffd" in text:
            excluded += 1
            continue
        if anchor and _decode(tok, anchor + [tid]) != prefix + text:
            excluded += 1
            continue
        buckets.setdefault(text.encode("utf-8"), []).append(tid)
    # Prefer the canonical single-token encode if present; otherwise lowest ID.
    selected = {}
    for key, ids in buckets.items():
        canonical = encoder(key.decode("utf-8"), add_special_tokens=False) if callable(encoder) and len(ids) > 1 else []
        selected[key] = canonical[0] if len(canonical) == 1 and canonical[0] in ids else ids[0]
    return selected, {"excluded_nonlossless_or_contextual": excluded,
                      "duplicate_emissions": sum(len(v) - 1 for v in buckets.values())}


def semantic_overlap(student, teacher):
    sv, sa = emitted_vocab(student)
    tv, ta = emitted_vocab(teacher)
    common = sorted(sv.keys() & tv.keys())
    pairs = [(sv[key], tv[key]) for key in common]
    se, te = student.eos_token_id, teacher.eos_token_id
    if se is None or te is None:
        raise ValueError("SimCT requires explicit student and teacher EOS IDs")
    pairs.append((se, te))  # EOS is a role mapping, never a literal text match.
    record = {"version": "utf8-emission-v1", "content_pairs": len(common),
              "eos_pair": [se, te], "student": sa, "teacher": ta}
    encoded = json.dumps([(k.hex(), *p) for k, p in zip(common, pairs)] + [("EOS", se, te)],
                         separators=(",", ":")).encode()
    record["sha256"] = hashlib.sha256(encoded).hexdigest()
    return [p[0] for p in pairs], [p[1] for p in pairs], record
