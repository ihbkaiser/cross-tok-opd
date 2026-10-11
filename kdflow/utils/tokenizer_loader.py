"""Preserve custom serialized GPT2 pre-tokenization across Transformers versions."""
import hashlib
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def custom_bytelevel_split(state):
    """Recognize regex Split + non-regex ByteLevel (e.g. Phi-4-mini)."""
    pre = state.get("pre_tokenizer") or {}
    parts = pre.get("pretokenizers", []) if pre.get("type") == "Sequence" else []
    return (any(p.get("type") == "Split" for p in parts)
            and any(p.get("type") == "ByteLevel" and p.get("use_regex") is False for p in parts))


def needs_serialized_fallback(class_name, serialized, loaded):
    # Qwen also has a Split/ByteLevel sequence, but its verified corpus IDs
    # match serialization. Do not generalize the GPT2 constructor regression.
    return (class_name in {"GPT2Tokenizer", "GPT2TokenizerFast"}
            and custom_bytelevel_split(serialized)
            and loaded.get("pre_tokenizer") != serialized["pre_tokenizer"])


def load_tokenizer(model_name_or_path, *, use_fast=True, **kwargs):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from transformers.utils.hub import cached_file

    automatic = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast, **kwargs)
    hub_keys = ("cache_dir", "revision", "token", "local_files_only", "subfolder", "force_download")
    path = cached_file(model_name_or_path, "tokenizer.json",
                       **{k: kwargs[k] for k in hub_keys if k in kwargs},
                       _raise_exceptions_for_missing_entries=False)
    result = automatic
    policy = "auto"
    if path:
        serialized = json.loads(Path(path).read_text(encoding="utf-8"))
        backend = getattr(automatic, "backend_tokenizer", None)
        loaded = json.loads(backend.to_str()) if backend is not None else {}
        if needs_serialized_fallback(type(automatic).__name__, serialized, loaded):
            if not use_fast:
                raise ValueError("Custom serialized ByteLevel Split requires use_fast=True")
            result = PreTrainedTokenizerFast.from_pretrained(model_name_or_path, **kwargs)
            raw = Tokenizer.from_file(path)
            # Transformers may rebuild the optional BOS/EOS post-processor.
            # The serialized pipeline is the contract, including that component.
            result.backend_tokenizer.post_processor = raw.post_processor
            actual = json.loads(result.backend_tokenizer.to_str())
            for component in ("model", "normalizer", "pre_tokenizer", "post_processor", "decoder"):
                if actual.get(component) != serialized.get(component):
                    raise RuntimeError("Serialized tokenizer fallback changed " + component)
            if result.get_vocab() != automatic.get_vocab():
                raise RuntimeError("Serialized tokenizer fallback changed vocabulary IDs")
            for name in ("eos_token_id", "bos_token_id", "pad_token_id", "chat_template"):
                if getattr(result, name, None) != getattr(automatic, name, None):
                    raise RuntimeError("Serialized tokenizer fallback changed " + name)
            for text in ("(n", "Let's", "def factorial(n):\n", "\n\n", "café 日本語"):
                for add_special in (False, True):
                    if result.encode(text, add_special_tokens=add_special) != raw.encode(text, add_special_tokens=add_special).ids:
                        raise RuntimeError("Serialized tokenizer fallback encode mismatch")
            policy = "serialized_custom_bytelevel_split"
            log.warning("TOKENIZER_LOADER: replaced %s with %s to preserve serialized pre-tokenization",
                        type(automatic).__name__, type(result).__name__)
        result.kdflow_tokenizer_provenance = {
            "policy": policy, "class": type(result).__name__, "auto_class": type(automatic).__name__,
            "tokenizer_json_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
    else:
        result.kdflow_tokenizer_provenance = {"policy": policy, "class": type(result).__name__}
    log.info("TOKENIZER_LOADER: %s", json.dumps(result.kdflow_tokenizer_provenance, sort_keys=True))
    return result
