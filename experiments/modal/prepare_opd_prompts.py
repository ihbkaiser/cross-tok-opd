"""Create a deterministic short-prompt OPD view from the pinned source parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=640)
    parser.add_argument("--max-tokens", type=int, default=240)
    args = parser.parse_args()

    if sha256(args.source) != args.source_sha256:
        raise RuntimeError("source data SHA mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        args.student,
        local_files_only=True,
        use_fast=True,
    )
    dataset = load_dataset("parquet", data_files=str(args.source), split="train")
    if len(dataset) < args.rows:
        raise RuntimeError(f"source dataset contains only {len(dataset)} rows")

    args.destination.parent.mkdir(parents=True, exist_ok=False)
    lengths: list[int] = []
    raw_prefix_lengths: list[int] = []
    with args.destination.open("x", encoding="utf-8") as stream:
        for index in range(args.rows):
            raw = dataset[index]["text"]
            token_ids = tokenizer.encode(raw, add_special_tokens=False)[: args.max_tokens]
            while token_ids:
                text = tokenizer.decode(token_ids, skip_special_tokens=False)
                processed_length = len(tokenizer.encode(text, add_special_tokens=True))
                if processed_length <= args.max_tokens:
                    break
                token_ids.pop()
            else:
                raise RuntimeError(f"empty prompt after length fitting at row {index}")
            if not text.strip():
                raise RuntimeError(f"empty prompt after transform at row {index}")
            lengths.append(processed_length)
            raw_prefix_lengths.append(len(token_ids))
            stream.write(json.dumps({"text": text, "source_row": index}) + "\n")

    manifest = {
        "rows": args.rows,
        "source_sha256": args.source_sha256,
        "output_sha256": sha256(args.destination),
        "min_prompt_tokens": min(lengths),
        "max_prompt_tokens": max(lengths),
        "min_raw_prefix_tokens": min(raw_prefix_lengths),
        "max_raw_prefix_tokens": max(raw_prefix_lengths),
        "transform": (
            f"first {args.rows} rows; longest student-tokenizer prefix whose "
            f"final encoded prompt including special tokens is <= {args.max_tokens} tokens"
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    pending = args.manifest.with_suffix(args.manifest.suffix + ".pending")
    pending.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    pending.replace(args.manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
