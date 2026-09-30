#!/usr/bin/env python3
"""Download official LCB release_v6 raw shards; stdlib only, no GPU or dataset scripts."""
import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
BASE = f"https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/{REVISION}/"
# HF LFS SHA256 and sizes, verified via HF tree API at this immutable revision.
# ALLOWED_FILES['release_v6'] in code_generation_lite.py lists all six shards.
SHARDS = [
    ("test.jsonl", 1252609773, "2bd02b38beb48e8c46b5b9987095d999ff38cd8efc255ea5d58974317c48f63f"),
    ("test2.jsonl", 713377060, "095df7c5daf15f882c51a9deb84085cff1e073495a5dbcf95015a564d485f3a3"),
    ("test3.jsonl", 623360766, "28ed26cc83363ce3f1fe2d5fad9f8393077beb1907b167a31bd3b32f80801b79"),
    ("test4.jsonl", 1204644685, "d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da"),
    ("test5.jsonl", 557699297, "7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214"),
    ("test6.jsonl", 134303240, "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"),
]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def download(opener, root, name, size, expected):
    target = root / name
    part = root / (name + ".part")
    if target.is_symlink() or part.is_symlink():
        raise ValueError("Symlink download destination refused")
    if target.exists():
        if target.stat().st_size != size or sha256(target) != expected:
            raise ValueError(f"Existing {name} does not match pin; refusing overwrite")
        print(f"REUSE_VERIFIED {name}", flush=True)
        return
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        raise ValueError(f"Oversize partial file: {part}")
    if offset < size:
        req = urllib.request.Request(BASE + name, headers={"Range": f"bytes={offset}-"} if offset else {})
        with opener.open(req, timeout=180) as response:
            resumed = response.status == 206
            if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Range response offset mismatch")
            if response.status not in (200, 206):
                raise ValueError(f"Unexpected HTTP status {response.status}")
            count = offset if resumed else 0
            report_at = count + 64 * 1024**2
            with part.open("ab" if resumed else "wb") as stream:
                while chunk := response.read(4 * 1024**2):
                    count += len(chunk)
                    if count > size:
                        raise ValueError("Download exceeds pinned size")
                    stream.write(chunk)
                    if count >= report_at:
                        print(f"DOWNLOAD {name} {count}/{size}", flush=True)
                        report_at = count + 64 * 1024**2
    if part.stat().st_size != size or sha256(part) != expected:
        raise ValueError(f"Incomplete or corrupt {name}; partial kept for inspection")
    part.rename(target)
    print(f"VERIFIED {name}", flush=True)


def validate_row(row):
    required = ("question_id", "question_content", "metadata", "starter_code", "public_test_cases", "private_test_cases")
    if any(key not in row for key in required):
        raise ValueError("Missing canonical LCB fields")
    if not str(row["question_id"]) or not row["question_content"]:
        raise ValueError("Empty ID/prompt")
    metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
    public = json.loads(row["public_test_cases"]) if isinstance(row["public_test_cases"], str) else row["public_test_cases"]
    if not isinstance(metadata, dict) or not isinstance(public, list) or not row["private_test_cases"]:
        raise ValueError("Invalid metadata/public/private tests")
    types = {case["testtype"] for case in public}
    if types - {"stdin", "functional"}:
        raise ValueError("Unknown test type")
    if "functional" in types and not metadata.get("func_name"):
        raise ValueError("Functional task missing func_name")
    return str(row["question_id"]), bool(metadata.get("func_name"))


def merge(root):
    output = root / "problems.jsonl"
    receipt = root / "manifest.json"
    if output.exists() or receipt.exists():
        if not output.is_file() or not receipt.is_file():
            raise ValueError("Incomplete previous export; inspect before retry")
        manifest = json.loads(receipt.read_text())
        if manifest["revision"] != REVISION or manifest["sha256"] != sha256(output):
            raise ValueError("Existing merged export mismatch")
        return manifest
    partial = root / "problems.jsonl.part"
    if partial.is_symlink():
        raise ValueError("Symlink output refused")
    ids, functional, per_file = set(), 0, {}
    h = hashlib.sha256()
    with partial.open("wb") as dst:
        for name, _, _ in SHARDS:
            count = 0
            with (root / name).open("rb") as src:
                for line in src:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    identifier, is_functional = validate_row(row)
                    if identifier in ids:
                        raise ValueError(f"Duplicate question_id {identifier}")
                    ids.add(identifier)
                    functional += is_functional
                    count += 1
                    # Preserve every original field and its encoding, including starter_code and private tests.
                    line = line.rstrip(b"\r\n") + b"\n"
                    dst.write(line)
                    h.update(line)
            per_file[name] = count
            print(f"VALIDATED_ROWS {name} {count}", flush=True)
    if len(ids) != 1055:
        raise ValueError(f"Expected release_v6 1055 tasks; got {len(ids)}")
    manifest = {"repo": "livecodebench/code_generation_lite", "revision": REVISION,
                "release": "release_v6", "split": "test", "count": len(ids),
                "functional_tasks": functional, "file": str(output), "sha256": h.hexdigest(),
                "source_files": [{"name": n, "bytes": size, "sha256": sha, "rows": per_file[n]} for n, size, sha in SHARDS],
                "validation": "source SHA256, IDs, count, required fields, public types and func_name; private payload preserved without execution/deserialization"}
    partial.rename(output)
    with receipt.open("x") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--proxy", required=True, help="Company-approved HTTP proxy")
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Keep one copy of raw shards plus one merged export. No data is deleted automatically.
    remaining = sum(max(0, size - ((root/name).stat().st_size if (root/name).is_file() else
                        (root/(name+'.part')).stat().st_size if (root/(name+'.part')).is_file() else 0)) for name,size,_ in SHARDS)
    if not (root/'problems.jsonl').exists():
        remaining += sum(size for _,size,_ in SHARDS)
    if shutil.disk_usage(root).free < remaining + 512 * 1024**2:
        raise RuntimeError(f"Need approximately {remaining/1024**3:.2f} GiB plus 0.5 GiB reserve")
    lock = root / '.download.lock'
    with lock.open('x'):
        pass
    try:
        # Explicit proxy only, normal TLS verification; no auth or policy bypass.
        os.environ.pop('no_proxy', None)
        os.environ.pop('NO_PROXY', None)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http':args.proxy,'https':args.proxy}))
        for shard in SHARDS:
            download(opener, root, *shard)
        manifest = merge(root)
        print('LCB_V6_READY ' + json.dumps(manifest), flush=True)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
