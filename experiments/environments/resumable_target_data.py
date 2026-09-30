"""Exact-revision Viewer prefix with bounded, visible transport retries.

Only cache entries carrying a verified revision and matching bytes are reusable.
Old pages without a sidecar remain untouched; they are not silently trusted.
The packing semantics and deterministic prefix match prepare_xtoken_target.data.
"""
import argparse
import datetime as dt
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

DATASET = 'nvidia/Nemotron-Pretraining-Specialized-v1.1'
SUBSET = 'Nemotron-Pretraining-Formal-Logic'
REVISION = '13fa979be2e7f7e62913eee0ec5e97c8fd6e24af'
TRANSIENT = {429, 500, 502, 503, 504}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def atomic(path, raw):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_bytes(raw)
    pending.replace(path)


def json_atomic(path, value):
    atomic(path, (json.dumps(value, indent=2) + '\n').encode())


def validate_page(raw, revision, offset):
    assert revision == REVISION, 'Viewer revision changed'
    page = json.loads(raw)['rows']
    assert [r['row_idx'] for r in page] == list(range(offset, offset + 100)), 'Noncontiguous page'
    assert all(not r.get('truncated_cells') for r in page), 'Truncated cells'
    assert all(isinstance(r['row']['text'], str) and r['row']['text'] for r in page)
    return page


def fetch(offset, deadline, emit, opener=urlopen, sleep=time.sleep, clock=time.monotonic):
    query = urlencode(dict(dataset=DATASET, config=SUBSET, split='train', offset=offset, length=100))
    for attempt in range(1, 9):
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError('Data phase deadline reached')
        delay = min(2 ** attempt, 60)
        try:
            with opener('https://datasets-server.huggingface.co/rows?' + query,
                        timeout=min(60, remaining)) as response:
                raw, revision = response.read(), response.headers.get('X-Revision')
            validate_page(raw, revision, offset)  # Integrity errors are never retried.
            emit(dict(offset=offset, attempt=attempt, status='verified', sha256=digest(raw)))
            return raw
        except HTTPError as error:
            retryable = error.code in TRANSIENT
            retry_after = error.headers.get('Retry-After') if error.headers else None
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    try:
                        delay = max(delay, (parsedate_to_datetime(retry_after) -
                                           dt.datetime.now(dt.timezone.utc)).total_seconds())
                    except (ValueError, TypeError):
                        pass
            emit(dict(offset=offset, attempt=attempt, status='http_error',
                      http_status=error.code, retryable=retryable, delay_seconds=delay))
            if not retryable or attempt == 8:
                raise
        except (URLError, TimeoutError, socket.timeout) as error:
            emit(dict(offset=offset, attempt=attempt, status='transport_error',
                      error_type=type(error).__name__, delay_seconds=delay))
            if attempt == 8:
                raise
        if delay >= deadline - clock():
            raise TimeoutError('Retry would exceed data phase deadline')
        sleep(delay)
    raise AssertionError('Unreachable')


def cached_page(out, offset):
    path, meta = out/f'viewer-{offset:04d}.json', out/f'viewer-{offset:04d}.meta.json'
    if not path.exists() and not meta.exists():
        return None
    # A crash between page and sidecar writes is explicitly recovered by refetching.
    if path.exists() and not meta.exists():
        return None
    record, raw = json.loads(meta.read_text()), path.read_bytes()
    assert record == dict(dataset=DATASET, subset=SUBSET, split='train',
                          revision=REVISION, offset=offset, sha256=digest(raw))
    validate_page(raw, record['revision'], offset)
    return raw


def prepare(target, seconds=1500):
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = target/'data'
    out.mkdir(exist_ok=True)
    assert not (out/'manifest.json').exists(), 'Complete data must not be replayed'
    deadline = time.monotonic() + seconds

    def emit(record):
        record['at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        with (out/'transport-attempts.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)

    rows, pages, packed, buffered = [], [], 0, 0
    minimum_packs = 704
    max_rows = ((minimum_packs * 16 + 99) // 100) * 100
    for offset in range(0, max_rows, 100):
        raw = cached_page(out, offset)
        if raw is None:
            raw = fetch(offset, deadline, emit)
            atomic(out/f'viewer-{offset:04d}.json', raw)
            json_atomic(out/f'viewer-{offset:04d}.meta.json',
                        dict(dataset=DATASET, subset=SUBSET, split='train',
                             revision=REVISION, offset=offset, sha256=digest(raw)))
        else:
            emit(dict(offset=offset, status='verified_cache_reuse', sha256=digest(raw)))
        for item in validate_page(raw, REVISION, offset):
            text = item['row']['text']
            rows.append(dict(text=text))
            buffered += len(text)
            if buffered >= 16384:
                packed += 1
                buffered = 0
        pages.append(dict(offset=offset, rows=100, sha256=digest(raw)))
        emit(dict(raw_rows=len(rows), complete_packs=packed))
        if packed >= minimum_packs:
            break
    assert packed >= minimum_packs, 'Insufficient prefix; no repetition or batch reduction'
    parquet = out/'formal-logic-prefix.parquet'
    pending = out/'formal-logic-prefix.parquet.pending'
    pq.write_table(pa.Table.from_pylist(rows), pending)
    pending.replace(parquet)
    json_atomic(out/'manifest.json', dict(dataset=DATASET, subset=SUBSET, revision=REVISION,
        raw_rows=len(rows), complete_packs=packed, characters_per_sample=16384,
        optimizer_updates=10, minimum_presentations=640, requested_complete_packs=704,
        parquet_sha256=digest(parquet.read_bytes()), pages=pages,
        fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='bounded workload test prefix, not full training corpus',
        transport_policy='8 attempts/page, bounded phase deadline; exact-revision cache'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, type=Path)
    args = parser.parse_args()
    assert args.target.is_absolute()
    prepare(args.target)
