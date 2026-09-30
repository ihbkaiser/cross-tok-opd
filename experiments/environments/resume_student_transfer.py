"""Single approved 30-minute HTTP resume; immutable prior evidence, no retries.

The token arrives over stdin only. Never emit exception text or signed URLs.
The incomplete Xet prefix is untrusted until the entire file passes pinned SHA256.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

REVISION = '4e20de362430cd3b72f300e6b0f18e50e7166e08'
CHECKSUM = '68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a'
TOTAL = 2471645608


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HardTimeout(TimeoutError):
    pass


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_destination(url):
    parsed = urlsplit(url)
    assert parsed.scheme == 'https' and parsed.hostname is not None
    assert parsed.hostname.endswith(('.hf.co', '.huggingface.co'))
    assert parsed.username is None and parsed.password is None


def append_range(response, stream, offset, total, progress):
    assert response.status == 206, 'Range response required'
    assert response.headers.get('Content-Range') == f'bytes {offset}-{total-1}/{total}'
    if response.headers.get('Content-Length') is not None:
        assert int(response.headers['Content-Length']) == total-offset
    position = offset
    while position < total:
        chunk = response.read(min(1024**2, total-position))
        if not chunk:
            raise EOFError('Incomplete response; no retry')
        stream.write(chunk)
        position += len(chunk)
        progress(position)
    assert response.read(1) == b'', 'Unexpected trailing data'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    assert args.root.is_absolute() and root not in (Path('/'), Path.home())
    os.umask(0o077)
    out = root/'student-http-resume-r4'
    out.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    report = dict(status='running', transfer_verified=False, timeout_seconds=1800,
                  started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                  model='meta-llama/Llama-3.2-1B', revision=REVISION,
                  expected_sha256=CHECKSUM, expected_bytes=TOTAL,
                  reason='User approved one 30-minute Range resume after student timeout',
                  automatic_retry=False, token_persisted=False, exit_code=None)

    def save():
        temporary = out/'result.pending.json'
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(out/'result.json')

    def deadline(signum, frame):
        raise HardTimeout('Approved 1800-second limit reached')

    def interrupted(signum, frame):
        raise InterruptedError('Transfer interrupted')

    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGTERM, interrupted)
    signal.alarm(1800)
    save()
    try:
        old = json.loads((root/'model-transfer90-student-adopt-r3/result.json').read_text())
        assert old['status'] == 'timeout' and old['process_state'] is None
        assert not Path(f"/proc/{old['pid']}").exists(), 'Prior PID must be absent'
        previous = json.loads((root/'previous-models.json').read_text())['models'][0]
        assert previous['repo'] == report['model'] and previous['revision'] == REVISION
        hub = root/'hf/hub/models--meta-llama--Llama-3.2-1B'
        snapshot = hub/'snapshots'/REVISION
        assert Path(previous['snapshot']) == snapshot
        for entry in previous['files']:
            if entry['name'] != 'model.safetensors':
                path = snapshot/entry['name']
                assert path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256']
        partials = list((hub/'blobs').glob(CHECKSUM+'.*.incomplete'))
        assert len(partials) == 1 and not partials[0].is_symlink()
        source = partials[0]
        before = source.stat()
        offset = before.st_size
        assert 0 < offset < TOTAL and shutil.disk_usage(root).free > TOTAL + 3*1024**3
        destination = hub/'blobs'/CHECKSUM
        assert not destination.exists(), 'Never overwrite an existing cache blob'
        stage = root/'hf/resume-student-r4'
        stage.mkdir(mode=0o700, exist_ok=False)
        resumed = stage/'model.safetensors.incomplete'
        with source.open('rb') as original, resumed.open('xb') as stream:
            shutil.copyfileobj(original, stream, length=8*1024**2)
        after = source.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
            after.st_ino, after.st_size, after.st_mtime_ns), 'Prior partial changed'
        report.update(prefix_bytes=offset, prefix_sha256=digest(source),
                      remaining_bytes=TOTAL-offset, received_bytes=0, original_partial_preserved=True)
        save()
        print(json.dumps(report), flush=True)
        token = sys.stdin.read().strip()
        assert token.startswith('hf_')
        opener = build_opener(NoRedirect)
        public_url = f'https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/{REVISION}/model.safetensors'
        try:
            with opener.open(Request(public_url, method='HEAD',
                             headers={'Authorization': 'Bearer '+token}), timeout=120):
                raise RuntimeError('Expected explicit signed redirect')
        except HTTPError as response:
            report['access_status'] = response.code
            assert response.code == 302
            signed_url = response.headers['Location']
            response.close()
        token = None
        validate_destination(signed_url)
        request = Request(signed_url, headers={'Range': f'bytes={offset}-{TOTAL-1}'})
        last_progress = time.monotonic()

        def progress(position):
            nonlocal last_progress
            report['received_bytes'] = position-offset
            if time.monotonic()-last_progress >= 15:
                save()
                print(json.dumps(dict(received_bytes=report['received_bytes'],
                                      remaining_bytes=TOTAL-position)), flush=True)
                last_progress = time.monotonic()

        with opener.open(request, timeout=120) as response, resumed.open('ab') as stream:
            report['range_status'] = response.status
            append_range(response, stream, offset, TOTAL, progress)
            stream.flush()
            os.fsync(stream.fileno())
        assert resumed.stat().st_size == TOTAL
        report['actual_sha256'] = digest(resumed)
        assert report['actual_sha256'] == CHECKSUM, 'Full file hash mismatch; do not promote'
        # Atomic no-overwrite publication; preserve the original partial and resume evidence.
        os.link(resumed, destination)
        link = snapshot/'model.safetensors'
        if link.is_symlink():
            assert link.resolve() == destination
        else:
            assert not link.exists()
            link.symlink_to(os.path.relpath(destination, snapshot))
        verified = []
        for entry in previous['files']:
            path = snapshot/entry['name']
            assert path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256']
            verified.append(entry['name'])
        report.update(status='verified', transfer_verified=True, verified_files=verified, exit_code=0)
    except BaseException as error:
        report.update(status='timeout' if isinstance(error, HardTimeout) else 'failed',
                      error_type=type(error).__name__, exit_code=124 if isinstance(error, HardTimeout) else 1)
        if isinstance(error, HTTPError):
            report['http_error_status'] = error.code
            error.close()
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic()-started
        report['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()
    print(json.dumps(report), flush=True)
    return report['exit_code']


if __name__ == '__main__':
    sys.exit(main())
