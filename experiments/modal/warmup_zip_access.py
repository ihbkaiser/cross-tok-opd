"""Inspect/extract selected files from a pinned public ZIP via checked HTTP ranges.

Never downloads an entire model archive merely to inspect its central directory.
Extraction is intended for remote persistent storage, not the local laptop.
"""
import binascii
import hashlib
import io
import struct
import time
import zlib
from pathlib import Path

import requests

REPO = 'ihbkaiser/qwen25vl-3b-msd-checkpoints'
REVISION = 'd77f01ffaebec859a65041a52dea7a5d11d79312'
ARCHIVE = 'sft-warmup-both-20261009.zip'
ARCHIVE_BYTES = 38656399804
ARCHIVE_SHA256 = '9f9af1b623d3be4efde8d251b75e9b8c6f052e6e4aa4818a8c9555927adc394d'
URL = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{ARCHIVE}'

def checked_range(start, end):
    response = requests.get(URL, params={'range_nonce': f'{start}-{end}-{time.time_ns()}'},
                            headers={'Range': f'bytes={start}-{end}'},
                            stream=True, timeout=(30, 180))
    expected = f'bytes {start}-{end}/{ARCHIVE_BYTES}'
    if response.status_code != 206 or response.headers.get('Content-Range') != expected:
        status = response.status_code
        content_range = response.headers.get('Content-Range')
        response.close()
        raise RuntimeError(f'Range request refused: status={status}, range={content_range}; expected={expected}')
    return response

class RemoteZipReader(io.RawIOBase):
    def __init__(self):
        self.position = 0
        self.transferred = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self.position = offset
        elif whence == io.SEEK_CUR: self.position += offset
        elif whence == io.SEEK_END: self.position = ARCHIVE_BYTES + offset
        else: raise ValueError(whence)
        if self.position < 0: raise ValueError('Negative seek')
        return self.position

    def read(self, size=-1):
        if size < 0: size = ARCHIVE_BYTES - self.position
        size = min(size, ARCHIVE_BYTES - self.position)
        if size <= 0: return b''
        if size > 4 * 1024 * 1024:
            raise RuntimeError('Metadata reader refuses large reads; use remote streaming extraction')
        with checked_range(self.position, self.position + size - 1) as response:
            data = response.content
        assert len(data) == size
        self.transferred += len(data)
        self.position += len(data)
        return data

def extract_member(info, root):
    root = Path(root).resolve()
    target = (root / info.filename).resolve()
    if not target.is_relative_to(root) or info.is_dir():
        raise ValueError('Unsafe or directory member')
    with checked_range(info.header_offset, info.header_offset + 29) as response:
        header = response.content
    values = struct.unpack('<4s5H3I2H', header)
    assert values[0] == b'PK\x03\x04'
    start = info.header_offset + 30 + values[-2] + values[-1]
    if info.compress_type not in (0, 8): raise ValueError('Unsupported ZIP compression')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.range-partial')
    decompressor = zlib.decompressobj(-15) if info.compress_type == 8 else None
    digest = hashlib.sha256(); crc = 0; written = 0; compressed_read = 0
    def consume(raw, output):
        nonlocal crc, written
        written += len(raw)
        if written > info.file_size: raise RuntimeError('ZIP size overflow')
        digest.update(raw); crc = binascii.crc32(raw, crc); output.write(raw)
    with temporary.open('wb') as output:
        if info.compress_size:
            with checked_range(start, start + info.compress_size - 1) as response:
                for chunk in response.iter_content(4 * 1024 * 1024):
                    compressed_read += len(chunk)
                    consume(decompressor.decompress(chunk) if decompressor else chunk, output)
        if decompressor: consume(decompressor.flush(), output)
    assert compressed_read == info.compress_size and written == info.file_size
    assert (crc & 0xffffffff) == info.CRC
    if decompressor: assert decompressor.eof
    temporary.replace(target)
    return {'member': info.filename, 'bytes': written, 'compressed_bytes': compressed_read,
            'sha256': digest.hexdigest(), 'zip_crc_pass': True, 'path': str(target)}
