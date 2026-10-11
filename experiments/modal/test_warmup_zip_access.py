"""Offline regression coverage for bounded HTTP range extraction."""
import binascii
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import warmup_zip_access as subject


class Response:
    def __init__(self, data, status=206, content_range=None):
        self.content = data
        self.status_code = status
        self.headers = {'Content-Range': content_range}
        self.closed = False

    def close(self): self.closed = True
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def iter_content(self, size):
        for offset in range(0, len(self.content), 7):
            yield self.content[offset:offset + 7]


class RangeTests(unittest.TestCase):
    def test_refuses_full_download_before_consuming_body(self):
        response = Response(b'not downloaded', status=200)
        with patch.object(subject.requests, 'get', return_value=response):
            with self.assertRaises(RuntimeError): subject.checked_range(0, 29)
        self.assertTrue(response.closed)

    def test_refuses_wrong_content_range(self):
        response = Response(b'', content_range='bytes 10-39/123')
        with patch.object(subject.requests, 'get', return_value=response):
            with self.assertRaises(RuntimeError): subject.checked_range(0, 29)

    def test_metadata_read_limit_and_64_bit_seek(self):
        reader = subject.RemoteZipReader()
        self.assertEqual(reader.seek(-22, io.SEEK_END), subject.ARCHIVE_BYTES - 22)
        reader.seek(0)
        with self.assertRaises(RuntimeError): reader.read(4 * 1024 * 1024 + 1)
        with self.assertRaises(ValueError): reader.seek(-1)

    def extract_fixture(self, compression, corrupt_crc=False):
        buffer = io.BytesIO()
        raw = b'checkpoint fixture\x00' * 50
        with zipfile.ZipFile(buffer, 'w', compression=compression) as archive:
            archive.writestr('checkpoint/model.fixture', raw)
        data = buffer.getvalue()
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            info = archive.infolist()[0]
        if corrupt_crc: info.CRC ^= 1
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(subject, 'checked_range', side_effect=lambda start, end: Response(data[start:end + 1])):
                if corrupt_crc:
                    with self.assertRaises(AssertionError): subject.extract_member(info, folder)
                    self.assertFalse((Path(folder) / info.filename).exists())
                else:
                    receipt = subject.extract_member(info, folder)
                    self.assertEqual((Path(folder) / info.filename).read_bytes(), raw)
                    self.assertEqual(receipt['bytes'], len(raw))
                    self.assertTrue(receipt['zip_crc_pass'])

    def test_stored_stream(self): self.extract_fixture(zipfile.ZIP_STORED)
    def test_deflate_stream(self): self.extract_fixture(zipfile.ZIP_DEFLATED)
    def test_crc_failure_never_publishes_target(self): self.extract_fixture(zipfile.ZIP_STORED, True)

    def test_path_traversal_rejected_before_network(self):
        info = zipfile.ZipInfo('../outside.fixture')
        with tempfile.TemporaryDirectory() as folder, patch.object(subject, 'checked_range') as request:
            with self.assertRaises(ValueError): subject.extract_member(info, folder)
            request.assert_not_called()


if __name__ == '__main__': unittest.main()
