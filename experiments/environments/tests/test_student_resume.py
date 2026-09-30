import importlib.util
import io
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('resume', Path(__file__).parents[1]/'resume_student_transfer.py')
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


class Response(io.BytesIO):
    def __init__(self, data, status=206, content_range='bytes 3-5/6'):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Range': content_range, 'Content-Length': '3'}


class ResumeContract(unittest.TestCase):
    def test_valid_range_appends_exact_suffix(self):
        target = io.BytesIO(b'abc')
        target.seek(3)
        positions = []
        resume.append_range(Response(b'def'), target, 3, 6, positions.append)
        self.assertEqual(target.getvalue(), b'abcdef')
        self.assertEqual(positions, [6])

    def test_full_response_or_wrong_range_cannot_write(self):
        for response in (Response(b'def', status=200), Response(b'def', content_range='bytes 0-2/6')):
            target = io.BytesIO()
            with self.assertRaises(AssertionError):
                resume.append_range(response, target, 3, 6, lambda n: None)
            self.assertEqual(target.getvalue(), b'')

    def test_truncated_response_is_not_retried(self):
        with self.assertRaises(EOFError):
            resume.append_range(Response(b'd'), io.BytesIO(), 3, 6, lambda n: None)

    def test_signed_destination_is_restricted_and_has_no_credentials(self):
        resume.validate_destination('https://cas-bridge.xethub.hf.co/example')
        for url in ('http://cas-bridge.xethub.hf.co/example', 'https://hf.co.attacker.example/a',
                    'https://user:password@cas-bridge.xethub.hf.co/a'):
            with self.assertRaises(AssertionError):
                resume.validate_destination(url)

    def test_deadline_no_overwrite_and_hash_gate(self):
        source = Path(resume.__file__).read_text()
        self.assertIn('signal.alarm(1800)', source)
        self.assertIn("report['actual_sha256'] == CHECKSUM", source)
        self.assertIn('os.link(resumed, destination)', source)
        self.assertNotIn('force_download', source)


if __name__ == '__main__':
    unittest.main()
