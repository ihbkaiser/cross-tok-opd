import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.error import HTTPError

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import resumable_target_data as data
from continue_target_data import validate_previous


def raw_page(offset=0):
    return json.dumps(dict(rows=[dict(row_idx=i, row=dict(text='text'), truncated_cells=[])
                                 for i in range(offset, offset + 100)])).encode()


class Response:
    headers = {'X-Revision': data.REVISION}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return raw_page()


class DataRecoveryTests(unittest.TestCase):
    def test_transient_retry_visible_and_success(self):
        calls, events, sleeps = [], [], []
        def opener(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise HTTPError('redacted', 502, 'Bad Gateway', {}, None)
            return Response()
        self.assertEqual(data.fetch(0, 1000, events.append, opener, sleeps.append, lambda: 0), raw_page())
        self.assertEqual(len(calls), 2)
        self.assertEqual(events[0]['http_status'], 502)
        self.assertEqual(events[-1]['status'], 'verified')
        self.assertEqual(sleeps, [2])

    def test_retry_limit_and_no_retry_auth(self):
        for status, expected in [(503, 8), (403, 1), (404, 1)]:
            events = []
            def opener(*args, **kwargs):
                raise HTTPError('redacted', status, 'error', {}, None)
            with self.assertRaises(HTTPError):
                data.fetch(0, 1000, events.append, opener, lambda _: None, lambda: 0)
            self.assertEqual(len(events), expected)

    def test_retry_after_cannot_exceed_deadline(self):
        def opener(*args, **kwargs):
            raise HTTPError('redacted', 429, 'error', {'Retry-After': '600'}, None)
        with self.assertRaises(TimeoutError):
            data.fetch(0, 100, lambda _: None, opener, lambda _: self.fail('sleep'), lambda: 0)

    def test_integrity_errors_not_retried(self):
        class Wrong(Response): headers = {'X-Revision': 'different'}
        events = []
        with self.assertRaises(AssertionError):
            data.fetch(0, 100, events.append, lambda *a, **k: Wrong(), lambda _: self.fail('sleep'), lambda: 0)
        self.assertEqual(events, [])
        with self.assertRaises(AssertionError): data.validate_page(raw_page(100), data.REVISION, 0)

    def test_cache_provenance_and_corruption(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            raw = raw_page()
            self.assertIsNone(data.cached_page(out, 0))
            data.atomic(out/'viewer-0000.json', raw)
            self.assertIsNone(data.cached_page(out, 0))  # legacy/uncommitted entry
            data.json_atomic(out/'viewer-0000.meta.json', dict(dataset=data.DATASET, subset=data.SUBSET,
                split='train', revision=data.REVISION, offset=0, sha256=data.digest(raw)))
            self.assertEqual(data.cached_page(out, 0), raw)
            (out/'viewer-0000.json').write_bytes(raw + b' ')
            with self.assertRaises(AssertionError): data.cached_page(out, 0)

    def test_cannot_continue_an_active_or_training_predecessor(self):
        record = dict(status='stopped', exit_code=1, optimizer_updates=10, training_phase_invoked=False,
            phases=[dict(phase=p, exit_code=c, expected_metadata_exclusions_only=p == 'check')
                    for p, c in zip(['check','canary','models','data'], [1,0,0,1])])
        validate_previous(record)
        record['training_phase_invoked'] = True
        with self.assertRaises(AssertionError): validate_previous(record)


if __name__ == '__main__': unittest.main()
