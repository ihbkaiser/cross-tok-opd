import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1]/'run_target_after_setup.py'
spec = importlib.util.spec_from_file_location('target_sequence', SCRIPT)
sequence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sequence)


class TargetSequence(unittest.TestCase):
    def test_tee_retains_durable_log_when_console_disconnects(self):
        class Gone:
            def write(self, text):
                raise BrokenPipeError()
            def flush(self):
                raise BrokenPipeError()
        durable = io.StringIO()
        tee = sequence.Tee(Gone(), durable)
        self.assertEqual(tee.write('first\n'), 6)
        tee.write('second\n')
        tee.flush()
        self.assertEqual(durable.getvalue(), 'first\nsecond\n')
        self.assertTrue(tee.stream.disconnected)

    def test_continuation_requires_zero_predecessor_phases(self):
        good = dict(status='stopped', exit_code=1, optimizer_updates=10,
                    training_phase_invoked=False, phases=[])
        sequence.validate_setup_only_predecessor(good)
        for changes in (dict(training_phase_invoked=True), dict(phases=['train']),
                        dict(optimizer_updates=3), dict(status='running')):
            with self.assertRaises(AssertionError):
                sequence.validate_setup_only_predecessor(good | changes)

    def test_only_terminal_success_allows_next_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp)/'result.json'
            self.assertEqual(sequence.dependency_result(result), 'waiting')
            for code, expected in [(None, 'waiting'), (0, 'ready'), (1, 'failed'), (124, 'failed')]:
                result.write_text(json.dumps(dict(exit_code=code)))
                self.assertEqual(sequence.dependency_result(result), expected)
            result.write_text('{')
            self.assertEqual(sequence.dependency_result(result), 'waiting')

    def test_student_adoption_requires_explicit_integrity_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp)/'result.json'
            for status, verified, expected in [('running', False, 'waiting'),
                    ('verified', True, 'ready'), ('verified', False, 'failed'),
                    ('timeout', False, 'failed'), ('failed_incomplete', False, 'failed')]:
                result.write_text(json.dumps(dict(status=status, transfer_verified=verified)))
                self.assertEqual(sequence.dependency_result(result, student=True), expected)

    def test_only_exact_five_native_exclusions_are_accepted(self):
        log = 'Found 5 incompatibilities\n' + '\n'.join(
            f"The package `nemo-rl` requires `{name}>=1`, but it's not installed"
            for name in sorted(sequence.EXPECTED_MISSING))
        self.assertTrue(sequence.expected_metadata_only(log))
        self.assertFalse(sequence.expected_metadata_only(log.replace('tilelang', 'torch')))
        self.assertFalse(sequence.expected_metadata_only(log + '\nThe package `bad` has an incompatible version'))
        self.assertFalse(sequence.expected_metadata_only(log.replace('Found 5', 'Found 6')))

    def test_run_is_claimed_and_training_never_retries(self):
        text = SCRIPT.read_text()
        self.assertIn('out.mkdir(exist_ok=False)', text)
        self.assertIn("'check', 'canary', 'models', 'data', 'projection', 'config', 'train'", text)
        self.assertIn("'failed' in statuses", text)
        self.assertIn('training_process_complete_pending_metric_review', text)
        self.assertIn("teacher = config.teachers[0].policy_config()",
                      (SCRIPT.parent/'validate_xtoken_target.py').read_text())


if __name__ == '__main__':
    unittest.main()
