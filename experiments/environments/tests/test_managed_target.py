import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1]/'managed_target.py'
spec = importlib.util.spec_from_file_location('managed_target', SCRIPT)
managed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(managed)


class ManagedTarget(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def simulate(self, install_exit, mirror=False):
        calls = []
        def execute(script, arguments, log):
            calls.append((script.name, arguments))
            if script.name == 'run_phase.py':
                evidence = self.root/'evidence/install'
                evidence.mkdir(parents=True)
                (evidence/'result.json').write_text(json.dumps({'exit_code': install_exit}))
                log.write_text(f'EVIDENCE_DIR={evidence}\n')
                return install_exit
            self.assertEqual(script.name, 'run_target_after_setup.py')
            sequence = self.root/'sequence-target-4b-2048-b64-full-10steps-r3'
            sequence.mkdir()
            (sequence/'result.json').write_text(json.dumps(dict(
                status='training_process_complete_pending_metric_review', training_phase_invoked=True)))
            log.write_text('fixture\n')
            return 0
        predecessor = self.root/'sequence-target-4b-2048-b64-full-10steps-r2'
        paths = [self.root/name for name in ('old/result.json', 'teacher/result.json', 'student/result.json')]
        with patch.object(managed, 'validate_recovery', return_value=(predecessor, paths)), \
             patch.object(managed, 'validate_mirror_recovery', return_value=(predecessor, paths)), \
             patch.object(managed, 'resource_gate'), patch.object(managed, 'execute', side_effect=execute):
            rc = managed.run(self.root, mirror=mirror)
            with self.assertRaises(FileExistsError):
                managed.run(self.root, mirror=mirror)
        name = managed.MIRROR_NAME if mirror else managed.NAME
        return rc, calls, json.loads((self.root/name/'result.json').read_text())

    def test_mirror_install_has_new_claim_and_explicit_transport_flag(self):
        rc, calls, result = self.simulate(0, mirror=True)
        self.assertEqual(rc, 0)
        self.assertIn('--pypi-mirror', calls[0][1])
        self.assertIn('3', calls[1][1])  # Sequence r3 has never run; managed r3 timed out in setup.
        self.assertFalse((self.root/managed.NAME).exists())
        self.assertIn('TUNA', result['reason'])

    def test_mirror_install_failure_never_runs_sequence_or_retries(self):
        rc, calls, result = self.simulate(1, mirror=True)
        self.assertEqual(rc, 1)
        self.assertEqual(len(calls), 1)
        self.assertFalse(result['training_phase_invoked'])

    def test_mirror_predecessor_must_be_timeout_with_no_sequence_or_training(self):
        previous = self.root/managed.NAME
        previous.mkdir()
        evidence = self.root/'evidence/previous'
        evidence.mkdir(parents=True)
        setup = evidence/'result.json'
        setup.write_text(json.dumps(dict(exit_code=124, timed_out=True, console_disconnected=False)))
        record = dict(status='setup_failed', exit_code=124, training_phase_invoked=False,
                      requested_optimizer_updates=10, setup_result=str(setup))
        (previous/'source-fingerprints.json').write_text(json.dumps({
            str(managed.HERE/p): hashlib.sha256((managed.HERE/p).read_bytes()).hexdigest()
            for p in managed.SCIENTIFIC}))
        with patch.object(managed, 'validate_recovery', return_value=('old', [])):
            for change in (dict(training_phase_invoked=True), dict(status='installing'),
                           dict(exit_code=0), dict(requested_optimizer_updates=3)):
                (previous/'result.json').write_text(json.dumps(record | change))
                with self.assertRaises(AssertionError):
                    managed.validate_mirror_recovery(self.root)
            (previous/'result.json').write_text(json.dumps(record))
            self.assertEqual(managed.validate_mirror_recovery(self.root), ('old', []))
            (self.root/'sequence-target-4b-2048-b64-full-10steps-r3').mkdir()
            with self.assertRaises(AssertionError):
                managed.validate_mirror_recovery(self.root)

    def test_one_successful_install_then_one_sequence(self):
        rc, calls, result = self.simulate(0)
        self.assertEqual(rc, 0)
        self.assertEqual([name for name, _ in calls], ['run_phase.py', 'run_target_after_setup.py'])
        self.assertIn('5400', calls[0][1])
        self.assertIn('300', calls[0][1])
        self.assertIn('10', calls[1][1])
        self.assertIn('3', calls[1][1])
        self.assertTrue(result['training_phase_invoked'])
        self.assertFalse(result['automatic_retry'])

    def test_failed_install_never_retries_or_launches_training(self):
        rc, calls, result = self.simulate(124)
        self.assertEqual(rc, 124)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['status'], 'setup_failed')
        self.assertFalse(result['training_phase_invoked'])

    def test_preflight_failure_preserves_terminal_result_without_workload(self):
        with patch.object(managed, 'validate_recovery', side_effect=AssertionError()), \
             patch.object(managed, 'execute') as execute:
            self.assertEqual(managed.run(self.root), 1)
            execute.assert_not_called()
        result = json.loads((self.root/managed.NAME/'result.json').read_text())
        self.assertEqual(result['status'], 'stopped')


if __name__ == '__main__':
    unittest.main()
