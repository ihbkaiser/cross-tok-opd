import ast
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_cpu_logits_overlay as overlay
import cpu_logits_attempt as attempt


class CpuOverlayTests(unittest.TestCase):
    def test_unique_anchors_fail_closed(self):
        for relative in overlay.PINS:
            with self.assertRaises(AssertionError): overlay.transform(relative, 'pass\n')

    def test_patch_only_transport_anchors(self):
        src = 'def allocate():\n' + overlay.ALLOC_OLD + '        True\n    )\n' + \
              'def rebuild():\n' + overlay.REBUILD_OLD + '    return func\n'
        out = overlay.transform('nemo_rl/models/policy/utils.py', src)
        restored = out.replace(overlay.ALLOC_NEW, overlay.ALLOC_OLD).replace(overlay.REBUILD_NEW, overlay.REBUILD_OLD)
        self.assertEqual(src, restored)
        self.assertIn('ensure_cpu_buffer', out)
        ast.parse(out)

    def test_cache_eviction_precedes_model_offload(self):
        src = 'class Worker:\n' + overlay.CACHE_OLD
        out = overlay.transform('nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py', src)
        self.assertEqual(out.replace(overlay.CACHE_NEW, overlay.CACHE_OLD), src)
        self.assertLess(out.index('torch.cuda.empty_cache()'), out.index('self.model = self.move_to_cpu'))

    def test_slice_before_gpu_upload(self):
        src = 'def view():\n' + overlay.VIEW_OLD
        out = overlay.transform('nemo_rl/algorithms/x_token/loss_utils.py', src)
        self.assertEqual(out.replace(overlay.VIEW_NEW, overlay.VIEW_OLD), src)
        self.assertIn('return transport_view(view, device)', out)

    def test_no_precision_or_loss_modification(self):
        self.assertEqual(len(overlay.PINS), 3)
        self.assertNotIn('bfloat16', overlay.ALLOC_NEW + overlay.REBUILD_NEW + overlay.VIEW_NEW)

    def test_config_equivalence_allows_only_run_identity(self):
        old = dict(batch=64, sequence=2048, log_dir='/runtime/'+attempt.OLD+'/logs')
        new = dict(batch=64, sequence=2048, log_dir='/runtime/'+attempt.RUN+'/logs')
        self.assertTrue(attempt.equivalent_config(old, new))
        new['batch'] = 32
        self.assertFalse(attempt.equivalent_config(old, new))

    def test_retry_requires_diagnosed_terminal_oom(self):
        record = dict(status='stopped', exit_code=1, training_phase_invoked=True,
            phases=[dict(phase=p,exit_code=code) for p,code in
                    [('check',1),('canary',0),('data',0),('projection',0),('config',0),('train',1)]])
        attempt.validate_previous(record, 'Tried to allocate 37.09 GiB ensure_teacher_ipc_buffer')
        with self.assertRaises(AssertionError): attempt.validate_previous(record, 'Other error')
        record['status'] = 'running_phase'
        with self.assertRaises(AssertionError):
            attempt.validate_previous(record, 'Tried to allocate 37.09 GiB ensure_teacher_ipc_buffer')


if __name__ == '__main__': unittest.main()
