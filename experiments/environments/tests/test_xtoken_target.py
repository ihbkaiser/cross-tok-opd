"""Guard the user-approved workload and full-projection recipe."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1]/'xtoken_target.py'
spec = importlib.util.spec_from_file_location('target_harness', SCRIPT)
target_harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(target_harness)


class TargetContract(unittest.TestCase):
    def test_three_requested_dimensions_and_short_scope(self):
        values = dict(x.lstrip('+').split('=', 1) for x in target_harness.overrides(Path('/run'), '/student', '/teacher', 3))
        for role in ('policy', 'teachers.0'):
            self.assertEqual(values[role+'.train_global_batch_size'], '64')
            self.assertEqual(values[role+'.max_total_sequence_length'], '2048')
        self.assertEqual(values['distillation.max_num_steps'], '3')
        self.assertEqual(values['distillation.num_prompts_per_step'], '64')
        self.assertEqual(values['data.train.characters_per_sample'], '16384')
        self.assertEqual(values['data.max_input_seq_length'], '2048')
        self.assertFalse(any(k.startswith('loss_fn.') for k in values))

    def test_ten_updates_change_only_execution_length(self):
        previous = target_harness.overrides(Path('/run'), '/student', '/teacher', 3)
        requested = target_harness.overrides(Path('/run'), '/student', '/teacher', 10)
        changes = [(before, after) for before, after in zip(previous, requested) if before != after]
        self.assertEqual(changes, [('distillation.max_num_steps=3', 'distillation.max_num_steps=10')])
        with self.assertRaises(AssertionError):
            target_harness.overrides(Path('/run'), '/student', '/teacher', 100)

    def test_data_capacity_tracks_authorized_steps(self):
        spec = importlib.util.spec_from_file_location('prepare_target', SCRIPT.parent/'prepare_xtoken_target.py')
        preparation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preparation)
        self.assertEqual(preparation.required_packs(3), 256)
        self.assertEqual(preparation.required_packs(10), 704)
        source = (SCRIPT.parent/'validate_xtoken_target.py').read_text()
        self.assertIn('minimum_presentations = args.expected_steps * 64', source)
        self.assertIn('len(dataset) >= minimum_presentations', source)
        self.assertIn('>= minimum_presentations', source)
        self.assertIn('Run identity/protocol mismatch', SCRIPT.read_text())

    def test_full_projection_not_smoke(self):
        cmds = target_harness.projection_commands(['uv', 'python'], Path('/run'), '/student', '/teacher')
        self.assertEqual(len(cmds), 3)
        self.assertIn('--enable-reverse-pass', cmds[0])
        self.assertIn('--enable-scale-trick', cmds[0])
        self.assertIn('--enable-special-token-mapping', cmds[0])
        self.assertNotIn('--enable-exact-match', cmds[0])
        self.assertEqual(cmds[0][cmds[0].index('--top-k')+1], '32')
        self.assertIn('tools.x_token.reapply_exact_map', cmds[1])
        self.assertIn('/run/projection-full/prep32_special.pt', cmds[1])
        self.assertIn('tools.x_token.sort_and_cut_projection_matrix', cmds[2])
        self.assertEqual(cmds[2][cmds[2].index('--top_k')+1], '4')
        self.assertIn('/run/projection-full/prep32_special_exact_map_remapped.pt', cmds[2])
        self.assertIn('/run/projection-full/runtime_top4.pt', cmds[2])

    def test_train_is_single_attempt_and_has_validation_gate(self):
        source = SCRIPT.read_text()
        self.assertIn(".open('x')", source)
        self.assertIn("['config_digest'] == config_digest", source)
        self.assertIn("NCCL_CUMEM_HOST_ENABLE='0'", source)
        self.assertNotIn('P2P_DISABLE=', source)


if __name__ == '__main__':
    unittest.main()
