import unittest

from kdflow.trajectory import collapse_observation, trajectory_tokens


class TrajectoryTests(unittest.TestCase):
    def test_bos_boundary_and_noncanonical_output_path(self):
        # Prompt BOS and final token must remain in conditioning. The sampled
        # two-token path must not be replaced by a canonical one-token encoding.
        ids, mask, synthetic = trajectory_tokens([128000, 17, 18], [21, 22], 128001)
        self.assertEqual(ids, [128000, 17, 18, 21, 22, 128001])
        shifted = ids[1:] + ids[:1]
        self.assertEqual([x for x, keep in zip(shifted, mask) if keep], [21, 22, 128001])
        self.assertTrue(synthetic)

    def test_real_eos_is_not_duplicated(self):
        self.assertEqual(trajectory_tokens([9], [4, 2], 2), ([9, 4, 2], [True, True, False], False))

    def test_empty_output_has_only_masked_sentinel(self):
        self.assertEqual(trajectory_tokens([9], [], 2), ([9, 2], [True, False], True))

    def test_collapse_requires_two_batches_and_recovers(self):
        first = collapse_observation([100, 120], None, 0)
        second = collapse_observation([0, 0], first["content_length_baseline"], 0)
        self.assertFalse(second["collapse_stop"])
        self.assertTrue(collapse_observation([0, 0], 110, 1)["collapse_stop"])
        self.assertEqual(collapse_observation([100, 120], 110, 1)["collapse_bad_streak"], 0)

    def test_zero_first_batch_is_not_a_valid_baseline(self):
        self.assertTrue(collapse_observation([0, 0], 0, 1)["collapse_stop"])


if __name__ == "__main__":
    unittest.main()


def test_bounded_sampling_preserves_other_options_and_total_sequence_cap():
    from kdflow.trajectory import bounded_sampling_params
    config={'max_new_tokens':4096,'temperature':.6,'top_p':.95}
    result=bounded_sampling_params([[1]*100,[1]*1600],config,4096)
    assert [x['max_new_tokens'] for x in result]==[3995,2495]
    assert all(x['temperature']==.6 and x['top_p']==.95 for x in result)
    assert config['max_new_tokens']==4096
