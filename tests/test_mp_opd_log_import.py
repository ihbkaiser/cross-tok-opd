"""Regression checks for historical MP-OPD log parsing."""
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/"experiments/modal/import_mp_opd_runai_logs.py"
spec=importlib.util.spec_from_file_location("mp_import",path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ParserTests(unittest.TestCase):
    def test_progress_times_are_not_metrics(self):
        line=("[2026-09-08 13:01:30] [INFO] [on_policy_kd_trainer.py:logging:747] "
              "[Rank 0] epoch [1/2], step [1/312], Elapsed: 0:01:30, ETA: 5:20:10, "
              "content_length_mean: 42.000000, loss: -0.125000, lr: 1.250000e-07, "
              "resource/gpu_0/utilization_percent: 7.000000")
        row=module.parse_metrics(line)[0]
        self.assertEqual(set(row),{"optimizer_step","content_length_mean","loss","lr","resource/gpu_0/utilization_percent"})
        self.assertEqual(row["loss"],-.125)
        self.assertEqual(row["lr"],1.25e-7)
    def test_nonfinite_metrics_fail(self):
        with self.assertRaises(ValueError):
            module.parse_metrics("on_policy_kd_trainer.py:logging:747 step [1/312], content_length_mean: 42, loss: nan")
    def test_bypass_not_a_training_step(self):
        self.assertEqual(module.parse_metrics("MP_PARITY_BYPASS mean=0.1 max=25.0 tokens=100 above_0p5=1"),[])
    def test_config_is_parsed_without_execution(self):
        import ast
        cfg=module.decode_config(ast.parse("AllArguments(train=TrainingArguments(eval_steps=inf, num_epochs=2))",mode="eval").body)
        self.assertEqual(cfg,{"train":{"eval_steps":"inf","num_epochs":2}})
        with self.assertRaises((ValueError,TypeError)):
            module.decode_config(ast.parse("__import__('os').system('false')",mode="eval").body.func)

if __name__=="__main__":
    unittest.main()
