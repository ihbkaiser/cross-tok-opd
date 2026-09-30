import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("contract", HERE / "xtoken_phase_a_contract.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


class ModalPhaseAContractTests(unittest.TestCase):
    def test_exact_scientific_shape(self):
        value = contract.scientific_contract()
        self.assertEqual(value["optimizer_updates"], 10)
        self.assertEqual(value["global_batch"], 64)
        self.assertEqual(value["micro_batch"], 1)
        self.assertEqual(value["sequence_length"], 2048)
        self.assertEqual(value["precision"], "bfloat16")
        self.assertEqual(value["seed"], 42)
        self.assertFalse(value["checkpointing"])
        self.assertFalse(value["wandb"])
        self.assertIn("off-policy", value["scope"])
        self.assertIn("not OPD", value["scope"])

    def test_exact_lineage(self):
        self.assertEqual(
            contract.STUDENT_REVISION,
            "4e20de362430cd3b72f300e6b0f18e50e7166e08",
        )
        self.assertEqual(
            contract.TEACHER_REVISION,
            "1cfa9a7208912126459214e8b04321603b3df60c",
        )
        self.assertEqual(
            contract.NEMO_REVISION,
            "13a10647ebbf0f940d2b06ea41800b3f2fb46099",
        )
        self.assertEqual(len(contract.STUDENT_WEIGHT_SHA256), 64)
        self.assertEqual(len(contract.NATIVE_LOCK_SHA256), 64)

    def test_modal_resource_and_retry_gate(self):
        value = contract.operational_contract()
        self.assertEqual(value["profile"], "no1ceboy")
        self.assertEqual(value["gpu"], "A10:2")
        self.assertEqual(value["gpu_count"], 2)
        self.assertGreaterEqual(value["container_memory_mib"], 120 * 1024)
        self.assertFalse(value["automatic_retry"])
        self.assertIn("exact FP32", value["teacher_logit_transport"])

    def test_unique_remote_identities(self):
        for value in (
            contract.APP_NAME,
            contract.VOLUME_NAME,
            contract.RUN_ID,
            contract.TARGET_NAME,
            contract.OVERLAY_NAME,
        ):
            self.assertIn("r14", value)


if __name__ == "__main__":
    unittest.main()
