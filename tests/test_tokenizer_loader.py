import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("tokenizer_loader", Path(__file__).resolve().parents[1] / "kdflow/utils/tokenizer_loader.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SerializedTokenizerTests(unittest.TestCase):
    def test_phi_custom_split_is_recognized(self):
        self.assertTrue(module.custom_bytelevel_split({"pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split"}, {"type": "ByteLevel", "use_regex": False}]}}))

    def test_regular_bytelevel_and_missing_backend_are_not_replaced(self):
        for state in ({}, {"pre_tokenizer": {"type": "ByteLevel", "use_regex": True}},
                      {"pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split"}, {"type": "ByteLevel", "use_regex": True}]}}):
            self.assertFalse(module.custom_bytelevel_split(state))

    def test_verified_qwen_loader_is_not_replaced_for_structural_difference(self):
        state = {"pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split"}, {"type": "ByteLevel", "use_regex": False}]}}
        self.assertTrue(module.needs_serialized_fallback("GPT2Tokenizer", state, {}))
        self.assertFalse(module.needs_serialized_fallback("Qwen2Tokenizer", state, {}))
        self.assertFalse(module.needs_serialized_fallback("GPT2Tokenizer", state, state))


if __name__ == "__main__": unittest.main()
