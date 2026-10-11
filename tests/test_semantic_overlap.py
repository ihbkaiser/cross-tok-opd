import importlib.util
from pathlib import Path
import random
import unittest

spec = importlib.util.spec_from_file_location("semantic_overlap", Path(__file__).resolve().parents[1] / "kdflow/algorithms/_semantic_overlap.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Tok:
    all_special_ids = [99]
    eos_token_id = 99

    def __init__(self, vocab, texts):
        self.vocab, self.texts = vocab, texts

    def get_vocab(self):
        items = list(self.vocab.items())
        random.shuffle(items)
        return dict(items)

    def decode(self, ids, **_):
        return "".join(self.texts[i] for i in ids)


class SemanticOverlapTests(unittest.TestCase):
    def test_collision_and_control_characters_use_emission_not_spelling(self):
        student = Tok({"▁": 1, "Ġ": 2, "Ċ": 3, "newline": 4, "partial": 5, "eos": 99},
                      {1: " ", 2: "Ġ", 3: "Ċ", 4: "\n", 5: "\ufffd", 99: "<s-eos>"})
        teacher = Tok({"Ġ": 10, "Ċ": 11, "eos": 99}, {10: " ", 11: "\n", 99: "<t-eos>"})
        for _ in range(30):
            si, ti, record = module.semantic_overlap(student, teacher)
            self.assertEqual(list(zip(si, ti)), [(4, 11), (1, 10), (99, 99)])
            self.assertEqual(record["content_pairs"], 2)
        self.assertNotIn(2, si)
        self.assertNotIn(3, si)
        self.assertNotIn(5, si)

    def test_duplicate_emission_has_stable_representative_and_digest(self):
        s = Tok({"a": 7, "b": 1, "eos": 99}, {7: "x", 1: "x", 99: "EOS"})
        t = Tok({"x": 2, "eos": 99}, {2: "x", 99: "EOS"})
        results = [module.semantic_overlap(s, t) for _ in range(10)]
        self.assertTrue(all(v == results[0] for v in results))
        self.assertEqual(results[0][0], [1, 99])
        self.assertEqual(results[0][2]["student"]["duplicate_emissions"], 1)

    def test_context_dependent_token_is_excluded(self):
        class ContextTok(Tok):
            def encode(self, text, **_): return [1]
            def decode(self, ids, **_): return " " if ids == [2] else ("x" if ids == [1, 2] else "x" * len(ids))
        s = ContextTok({"x": 1, "space": 2, "eos": 99}, {})
        values, counts = module.emitted_vocab(s)
        self.assertNotIn(b" ", values)
        self.assertEqual(counts["excluded_nonlossless_or_contextual"], 1)


if __name__ == "__main__": unittest.main()
