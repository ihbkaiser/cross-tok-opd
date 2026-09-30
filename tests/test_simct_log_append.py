import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments/modal"))
from append_simct_runai_logs import verify_prefix, canonical_readback

class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{"optimizer_step":i,"loss":-.125*i} for i in range(1,4)]
    def test_matching_prefix(self):
        self.assertEqual(verify_prefix(self.rows,self.rows[:2]),2)
    def test_changed_metric_rejected(self):
        altered=[{**self.rows[0],"loss":0}]
        with self.assertRaises(ValueError):verify_prefix(self.rows,altered)
    def test_duplicate_and_gap_rejected(self):
        for history in ([self.rows[0],self.rows[0]],[self.rows[0],self.rows[2]]):
            with self.assertRaises(ValueError):verify_prefix(self.rows,history)
    def test_readback_exact_duplicates_only(self):
        self.assertEqual(canonical_readback([self.rows[0],self.rows[0],self.rows[1]]),self.rows[:2])
        with self.assertRaises(ValueError):
            canonical_readback([self.rows[0],{**self.rows[0],"loss":1}])
    def test_missing_metric_rejected(self):
        with self.assertRaises(ValueError):verify_prefix(self.rows,[{"optimizer_step":1}])

if __name__=="__main__":unittest.main()
