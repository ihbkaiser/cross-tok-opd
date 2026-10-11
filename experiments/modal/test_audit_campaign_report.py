import unittest
from audit_campaign_report import BENCHES,audit,macro_by_seed

class ReportStatisticsTests(unittest.TestCase):
    def test_covariance_cancels_macro_variation(self):
        # Each task varies across samples; the paired aggregate is constant.
        cell={b:{'seeds':{'42':v[0],'43':v[1]}} for b,v in zip(BENCHES,[(0,1),(1,0),(0,1),(1,0)])}
        self.assertEqual(macro_by_seed(cell),{'42':.5,'43':.5})
        cell['macro']={'mean':.5,'eval_std':.7071067811865476}
        rows,issues=audit({'groups':{'x':[{'id':'MP-x-s44','train_seed':'42','steps':{'312':cell}}]}})
        self.assertEqual(rows[0]['macro_std_recomputed'],0)
        self.assertEqual({i['kind'] for i in issues},{'training_seed_conflict','macro_std_mismatch'})

    def test_missing_benchmark_has_no_macro(self):
        self.assertEqual(macro_by_seed({BENCHES[0]:{'seeds':{'42':1,'43':0}}}),{})

    def test_benchmark_names_are_not_replicates(self):
        cell={b:{'seeds':{'math500':1,'mbpp':0}} for b in BENCHES}
        self.assertEqual(macro_by_seed(cell),{})

if __name__=='__main__':unittest.main()
