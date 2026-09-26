"""Independent edge cases for ties, direction and per-query aggregation."""
import unittest
from metrics_core import per_query,summarize


class RankingMetrics(unittest.TestCase):
    def test_perfect_and_reverse(self):
        good=per_query([9,7,4,1]);bad=per_query([1,4,7,9])
        self.assertAlmostEqual(good['spearman'],1)
        self.assertAlmostEqual(bad['spearman'],-1)
        self.assertEqual(good['kendall_tau_b'],1)
        self.assertEqual(bad['kendall_tau_b'],-1)
        self.assertEqual(good['ndcg4_among_executed_candidates'],1)
        self.assertEqual(good['pairwise_concordance'],1)
        self.assertEqual(bad['pairwise_concordance'],0)

    def test_ties_are_not_dropped(self):
        r=per_query([5,5,5,5])
        self.assertIsNone(r['spearman']);self.assertIsNone(r['kendall_tau_b'])
        self.assertEqual(r['pairwise_concordance'],.5)
        self.assertEqual(r['top_bottom_tie'],1)
        self.assertEqual(r['ndcg4_among_executed_candidates'],1)
        self.assertIsNone(per_query([0,0,0,0])['ndcg4_among_executed_candidates'])

    def test_all_undefined_group_remains_accounted_for(self):
        s=summarize([{'metrics':per_query([0,0,0,0])}],repetitions=100)
        self.assertEqual(s['spearman']['undefined_all_tied_queries'],1)
        self.assertEqual(s['spearman']['mean_with_undefined_set_to_zero'],0)
        self.assertIsNone(s['spearman']['mean'])
        self.assertEqual(s['pairwise_concordance']['mean'],.5)

    def test_tau_b_partial_ties(self):
        from scipy.stats import kendalltau,spearmanr
        r=per_query([8,8,3,1])
        self.assertAlmostEqual(r['kendall_tau_b'],kendalltau([-1,-5,-10,-15],[8,8,3,1]).statistic)
        self.assertAlmostEqual(r['spearman'],spearmanr([-1,-5,-10,-15],[8,8,3,1]).statistic)
        self.assertEqual(r['pairwise_concordance'],5.5/6)

    def test_macro_denominator_and_bootstrap(self):
        s=summarize([{'metrics':per_query([4,3,2,1])},{'metrics':per_query([5,5,5,5])}],repetitions=100)
        self.assertEqual(s['n_queries'],2);self.assertEqual(s['n_executions'],8)
        self.assertEqual(s['spearman']['defined_queries'],1)
        self.assertIsNone(s['spearman']['ci95'])
        self.assertEqual(s['spearman']['mean_with_undefined_set_to_zero'],.5)
        self.assertEqual(s['pairwise_concordance']['mean'],.75)

    def test_single_query_has_no_uncertainty_estimate(self):
        s=summarize([{'metrics':per_query([4,3,2,1])}],repetitions=100)
        self.assertEqual(s['spearman']['mean'],1)
        self.assertIsNone(s['spearman']['ci95'])
        self.assertIsNone(s['pairwise_concordance']['ci95'])


if __name__=='__main__':unittest.main()
