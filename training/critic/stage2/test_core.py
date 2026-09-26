import unittest
from core import blend_scores,build_pairs,build_toolsets,check_splits,make_case,metrics,parse_target,ranked_metrics,recall_guard,signature

def c(ts,llm='<LLM_A>'):return {'llm':llm,'tools':ts}
G=[c(['a','b'])]

class CoreTests(unittest.TestCase):
    def test_blend_endpoints_and_constant_scores(self):
        self.assertGreater(blend_scores([3,1],[1,2],0)[0],blend_scores([3,1],[1,2],0)[1])
        self.assertLess(blend_scores([3,1],[1,2],1)[0],blend_scores([3,1],[1,2],1)[1])
        self.assertEqual(blend_scores([1,1],[2,2],.5),[0.,0.])
    def test_redundancy_keeps_every_reference_hit(self):
        cs=[c(['a','b']),c(['a','b','x']),c(['a']),c(['a','x'])]
        pairs,_=build_pairs(cs,G,{},42)
        self.assertIn((0,1,'redundancy'),[(p['positive'],p['negative'],p['kind']) for p in pairs])
        self.assertIn((0,2,'missing_required'),[(p['positive'],p['negative'],p['kind']) for p in pairs])
        for p in pairs:
            if p['kind']=='redundancy':
                a,b=cs[p['positive']],cs[p['negative']]
                self.assertEqual(metrics(a,G)['cr'],metrics(b,G)['cr'])
                self.assertGreater(metrics(a,G)['cp'],metrics(b,G)['cp'])

    def test_extra_in_alternate_reference_is_not_redundancy(self):
        gs=[c(['a']),c(['a','b'])]
        pairs,_=build_pairs([c(['a']),c(['a','b'])],gs,{},42)
        self.assertFalse(any(p['kind']=='redundancy' for p in pairs))

    def test_longer_necessary_bundle_can_win(self):
        pairs,_=build_pairs([c(['a']),c(['a','b'])],G,{},42)
        self.assertEqual(pairs,[{'positive':1,'negative':0,'kind':'missing_required','margin':.1}])

    def test_teacher_replay_only_changes_llm(self):
        cs=[c(['a']),c(['a'],'<LLM_B>'),c(['a','x'],'<LLM_B>')]
        teacher={signature(x):s for x,s in zip(cs,[.2,.9,.95])}
        pairs,_=build_pairs(cs,G,teacher,42)
        ps=[p for p in pairs if p['kind']=='teacher_llm']
        self.assertEqual([(p['positive'],p['negative']) for p in ps],[(1,0)])

    def test_missing_rank_is_zero_and_full_denominator(self):
        m=ranked_metrics([c(['a','b'])],[1.],G)
        self.assertGreater(m['RDCP@10'],.2);self.assertLess(m['RDCP@10'],.23)
        self.assertEqual(m['Tool-Hit@10'],1.)

    def test_no_recall_slack(self):
        b={'RDCR@10':.5,'Tool-Hit@10':.6,'CR-Hit@10':.4,'CompR@1':.5,'RDCP@10':.4}
        a={**b,'RDCP@10':.9,'RDCR@10':.499}
        self.assertFalse(recall_guard(a,b));self.assertTrue(recall_guard(b,b))

    def test_split_query_family_and_id(self):
        with self.assertRaises(ValueError):check_splits([{'qid':'1','query':'a  b'}],[{'qid':'2','query':'a b'}],[])
        with self.assertRaises(ValueError):check_splits([{'qid':'1','query':'x'}],[],[{'qid':'1','query':'y'}])
        with self.assertRaises(ValueError):check_splits([{'qid':'1','query':'x','original_query':'original'}],[],[{'qid':'3','query':'original'}])

    def test_deterministic_inventory_bound_augmentation(self):
        case={'query_hash':'123','inventory':{'tools':['a','b','x','y','z','w','v']},'candidates':[c(['a','b'])]}
        a=build_toolsets(case,G);self.assertEqual(a,build_toolsets(case,G))
        self.assertIn(('a','b'),a);self.assertIn(('a',),a)
        self.assertTrue(any(set(s)>{'a','b'} for s in a))
        self.assertTrue(all(1<=len(s)<=6 for s in a))

    def test_parse_target_and_no_empty_token_leak(self):
        r=parse_target('<LLM_A> <TOOL_SEP> <<api&&a>> <<api&&b>> <SPECIAL_END>')
        self.assertEqual(r,c(['<<api&&a>>','<<api&&b>>']))
        self.assertEqual(parse_target('<LLM_A> <TOOL_SEP> <TOOL_EMPTY> <SPECIAL_END>')['tools'],[])

if __name__=='__main__':unittest.main()
