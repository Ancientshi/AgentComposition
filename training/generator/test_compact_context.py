import unittest
from compact_context import build_prompt, components, encode_supervision, parse_context

class TinyTokenizer:
    eos_token='[EOS]'
    def encode(self,text,add_special_tokens=True):return ([1] if add_special_tokens else [])+list(text.encode())
    def decode(self,ids,skip_special_tokens=False):return bytes(ids).decode(errors='ignore')

class CompactTests(unittest.TestCase):
    def setUp(self):
        self.t=TinyTokenizer()
        self.context=('cf_retrieved_llm:\n  <LLM_A> [1], <LLM_B> [2]\n\n'
          'cf_retrieved_tool_bundle:\n  {<<API&&/users/{id}>>, <TOOL_search>} [3]\n\n'
          'semantic_retrieved_tool:\n  <<API&&/users/{id}>> [4]\n\nevidence:\n'
          '  [1] CF-LLM | token=<LLM_A> | strengths=reasoning | desc=reasoning model\n'
          '  [3] CF-TOOL-BUNDLE | support_question=DO NOT COPY THIS | tools=<<API&&/users/{id}>>: user lookup; <TOOL_search>: search web\n'
          '  [4] SEMANTIC-TOOL | token=<<API&&/users/{id}>> | matched_subquery=VERY LONG REPEATED QUERY | desc=marketing ' + 'x'*500 + ' Api Description: Find user by id')
    def test_preserves_candidates_and_query(self):
        p,s=build_prompt(self.context,'Find user 123',self.t,2000)
        self.assertEqual(set(sum(components(self.context),[])),set(sum(components(p),[])))
        self.assertIn('Find user 123\n\n### Answer:\n',p)
        self.assertNotIn('DO NOT COPY THIS',p);self.assertNotIn('VERY LONG REPEATED QUERY',p)
        self.assertLessEqual(s['prompt_tokens'],2000)
        self.assertEqual(parse_context(self.context)[2],[['<<API&&/users/{id}>>','<TOOL_search>']])
    def test_budget_never_drops_query_or_candidates(self):
        with self.assertRaises(ValueError):build_prompt(self.context,'z'*1000,self.t,200)
    def test_complete_supervision_not_prompt(self):
        p,_=build_prompt(self.context,'Find user 123',self.t,2000)
        target='<LLM_A> <TOOL_SEP> <<API&&/users/{id}>> <SPECIAL_END>'
        e=encode_supervision(p,target,self.t,4096);n=len(self.t.encode(p))
        self.assertTrue(all(x==-100 for x in e['labels'][:n]));self.assertEqual(e['labels'][n:],e['input_ids'][n:])
        with self.assertRaises(ValueError):encode_supervision(p,target,self.t,n+5)
    def test_injected_gold_hint_is_not_copied(self):
        c=self.context+'\n[5] X | token=<LLM_B> | desc=Gold target injected for supervised target'
        p,_=build_prompt(c,'q',self.t,2000);self.assertNotIn('Gold target',p)

if __name__=='__main__':unittest.main()
