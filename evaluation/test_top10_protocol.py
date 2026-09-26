"""Regression checks for distinct-agent parsing, missing-rank scoring and beams."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import math
import baseline5_run_infer_rag_gpt as b5
b5.ensure_env_cuda_library()
import baseline_top10_common as common
import baseline3_train_text2bundle_agent_adapted as b3
from baseline3_infer_top10 import beam_decode
import torch
import tempfile
from pathlib import Path

b4,sota=common.modules()
llms=['<LLM_A>','<LLM_B>']
tools=['<<T&&x>>','<<T&&y>>']
line='<LLM_A> <TOOL_SEP> <<T&&x>> <<T&&y>> <SPECIAL_END>'
reordered='<LLM_A> <TOOL_SEP> <<T&&y>> <<T&&x>> <SPECIAL_END>'
second='<LLM_B> <TOOL_SEP> <<T&&x>> <SPECIAL_END>'
results,rejected=common.parse_candidates('\n'.join([line,reordered,second]),llms,tools,b4,sota)
assert len(results)==2 and rejected['duplicate']==1
unknown=common.parse_candidates('<LLM_Z> <TOOL_SEP> <<T&&x>> <SPECIAL_END>',llms,tools,b4,sota)[0]
assert len(unknown)==1 and not unknown[0]['inventory_valid']
no_end=common.parse_candidates(line.replace('<SPECIAL_END>',''),llms,tools,b4,sota)[0]
assert len(no_end)==1 and not no_end[0]['end_marker_present']
assert common.parse_candidates('<LLM_A> <TOOL_SEP> <<T&&unfin',llms,tools,b4,sota)[0]==[]
assert len(common.parse_candidates(line.replace('<SPECIAL_END>','')+'\n'+second,llms,tools,b4,sota)[0])==2
assert len(common.parse_candidates(line.replace('>> <<','>>, <<'),llms,tools,b4,sota)[0])==1
assert len(common.parse_candidates(line.replace('<SPECIAL_END>','<<SPECIAL_END>>'),llms,tools,b4,sota)[0])==1
ev=b5.load_module(common.ROOT/'evaluation/reference/exp3/evaluate_ranked_recall_baseline2.py','test_top10_eval')
record={'dataset_example':{'target':line},'results':results[:1]}
metric=ev.evaluate_one(record)
assert abs(metric['rdcr@10']-1/sum(1/math.log2(r+1) for r in range(1,11)))<1e-8
record['results']=results
assert abs(ev.evaluate_one(record)['cr_mrr@10']-1)<1e-8
record['baseline']='test'
with tempfile.TemporaryDirectory() as tmp:
    report=common.evaluate(Path(tmp),[record],1)
    assert report['mean']['cr_hit@1']==1 and not report['all_have_ten']

class DummyPolicy:
    def __call__(self,q,cand,cm,sm):
        return torch.zeros((q.shape[0],cand.shape[1]+1))
r=b3.PreparedRecord(row_index=0,qid='test',query_index=0,candidate_tool_ids=[0,1,2,3],
    gold_tool_ids=[],retrieved_tool_ids=[0,1,2,3],gold_llm='',selected_llm='<LLM_A>',
    gold_tool_tokens=[],candidate_surface_by_tool_id={},injected_gold_count=0)
runtime=b3.RuntimeData(torch.ones(1,2),torch.ones(4,2),[],[r])
bundles=beam_decode(DummyPolicy(),runtime,r,torch.device('cpu'),b3,width=50,count=10,max_tools=3)
assert len(bundles)==10
assert bundles[0][0]==[0,1,2]
assert len({tuple(sorted(s)) for s,_ in bundles})==10
assert all(1<=len(s)<=3 and len(set(s))==len(s) and math.isfinite(score) for s,score in bundles)
assert all(bundles[i][1]>=bundles[i+1][1] for i in range(9))
print('PASS: permutation dedup, invalid/incomplete rejection, missing-rank denominator, distinct scored beam bundles')
