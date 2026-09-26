"""Check that root count and per-LLM tool beams cannot silently collapse."""
import copy
import run_infer_v13_sota_LLM_ROOT_COVERAGE_ABLATION as s

def node(i,llm,score):
    n=s.SearchNode(node_id=str(i),parent_id='',depth=1,llm=llm,tools=(f'T{i}',),
        suffix_ids=[i],generator_logprob=score,generator_token_count=1,last_action=f'T{i}')
    n.search_score=score
    return n

nodes=[node(i*5+j,f'LLM{i}',-100*i-j) for i in range(10) for j in range(5)]
kwargs=dict(stage='tool_depth_1',retain_ratio=.4,min_beam_size=1,max_beam_size=2)
global_kept,_,_=s.prune_tool_nodes_with_scope(copy.deepcopy(nodes),scope='global',**kwargs)
assert len(global_kept)==2 and len({n.llm for n in global_kept})==1
kept,_,summary=s.prune_tool_nodes_with_scope(copy.deepcopy(nodes),scope='per_llm',**kwargs)
assert len(kept)==20 and len({n.llm for n in kept})==10
assert all(g['retained_count']==2 for g in summary['groups'])
original,_,_=s.prune_nodes(copy.deepcopy(nodes),**kwargs)
assert [n.node_id for n in original]==[n.node_id for n in global_kept]
roots=[node(i,f'LLM{i}',-i) for i in range(10)]
for k in [2,4,10]:
    kept,_,_=s.prune_nodes(copy.deepcopy(roots),stage='llm_roots',retain_ratio=1,
                         min_beam_size=k,max_beam_size=k)
    assert len(kept)==k
empty,_,_=s.prune_tool_nodes_with_scope([],scope='per_llm',**kwargs)
assert not empty
print('PASS: independent root counts; per-LLM beam retains 10 LLMs despite score imbalance; legacy global behavior preserved')
