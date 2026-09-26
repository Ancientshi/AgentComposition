from run_OURS_CARTESIAN_LLM10_CRITIC_RERANK import build_expansion,select_view,candidate_key

llms=['L1','L2','L3']
completed=[{'llm':'L1','tools':['A','B']},{'llm':'L2','tools':['B','A']},{'llm':'L1','tools':['C']}]
final=[{'llm_token':'L1','tool_tokens':['A','B']},{'llm_token':'L2','tool_tokens':['B','A']}]
products,f,c=build_expansion(final,completed,llms)
assert len(products)==6
assert len({candidate_key(p['llm_token'],p['tool_tokens']) for p in products})==6
for i,p in enumerate(products):p['critic_raw']=float(i)
assert len(select_view(products,'FINAL10_TOOLSETS_X_TOP10_LLM',f,c))==3
assert len(select_view(products,'COMPLETED_TOOLSETS_X_TOP10_LLM',f,c))==6
assert len(select_view(products,'CONTROL_ORIGINAL_FINAL10_CRITIC_ONLY',f,c))==2
assert len(select_view(products,'CONTROL_ORIGINAL_COMPLETED_CRITIC_ONLY',f,c))==3
assert select_view(products,'COMPLETED_TOOLSETS_X_TOP10_LLM',f,c)[0]['critic_raw']==5
assert not any('generator_logprob' in p for p in products)
print('PASS: tool-permutation dedup, exact Cartesian coverage, view membership, critic-only ranking, no stale generator score')
