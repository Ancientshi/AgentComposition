"""Mechanism checks for feasibility pruning and set invariance."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json
from pathlib import Path
import numpy as np
from sampling_v2 import has_distinct_assignment
rng=np.random.default_rng(43)
for k in range(1,7):
 for _ in range(250):
  domains=[rng.choice(8,int(rng.integers(0,9)),replace=False).tolist() for _ in range(k)]
  def brute(i,used):return True if i==k else any(brute(i+1,used|{v}) for v in domains[i] if v not in used)
  assert has_distinct_assignment(domains)==brute(0,set()),domains
assert not has_distinct_assignment([[1,2],[1,2],[1,2]])
assert has_distinct_assignment([[1],[1,2],[1,2,3]])
from round2 import torch,BundleCritic,pad_components,RVQDiffusion,coverage_labels
p=Path(str(AC_ROOT / 'outputs/ddbc_semantic_data_v1'));cat=json.loads((p/'catalog.json').read_text());codes=np.load(p/'codes.npy');audit=json.loads((p/'audit.json').read_text());q=torch.tensor(np.load(p/'query_semantics.npy')[:2],device='cuda');c=torch.tensor(np.load(p/'component_semantics.npy'),device='cuda');nllm=sum(x['kind']=='llm' for x in cat)
torch.manual_seed(42);torch.set_num_threads(4);net=BundleCritic().cuda().eval();prob=torch.ones(2,len(cat),device='cuda')/len(cat)
ids=pad_components([[0,nllm,nllm+1,nllm+2],[0,nllm+2,nllm,nllm+1]])
with torch.no_grad():
 logits=net(q[:1].expand(2,-1),c,ids,prob);assert torch.allclose(logits[0],logits[1],atol=2e-6),logits
zero=pad_components([[0]]);logits=net(q[:1],c,zero,prob[:1]);assert torch.isfinite(logits).all()
net.train();loss=net(q,c,ids,prob).square().mean();loss.backward();assert all(torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None)
lab=coverage_labels([{'components':[0,nllm]}],[{'components':[0,nllm,-1]}]);assert np.allclose(lab,[[2/3,0]])
model=RVQDiffusion(np.load(p/'rvq_centroids.npy'),audit['codebook_sizes']).cuda().eval();model.load_state_dict(torch.load(str(AC_ROOT / 'outputs/ddbc_semantic_rvq_seed42_v1/best.pt'),map_location='cuda',weights_only=False)['model'])
from sampling_v2 import draw_bundles
with torch.no_grad():
 prior=np.full(len(cat),1e-10);prior[0]=1;prior[nllm]=1;prior[nllm+1]=.5
 a,st=draw_bundles(model,q[0],codes,cat,[0,1,2,3,4,5,6],4,42,prior)
 b,st2=draw_bundles(model,q[0],codes,cat,[0,1,2,3,4,5,6],4,42,prior)
 assert a==b and st==st2;assert len(a)==7
 for row in a:
  ii=row['components'];assert len(ii)==len(set(ii));assert cat[ii[0]]['kind']=='llm' and all(cat[i]['kind']=='tool' for i in ii[1:])
print('PASS: 1500 exact matching checks, set permutation invariance, empty tool bundle, finite gradients, unknown-label denominator, legal reproducible diffusion for 0..6 tools.')
