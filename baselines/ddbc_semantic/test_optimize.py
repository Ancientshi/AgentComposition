from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json,numpy as np,torch
from pathlib import Path
from optimize import SemanticAdapter,targets,log_probs,rerank
from infer import draw_bundles
from guided import draw_bundles as guided
from model import RVQDiffusion
p=Path(str(AC_ROOT / 'outputs/ddbc_semantic_data_v1'));cat=json.loads((p/'catalog.json').read_text());au=json.loads((p/'audit.json').read_text());codes=np.load(p/'codes.npy');q=torch.tensor(np.load(p/'query_semantics.npy')[0],device='cuda');c=torch.tensor(np.load(p/'component_semantics.npy'),device='cuda')
torch.manual_seed(42);torch.set_num_threads(4)
m=RVQDiffusion(np.load(p/'rvq_centroids.npy'),au['codebook_sizes']).cuda().eval()
with torch.no_grad():
 a=draw_bundles(m,q,codes,cat,[1,2],2,77);b=guided(m,q,codes,cat,[1,2],2,77,np.ones(len(cat)),0)
 assert a==b,'Beta zero must preserve sampling exactly'
 cc,fail=guided(m,q,codes,cat,[1,2],2,77,np.ones(len(cat)),1)
 for row in cc:
  ids=row['components'];assert cat[ids[0]]['kind']=='llm' and all(cat[i]['kind']=='tool' for i in ids[1:]);assert len(ids)==len(set(ids))
nllm=sum(x['kind']=='llm' for x in cat);net=SemanticAdapter(len(cat)).cuda();t=targets([{'components':[0,nllm,-1]}],len(cat));assert abs(float(t.sum())-2/3)<1e-6
lp=log_probs(net(q[None],c,nllm),nllm);assert torch.allclose(lp[:,:nllm].exp().sum(),torch.tensor(1.,device='cuda'));loss=-(lp*t).sum();loss.backward();assert all(torch.isfinite(x.grad).all() for x in net.parameters() if x.grad is not None)
props=[{'components':[0,nllm],'ranking_score':-1},{'components':[1,nllm+1],'ranking_score':-2}];prob=np.zeros(len(cat));prob[1]=1;prob[nllm+1]=1
assert rerank(props,prob,[0,1,0,0,0,0,0],0)[0]['components']==props[0]['components']
assert rerank(props,prob,[0,1,0,0,0,0,0],1)[0]['components']==props[1]['components']
print('PASS: zero-guidance equivalence, guided legality, missing-label denominator, typed probability normalization, finite gradients, ranking endpoints')
