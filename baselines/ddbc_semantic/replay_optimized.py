#!/usr/bin/env python3
"""Replay fixed predictions from checkpoints without opening any test targets."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from optimize import SemanticAdapter,log_probs,generate,rerank,RVQDiffusion,sha,write
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args();root=Path(str(AC_ROOT));cfg=json.loads((a.run/'config.json').read_text());d=root/cfg['data'];denpath=root/cfg['denoiser'];orig=root/cfg['original'];selection=json.loads((a.run/'selection_frozen.json').read_text())
assert sha(a.run/'semantic_best.pt')==selection['semantic_checkpoint_sha256'];assert sha(denpath/'best.pt')==cfg['denoiser_sha256']
for f,h in cfg['code_sha256'].items():assert sha(Path(__file__).parent/f)==h
seed=cfg['seed'];torch.set_num_threads(4);torch.manual_seed(seed);np.random.seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
cat=json.loads((d/'catalog.json').read_text());codes=np.load(d/'codes.npy');q=torch.tensor(np.load(d/'query_semantics.npy'),device='cuda');c=torch.tensor(np.load(d/'component_semantics.npy'),device='cuda');nllm=sum(x['kind']=='llm' for x in cat)
net=SemanticAdapter(len(cat)).cuda();net.load_state_dict(torch.load(a.run/'semantic_best.pt',map_location='cuda',weights_only=False)['state_dict']);net.eval()
da=json.loads((d/'audit.json').read_text());dc=json.loads((denpath/'config.json').read_text());den=RVQDiffusion(np.load(d/'rvq_centroids.npy'),da['codebook_sizes'],**dc['architecture']).cuda();den.load_state_dict(torch.load(denpath/'best.pt',map_location='cuda',weights_only=False)['model']);den.eval()
inputs=json.loads((d/'test_inputs.json').read_text());old={r['sample_id']:r for r in (json.loads(x) for x in (orig/'predictions_blind.jsonl').open())}
saved={name:[json.loads(x) for x in (a.run/name/'predictions_blind.jsonl').open()] for name in ['rerank_only','guided_selected']}
if a.limit:inputs=inputs[:a.limit]
with torch.inference_mode():
 for index,inp in enumerate(inputs):
  qid=inp['query_index'];prior=log_probs(net(q[qid:qid+1],c,nllm),nllm)[0].exp().cpu().numpy();s=selection['overall'];new=generate(den,q[qid],codes,cat,prior,s['beta'],index,seed) if s['beta'] else None
  for name,key in [('rerank_only','rerank_only'),('guided_selected','overall')]:
   choice=selection[key];item=old[inp['sample_id']] if choice['beta']==0 else new;ranked=rerank(item['all_ranked_proposals'],prior,item['length_probabilities'],choice['alpha']);target=saved[name][index]
   assert inp['sample_id']==target['sample_id'];assert ranked==target['all_ranked_proposals'],(index,name,'ranking mismatch')
   for field in ['length_probabilities','sampled_draws','unique_proposals','failures']:assert item[field]==target[field],(index,field)
  if index%20==0:print(f'Replayed {index+1}/{len(inputs)}',flush=True)
write(a.run/'replay_audit.json',{'exact_all_proposal_values_match':True,'n':len(inputs),'test_labels_read':False,'checkpoint_sha256':selection['semantic_checkpoint_sha256'],'predictions_sha256':{name:sha(a.run/name/'predictions_blind.jsonl') for name in saved}})
print('PASS exact prediction replay',flush=True)
