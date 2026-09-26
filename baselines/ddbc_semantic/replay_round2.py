#!/usr/bin/env python3
"""Replay round-two predictions without opening test labels."""
import argparse,json
from pathlib import Path
import numpy as np
from round2 import torch,ROOT,SemanticAdapter,log_probs,RVQDiffusion,BundleCritic,generate,critic_scores,rank_items,to_prediction,sha,write
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--limit',type=int,default=0);a=p.parse_args();cfg=json.loads((a.run/'config.json').read_text());sel=json.loads((a.run/'selection_frozen.json').read_text());data=ROOT/cfg['data'];previous=ROOT/cfg['previous'];denpath=ROOT/cfg['denoiser'];seed=cfg['seed']
for f,h in cfg['code_sha256'].items():assert sha(Path(__file__).parent/f)==h,f
assert sha(a.run/'critic_best.pt')==sel['critic_sha256'];assert sha(previous/'semantic_best.pt')==cfg['semantic_sha256'];assert sha(denpath/'best.pt')==cfg['denoiser_sha256']
ready=json.loads((data/'ready.json').read_text())
for f,h in ready['file_hashes'].items():assert sha(data/f)==h
assert sha(data/'ready.json')==cfg['data_sha256']
torch.set_num_threads(4);torch.manual_seed(seed);np.random.seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
cat=json.loads((data/'catalog.json').read_text());codes=np.load(data/'codes.npy');audit=json.loads((data/'audit.json').read_text());dc=json.loads((denpath/'config.json').read_text());nllm=sum(x['kind']=='llm' for x in cat)
q=torch.tensor(np.load(data/'query_semantics.npy'),device='cuda');c=torch.tensor(np.load(data/'component_semantics.npy'),device='cuda')
den=RVQDiffusion(np.load(data/'rvq_centroids.npy'),audit['codebook_sizes'],**dc['architecture']).cuda();den.load_state_dict(torch.load(denpath/'best.pt',map_location='cuda',weights_only=False)['model']);den.eval()
sem=SemanticAdapter(len(cat)).cuda();sem.load_state_dict(torch.load(previous/'semantic_best.pt',map_location='cuda',weights_only=False)['state_dict']);sem.eval()
with torch.no_grad():probs=torch.cat([log_probs(sem(q[i:i+64],c,nllm),nllm).exp() for i in range(0,len(q),64)],0)
priors=probs.cpu().numpy();critic=BundleCritic().cuda();critic.load_state_dict(torch.load(a.run/'critic_best.pt',map_location='cuda',weights_only=False)['state_dict']);critic.eval()
variants={'critic_only':sel['critic_only'],'sampling_only':sel['sampling_only'],'selected':sel['final']};saved={name:[json.loads(s) for s in (a.run/name/'predictions_blind.jsonl').open()] for name in variants};old={r['sample_id']:r for r in (json.loads(s) for s in (previous/'guided_selected/predictions_blind.jsonl').open())};inputs=json.loads((data/'test_inputs.json').read_text())
if a.limit:inputs=inputs[:a.limit]
with torch.inference_mode():
 for index,inp in enumerate(inputs):
  qid=inp['query_index'];cache={'previous':old[inp['sample_id']]};scores={}
  for mode in dict.fromkeys(v['mode'] for v in variants.values()):
   if mode!='previous':cache[mode]=generate(den,q[qid],codes,cat,priors[qid],mode,index,seed)
   scores[mode]=critic_scores(critic,q,c,probs,qid,cache[mode]['all_ranked_proposals'])
  for name,v in variants.items():
   item=cache[v['mode']];ranked=rank_items(item['all_ranked_proposals'],priors[qid],item['length_probabilities'],scores[v['mode']],v['gamma']);prediction=to_prediction(inp['sample_id'],item,ranked,cat)
   assert prediction==saved[name][index],(name,index,'replay mismatch')
  if index%20==0:print(f'Replayed {index+1}/{len(inputs)}',flush=True)
write(a.run/'replay_audit.json',{'exact_all_prediction_fields_match':True,'n':len(inputs),'test_labels_read':False,'predictions_sha256':{name:sha(a.run/name/'predictions_blind.jsonl') for name in variants},'critic_sha256':sel['critic_sha256']})
print('PASS exact round-two replay',flush=True)
