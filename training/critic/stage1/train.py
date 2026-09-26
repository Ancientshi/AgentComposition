from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import os,sys,json,pathlib,random,math,time,hashlib,collections
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from transformers import AutoConfig,AutoTokenizer,get_linear_schedule_with_warmup
ROOT=pathlib.Path(__file__).resolve().parent;D=AC_ROOT/'datasets/critic_stage1';OUT=AC_ROOT/'checkpoints/bundle_critic_terra_v3'
sys.path.insert(0,str(AC_ROOT/'models'));from train_bundle_critic_improved import EasyRecBundleCritic
sys.path.insert(0,str(AC_ROOT / 'models/easyrec'));from model import Easyrec
from compact_input import VERSION
BASE=str(AC_EASYREC_MODEL)
def read(split):return [json.loads(x) for x in (D/f'encoded_{split}.jsonl').read_text().splitlines()]
def pairs(rows):
 out=[];counts=collections.Counter()
 for r in rows:
  cs=r['candidates'];groups=collections.defaultdict(list);rng=random.Random(r['query_hash'])
  for i,a in enumerate(cs):
   for b in cs[i+1:]:
    if abs(a['score']-b['score'])<.05:continue
    pos,neg=(a,b) if a['score']>b['score'] else (b,a)
    kind='llm' if set(a['tools'])==set(b['tools']) else ('tools' if a['llm']==b['llm'] else 'both')
    groups[kind].append((pos,neg,kind))
  for kind,n in [('llm',64),('tools',32),('both',32)]:
   rng.shuffle(groups[kind]);chosen=groups[kind][:n];out.extend(chosen);counts[kind]+=len(chosen)
 return out,dict(counts)
def main():
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 if OUT.exists() and any(OUT.iterdir()):raise RuntimeError('Output nonempty: inspect, do not silently overwrite/resume')
 OUT.mkdir(parents=True,exist_ok=True)
 train,valid=read('train'),read('valid');assert not {r['query_hash'] for r in train}&{r['query_hash'] for r in valid};assert not {r['qid'] for r in train}&{r['qid'] for r in valid}
 tp,tc=pairs(train);vp,vc=pairs(valid);assert tp and vp
 tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
 def pad(candidates):return tok.pad({'input_ids':[c['input_ids'] for c in candidates]},padding=True,return_tensors='pt').to('cuda')
 def collate(batch):
  return pad([p for p,n,k in batch]+[n for p,n,k in batch]),torch.tensor([p['score'] for p,n,k in batch]+[n['score'] for p,n,k in batch],device='cuda'),[k for p,n,k in batch]
 loader=DataLoader(tp,batch_size=16,shuffle=True,collate_fn=collate,num_workers=0)
 cfg=AutoConfig.from_pretrained(BASE,local_files_only=True);encoder=Easyrec.from_pretrained(BASE,config=cfg,local_files_only=True)
 model=EasyRecBundleCritic(encoder,cfg.hidden_size,512,.1,True).cuda()
 opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':5e-6},{'params':model.scorer.parameters(),'lr':1e-4}],weight_decay=.01)
 epochs=4;accum=2;steps=math.ceil(len(loader)/accum)*epochs;sched=get_linear_schedule_with_warmup(opt,int(.06*steps),steps);scaler=torch.amp.GradScaler('cuda')
 args={'model_dir':BASE,'max_length':512,'head_hidden':512,'dropout':.1,'normalize_embedding':True,'serialization_version':VERSION,'teacher':'gpt-5.6-terra','epochs':epochs,'encoder_lr':5e-6,'head_lr':1e-4,'batch':16,'grad_accum':accum,'seed':42}
 prov={'args':args,'encoding':json.loads((D/'encoding_report.json').read_text()),'preparation':json.loads((D/'preparation.json').read_text()),'train_pairs':tc,'valid_pairs':vc,'code_hashes':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['train.py','compact_input.py','encode_dataset.py','label_terra.py']}}
 (OUT/'provenance.json').write_text(json.dumps(prov,indent=2));print(json.dumps({'train_pairs':tc,'valid_pairs':vc,'total_steps':steps}),flush=True)
 best=-1;history=[];globalstep=0
 for epoch in range(1,epochs+1):
  model.train();opt.zero_grad();total=0;start=time.time()
  for i,(batch,target,kinds) in enumerate(loader,1):
   with torch.autocast('cuda',dtype=torch.float16):
    score=model(batch);n=len(kinds);gap=target[:n]-target[n:];weight=torch.sqrt(gap.clamp(min=.05));rank=(-torch.nn.functional.logsigmoid(score[:n]-score[n:])*weight).sum()/weight.sum();cal=torch.nn.functional.mse_loss(score.sigmoid(),target);loss=rank+.1*cal
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
   scaler.scale(loss/accum).backward();total+=float(loss.detach())
   if i%accum==0 or i==len(loader):
    scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update();sched.step();opt.zero_grad();globalstep+=1
   if i%100==0:print(json.dumps({'epoch':epoch,'batch':i,'batches':len(loader),'loss':total/i,'elapsed':time.time()-start}),flush=True)
  model.eval();scores={};ndcgs=[];top=[]
  with torch.inference_mode():
   for r in valid:
    cs=r['candidates'];ss=[]
    for j in range(0,len(cs),32):
     with torch.autocast('cuda',dtype=torch.float16):vals=model(pad(cs[j:j+32]))
     ss.extend(vals.float().cpu().tolist())
    assert all(math.isfinite(x) for x in ss)
    for c,s in zip(cs,ss):scores[id(c)]=s
    order=sorted(range(len(cs)),key=lambda i:ss[i],reverse=True);ideal=sorted([c['score'] for c in cs],reverse=True)
    dcg=sum((2**(cs[i]['score']*3)-1)/math.log2(j+2) for j,i in enumerate(order[:10]));idcg=sum((2**(s*3)-1)/math.log2(j+2) for j,s in enumerate(ideal[:10]));ndcgs.append(dcg/idcg if idcg else 1.);top.append(cs[order[0]]['score']>=ideal[0]-.01)
  pc=collections.Counter();pn=collections.Counter()
  for a,b,k in vp:pc[k]+=scores[id(a)]>scores[id(b)];pn[k]+=1
  met={'epoch':epoch,'train_loss':total/len(loader),'list_ndcg':float(np.mean(ndcgs)),'teacher_top1_tie_accuracy':float(np.mean(top)),'pair_accuracy':{k:pc[k]/pn[k] for k in pn},'global_update':globalstep};history.append(met);print('[VALID]',json.dumps(met),flush=True)
  if met['list_ndcg']>best:
   best=met['list_ndcg'];ck={'model_state_dict':model.state_dict(),'args':args,'hidden_size':cfg.hidden_size,'best_list_ndcg':best,'epoch':epoch};torch.save(ck,OUT/'best_critic.tmp');(OUT/'best_critic.tmp').replace(OUT/'best_critic.pt');tok.save_pretrained(OUT/'tokenizer');(OUT/'best_metrics.json').write_text(json.dumps(met,indent=2))
  (OUT/'metrics.json').write_text(json.dumps({'history':history,'best_list_ndcg':best,'args':args},indent=2))
 (OUT/'COMPLETE.json').write_text(json.dumps({'epochs':epochs,'best_ndcg':best,'global_steps':globalstep}));print('[COMPLETE]',OUT,flush=True)
if __name__=='__main__':main()
