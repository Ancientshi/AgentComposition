from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import os, json, sys, time, random, argparse, hashlib
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel
from stopping import stop_reason
R=Path(str(AC_ROOT));W=R/'experiments/domain_generalization/quickadapt';D=W/'data';INIT=R/'checkpoints/generative_v10_compact';BASE=str(AC_BASE_MODEL)
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);a=ap.parse_args();job=next(j for j in json.loads((D/'manifest.json').read_text())['jobs'] if j['name']==a.job)
 O=W/'runs'/a.job;assert not O.exists(),'Refusing overwrite';O.mkdir(parents=True)
 audit=json.loads((D/'audit.json').read_text())
 for f in [a.job+'.jsonl','val.jsonl']:assert hashlib.sha256((D/f).read_bytes()).hexdigest()==audit['data_hashes'][f]
 train=rows(D/(a.job+'.jsonl'));val=rows(D/'val.jsonl');manifest=json.loads((D/'manifest.json').read_text());forbidden=set(manifest['query_ids']['test'])
 assert not {r['sample_id'] for r in train+val}&forbidden
 set_seed(job['seed']);tok=AutoTokenizer.from_pretrained(INIT,local_files_only=True);tok.pad_token=tok.eos_token
 base=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
 model=PeftModel.from_pretrained(base,INIT,is_trainable=True)
 for n,p in model.named_parameters():p.requires_grad_('lora_' in n)
 model=model.cuda();model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=0.)
 def batch(row):return {k:torch.tensor([row[k]],device='cuda') for k in ['input_ids','attention_mask','labels']}
 def evaluate():
  model.eval();byquery=defaultdict(list);tasks={}
  with torch.inference_mode():
   for row in val:
    loss=float(model(**batch(row)).loss);byquery[row['sample_id']].append(loss);tasks[row['sample_id']]=row['task_id']
  bytask=defaultdict(list)
  for sid,v in byquery.items():bytask[tasks[sid]].append(sum(v)/len(v))
  per_task={k:sum(v)/len(v) for k,v in bytask.items()};return sum(per_task.values())/len(per_task),per_task
 def snapshot():return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
 start=time.time();loss,pt=evaluate();history=[{'step':0,'loss':loss,'per_task':pt}];best=loss;beststep=0;weights=snapshot();reason='maximum_updates'
 write(O/'protocol.json',{'job':job,'initial_checkpoint':str(INIT),'lr':1e-5,'lr_schedule':'constant','effective_batch':4,'max_updates':40,'eval_every':2,'critic':False,'selection':'minimum task-macro validation completion loss including step zero','validation_aggregation':'tokens within completion; positives within query; queries within task; equal tasks','template_control':'source rehearsal; outer instruction already identical','trainable_parameters':sum(p.numel() for p in params)})
 write(O/'history.json',history);print(a.job,'STEP',0,'VAL',loss,flush=True)
 rng=random.Random(job['seed']);order=[];cursor=0
 for step in range(1,41):
  model.train();opt.zero_grad(set_to_none=True);tl=0
  for _ in range(4):
   if cursor==len(order):order=list(range(len(train)));rng.shuffle(order);cursor=0
   row=train[order[cursor]];cursor+=1;out=model(**batch(row));tl+=float(out.loss.detach())/4;(out.loss/4).backward();del out
  torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
  if step%2:continue
  loss,pt=evaluate();history.append({'step':step,'loss':loss,'train_loss_last_batch':tl,'per_task':pt,'elapsed_sec':time.time()-start})
  if loss<best:best=loss;beststep=step;weights=snapshot()
  write(O/'history.json',history);print(a.job,'STEP',step,'VAL',loss,'BEST',beststep,flush=True)
  why=stop_reason(history)
  if why:reason=why;break
 if reason=='nonfinite':raise RuntimeError('Nonfinite validation loss')
 with torch.no_grad():
  for n,p in model.named_parameters():
   if n in weights:p.copy_(weights[n].to(device=p.device,dtype=p.dtype))
 model.save_pretrained(O/'best');tok.save_pretrained(O/'best')
 write(O/'COMPLETE.json',{'job':job,'best_step':beststep,'best_loss':best,'stopped_step':step,'stop_reason':reason,'elapsed_sec':time.time()-start,'checkpoint':str(O/'best')})
 print('COMPLETE',a.job,flush=True)
if __name__=='__main__':main()
