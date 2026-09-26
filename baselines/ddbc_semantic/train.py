from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
#!/usr/bin/env python3
import argparse,collections,copy,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
from prepare import sha,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(str(AC_ROOT)));p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=6000);p.add_argument('--seed',type=int,default=42);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--sanity',action='store_true');a=p.parse_args()
    sys.path.insert(0,str(a.root));import baseline5_run_infer_rag_gpt as b5;b5.ensure_env_cuda_library()
    import torch
    from torch.nn import functional as F
    from model import RVQDiffusion,EMA,Noise,corrupt_loss,MASK
    torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    # FlashAttention upstream backward does not promise bitwise training determinism.
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'config.json').exists(),'Choose a new run directory'
    ready=json.loads((a.data/'ready.json').read_text())
    for name,h in ready['file_hashes'].items():assert sha(a.data/name)==h,name
    rows=json.loads((a.data/'train.json').read_text());val=json.loads((a.data/'validation.json').read_text());audit=json.loads((a.data/'audit.json').read_text())
    val=[r for r in val if -1 not in r['components'] and r['num_tools']<=6]
    if a.sanity:
        rows=[r for r in rows if r['num_tools']==2][:32];val=rows
    q=torch.tensor(np.load(a.data/'query_semantics.npy'),device='cuda');codes=torch.tensor(np.load(a.data/'codes.npy'),device='cuda')
    model=RVQDiffusion(np.load(a.data/'rvq_centroids.npy'),audit['codebook_sizes']).cuda()
    ema=EMA(model.parameters(),.9999);noise=Noise();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.999),weight_decay=0,eps=1e-8)
    grouped=collections.defaultdict(list)
    for r in rows:grouped[r['num_tools']].append(r)
    # Group equal-length bundles to reuse official unmodified FlashAttention blocks (no padding).
    batches=[]
    def epoch_batches():
        result=[]
        for k,group in sorted(grouped.items()):
            order=np.random.permutation(len(group))
            for ix in [order[j:j+a.batch_size] for j in range(0,len(order),a.batch_size)]:result.append([group[i] for i in ix])
        random.shuffle(result);return result
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update({'data_ready_sha256':sha(a.data/'ready.json'),'architecture':{'hidden':128,'blocks':6,'heads':8,'dropout':.1},'optimizer':'AdamW','lr':3e-4,'warmup_steps':200 if not a.sanity else 20,'ema':.9999,
        'official_commit':'e1f4afc59121662e626ffefde79316a5e0dde044','selection':'minimum fixed-corruption validation masked CE + 0.2 length CE; EMA parameters',
        'validation_representable_rows':len(val),'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'attention_kernel':'PyTorch SDPA, equivalent RoPE; official DDiT equations retained',
        'code_hashes':{str(f.relative_to(Path(__file__).parent)):sha(f) for f in [Path(__file__),Path(__file__).parent/'model.py',Path(__file__).parent/'upstream/models/dit.py',Path(__file__).parent/'upstream/models/dit_torch.py',Path(__file__).parent/'upstream/models/ema.py',Path(__file__).parent/'upstream/noise_schedule.py']}})
    write(out/'config.json',config)
    @torch.no_grad()
    def validation(shuffle_query=False,all_mask=False):
        model.eval();sums=collections.Counter();num=0
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(117);torch.cuda.manual_seed_all(117)
            groups=collections.defaultdict(list)
            for r in val:groups[r['num_tools']].append(r)
            for k,group in sorted(groups.items()):
                for start in range(0,len(group),128):
                    rr=group[start:start+128];ids=torch.tensor([r['components'] for r in rr],device='cuda');qq=q[[r['query_index'] for r in rr]]
                    if shuffle_query:qq=q[torch.randperm(len(q),device='cuda')[:len(qq)]]
                    x=model.serialize(ids,codes);pos=model.code_positions(k);n=len(rr)
                    t=torch.tensor([.25,.5,.75,.999],device='cuda')[torch.arange(n,device='cuda')%4]
                    if all_mask:t=torch.full((n,),.999,device='cuda')
                    sigma,_=noise(t);mask=torch.rand((n,len(pos)),device='cuda')<(-torch.expm1(-sigma))[:,None]
                    if all_mask:mask[:]=True
                    xt=x.clone();xt[:,pos]=x[:,pos].masked_fill(mask,MASK)
                    z=model.code_logits(model(xt,qq,sigma),k);ce=F.cross_entropy(z.transpose(1,2),x[:,pos],reduction='none')
                    sums['ce_sum']+=float((ce*mask).sum());sums['tokens']+=int(mask.sum());sums['correct']+=int(((z.argmax(-1)==x[:,pos])&mask).sum())
                    sums['length_ce']+=float(F.cross_entropy(model.length_logits(qq),torch.full((n,),k,device='cuda',dtype=torch.long),reduction='sum'));num+=n
        metrics={'masked_ce':sums['ce_sum']/sums['tokens'],'code_accuracy':sums['correct']/sums['tokens'],'length_ce':sums['length_ce']/num,'n':num}
        metrics['selection_loss']=metrics['masked_ce']+.2*metrics['length_ce'];return metrics
    initial=validation();write(out/'initial_validation.json',initial);history=[];best=float('inf');beststep=0;start=time.time()
    interval=100 if a.sanity else 250
    for step in range(1,a.steps+1):
        if not batches:batches=epoch_batches()
        rr=batches.pop();model.train();ids=torch.tensor([r['components'] for r in rr],device='cuda');qq=q[[r['query_index'] for r in rr]]
        for pg in optimizer.param_groups:pg['lr']=3e-4*min(step/config['warmup_steps'],1.)
        loss,detail=corrupt_loss(model,qq,ids,codes,noise)
        if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();ema.update(model.parameters())
        if step%interval==0 or step==a.steps:
            ema.store(model.parameters());ema.copy_to(model.parameters());v=validation();record={'step':step,'train_last_batch':detail,'validation':v,'elapsed_seconds':time.time()-start};history.append(record)
            if v['selection_loss']<best:
                best=v['selection_loss'];beststep=step;torch.save({'model':model.state_dict(),'step':step,'validation':v},out/'best.pt')
            ema.restore(model.parameters());write(out/'history.json',history);print(json.dumps(record),flush=True)
    # Final is saved separately; test always uses validation-selected best.
    ema.copy_to(model.parameters());torch.save({'model':model.state_dict(),'step':a.steps},out/'last.pt')
    model.load_state_dict(torch.load(out/'best.pt',map_location='cuda',weights_only=False)['model'])
    diagnostics={'initial':initial,'best':validation(),'best_all_mask':validation(all_mask=True),'shuffled_query':validation(shuffle_query=True),'best_step':beststep,'training_seconds':time.time()-start}
    write(out/'diagnostics.json',diagnostics);print(json.dumps(diagnostics,indent=2),flush=True)
    write(out/'completed.json',{'best_sha256':sha(out/'best.pt'),'last_sha256':sha(out/'last.pt'),'best_step':beststep})
if __name__=='__main__':main()
