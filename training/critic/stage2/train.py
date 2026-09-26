"""Fine-tune clean V3 with explicit redundancy and missing-tool pair losses."""
import argparse,collections,hashlib,importlib.util,json,math,os,random,sys,time
from pathlib import Path
from core import VERSION,MIX_GRID,blend_scores,ranked_metrics,recall_guard,signature
from prepare import jsonl,sha

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--v3-root',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--base-model',type=Path,required=True)
    ap.add_argument('--easyrec-code',type=Path,required=True)
    ap.add_argument('--critic-code',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--batch-size',type=int,default=24)
    ap.add_argument('--encoder-lr',type=float,default=2e-6)
    ap.add_argument('--head-lr',type=float,default=2e-5)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Refusing to overwrite V4 checkpoint output')
    import numpy as np
    import torch
    from transformers import AutoConfig,AutoTokenizer,get_linear_schedule_with_warmup
    from torch.utils.data import DataLoader
    sys.path.insert(0,str(args.v3_root));from compact_input import serialize,VERSION as INPUT_VERSION
    sys.path.insert(0,str(args.critic_code));from train_bundle_critic_improved import EasyRecBundleCritic
    sys.path.insert(0,str(args.easyrec_code));from model import Easyrec
    if not torch.cuda.is_available():raise RuntimeError('A free CUDA device is required')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    initial_model_args=dict(ck['args'])
    if ck['args'].get('serialization_version')!=INPUT_VERSION:raise ValueError('Expected clean compact V3 initialization')
    config=AutoConfig.from_pretrained(args.base_model,local_files_only=True)
    tok=AutoTokenizer.from_pretrained(args.base_model,local_files_only=True)
    encoder=Easyrec.from_pretrained(args.base_model,config=config,local_files_only=True)
    model=EasyRecBundleCritic(encoder,config.hidden_size,ck['args']['head_hidden'],ck['args']['dropout'],ck['args']['normalize_embedding']).cuda()
    model.load_state_dict(ck['model_state_dict'],strict=True);del ck
    args.output.mkdir(parents=True)
    status=args.output/'status.json'
    def state(stage,**kw):
        status.write_text(json.dumps({'stage':stage,'time':time.time(),'pid':os.getpid(),**kw},indent=2)+'\n')
        print(json.dumps({'stage':stage,**kw}),flush=True)
    state('encoding')
    sets={};encoding={}
    for split in ['train','valid']:
        rows=list(jsonl(args.data/f'cases_{split}.jsonl'));stats=collections.Counter()
        for index,r in enumerate(rows):
            required=set(range(len(r['candidates']))) if split=='valid' else {p[k] for p in r['pairs'] for k in ['positive','negative']}
            seen={}
            for i,c in enumerate(r['candidates']):
                if i not in required:continue
                _,ids,meta=serialize(r['query'],c,r['inventory'],tok,512)
                sig=signature(c)
                if tuple(ids) in seen and seen[tuple(ids)]!=sig:raise ValueError('Different candidates have identical input')
                seen[tuple(ids)]=sig;c['input_ids']=ids
                stats['encoded_candidates']+=1;stats['max_tokens']=max(stats['max_tokens'],len(ids))
            if index%100==0:state('encoding',split=split,queries=index+1,total=len(rows))
        sets[split]=rows;encoding[split]=dict(stats)
    provenance={'version':VERSION,'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                'runtime':{'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,
                           'transformers':__import__('transformers').__version__,'gpu':torch.cuda.get_device_name()},
                'initial_checkpoint_sha256':sha(args.checkpoint),'serializer_sha256':sha(args.v3_root/'compact_input.py'),
                'preparation':json.loads((args.data/'preparation.json').read_text()),'encoding':encoding,
                'selection':'Max mean validation RDCP across original and augmented pools over epochs and a preregistered V3/V4 mix grid, subject to no decline in RDCR, Tool-Hit, CR-Hit and CompR@1 in BOTH pools versus frozen V3.',
                'mix_grid':MIX_GRID,
                'limitation':'Validation pools are reference-informed synthetic pools, not actual generator outputs; test recall stability is not guaranteed.',
                'code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}}
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    def pad(cs):
        return tok.pad({'input_ids':[c['input_ids'] for c in cs]},padding=True,return_tensors='pt').to('cuda')
    @torch.inference_mode()
    def validate():
        model.eval();metrics={'original':[],'augmented':[]};correct=collections.Counter();count=collections.Counter();per=[];saved_scores={}
        for r in sets['valid']:
            cs=r['candidates'];scores=[]
            for j in range(0,len(cs),48):
                with torch.autocast('cuda',dtype=torch.float16):v=model(pad(cs[j:j+48]))
                scores.extend(v.float().cpu().tolist())
            for p in r['pairs']:
                count[p['kind']]+=1;correct[p['kind']]+=scores[p['positive']]>scores[p['negative']]
            saved_scores[r['qid']]=scores
            one={'qid':r['qid']}
            for view,idx in [('original',r['original_candidate_indices']),('augmented',list(range(len(cs))))]:
                m=ranked_metrics([cs[i] for i in idx],[scores[i] for i in idx],r['references']);metrics[view].append(m);one[view]=m
            per.append(one)
        aggregate={v:{k:float(np.mean([m[k] for m in ms])) for k in ms[0]} for v,ms in metrics.items()}
        return {'metrics':aggregate,'pair_accuracy':{k:correct[k]/count[k] for k in count},'pair_counts':dict(count),'per_query':per},saved_scores
    state('baseline_validation');baseline,baseline_scores=validate()
    (args.output/'baseline_validation.json').write_text(json.dumps(baseline,indent=2)+'\n')
    def mixed_validation(trained_scores,alpha):
        views={'original':[],'augmented':[]}
        for r in sets['valid']:
            cs=r['candidates'];b=baseline_scores[r['qid']];t=trained_scores[r['qid']]
            for view,idx in [('original',r['original_candidate_indices']),('augmented',list(range(len(cs))))]:
                scores=blend_scores([b[i] for i in idx],[t[i] for i in idx],alpha)
                views[view].append(ranked_metrics([cs[i] for i in idx],scores,r['references']))
        return {v:{k:float(np.mean([m[k] for m in ms])) for k in ms[0]} for v,ms in views.items()}
    pairs=[(r,p) for r in sets['train'] for p in r['pairs']]
    def collate(batch):
        pos=[r['candidates'][p['positive']] for r,p in batch];neg=[r['candidates'][p['negative']] for r,p in batch]
        return pad(pos+neg),torch.tensor([p['margin'] for r,p in batch],device='cuda'),[p['kind'] for r,p in batch]
    loader=DataLoader(pairs,batch_size=args.batch_size,shuffle=True,collate_fn=collate,num_workers=0)
    opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':args.encoder_lr},{'params':model.scorer.parameters(),'lr':args.head_lr}],weight_decay=.01)
    totalsteps=len(loader)*args.epochs;sched=get_linear_schedule_with_warmup(opt,int(.05*totalsteps),totalsteps)
    scaler=torch.amp.GradScaler('cuda');history=[];best=None;best_alpha=None;best_objective=-math.inf
    modelargs={**initial_model_args,'model_dir':str(args.base_model),'max_length':512,
               'serialization_version':INPUT_VERSION,'training_version':VERSION}
    def checkpoint(path,epoch):
        tmp=path.with_suffix('.tmp');torch.save({'model_state_dict':model.state_dict(),'args':modelargs,'hidden_size':config.hidden_size,
                'epoch':epoch,'training_version':VERSION,'initial_checkpoint_sha256':provenance['initial_checkpoint_sha256']},tmp);tmp.replace(path)
    for epoch in range(1,args.epochs+1):
        model.train();total=0.;start=time.time()
        for step,(batch,margins,kinds) in enumerate(loader,1):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                score=model(batch);n=len(kinds);loss=torch.nn.functional.softplus(margins-(score[:n]-score[n:])).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(opt);scaler.update();sched.step();total+=float(loss.detach())
            if step%100==0:state('training',epoch=epoch,batch=step,batches=len(loader),loss=total/step,elapsed=time.time()-start)
        state('validation',epoch=epoch);val,trained_scores=validate()
        base_obj=sum(baseline['metrics'][v]['RDCP@10'] for v in ['original','augmented'])/2
        mixes=[]
        for alpha in MIX_GRID:
            met=mixed_validation(trained_scores,alpha)
            guard=all(recall_guard(met[v],baseline['metrics'][v]) for v in ['original','augmented'])
            obj=sum(met[v]['RDCP@10'] for v in ['original','augmented'])/2
            mixes.append({'alpha':alpha,'recall_guard_passed':guard,'objective':obj,'metrics':met})
        acceptable=[x for x in mixes if x['alpha']>0 and x['recall_guard_passed'] and x['objective']>base_obj+1e-12]
        winner=max(acceptable,key=lambda x:(x['objective'],-x['alpha'])) if acceptable else None
        record={'epoch':epoch,'train_loss':total/len(loader),'mixes':mixes,**val}
        history.append(record)
        (args.output/f'validation_epoch_{epoch}.json').write_text(json.dumps(record,indent=2)+'\n')
        checkpoint(args.output/'last_critic.pt',epoch)
        if winner and winner['objective']>best_objective:
            checkpoint(args.output/'best_critic.pt',epoch);best=epoch;best_alpha=winner['alpha'];best_objective=winner['objective']
            (args.output/'best_metrics.json').write_text(json.dumps({**record,'selected_mix':winner},indent=2)+'\n')
            (args.output/'ranking_config.json').write_text(json.dumps({'alpha':best_alpha,'baseline_checkpoint':str(args.checkpoint),
                    'baseline_sha256':provenance['initial_checkpoint_sha256'],'formula':'(1-alpha)*z(V3) + alpha*z(V4), normalized within the candidate pool before Top-10 selection',
                    'selected_on':'held-out V3 validation queries; original and augmented candidate pools','epoch':best},indent=2)+'\n')
        (args.output/'metrics.json').write_text(json.dumps({'history':history,'selected_epoch':best},indent=2)+'\n')
        state('epoch_complete',epoch=epoch,metrics=val['metrics'],pair_accuracy=val['pair_accuracy'],selected_epoch=best,selected_alpha=best_alpha)
    result={'training_complete':True,'selected_epoch':best,'accepted_on_synthetic_validation':best is not None,
            'deployment_validated':False,'new_llm_calls':0,'test_used_for_selection':False}
    (args.output/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n');state('complete',**result)

if __name__=='__main__':main()
