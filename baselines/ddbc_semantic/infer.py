#!/usr/bin/env python3
"""Query-conditional RVQ-code diffusion; test labels enter only final evaluation."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse,collections,json,math,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from prepare import sha,write

class LegalCodes:
    def __init__(self,codes,catalog):
        self.codes=codes;self.pool={kind:np.array([c['id'] for c in catalog if c['kind']==kind]) for kind in ['llm','tool']};self.cache={}
    def candidates(self,kind,partial):
        key=(kind,tuple(partial))
        if key not in self.cache:
            ix=self.pool[kind]
            for l,v in enumerate(partial):
                if v>=0:ix=ix[self.codes[ix,l]==v]
            self.cache[key]=ix
        return self.cache[key]


def draw_bundles(model,query,codes,catalog,lengths,steps,seed):
    import torch
    from model import MASK
    rng=np.random.default_rng(seed);legality=LegalCodes(codes,catalog);device=query.device
    proposals=[];failures=collections.Counter()
    for k in sorted(set(lengths)):
        n=sum(x==k for x in lengths);x=model.empty(n,k,device).cpu().numpy();partial=np.full((n,k+1,4),-1,dtype=np.int64)
        invalid=np.zeros(n,dtype=bool);score=np.zeros(n);qq=query[None].expand(n,-1)
        for step in range(steps,0,-1):
            raw=model(torch.tensor(x,device=device),qq,torch.full((n,),-math.log(1-.999*step/steps),device=device)).cpu().numpy()
            reveals=(partial<0)&(rng.random(partial.shape)<1/step)&(~invalid[:,None,None])
            for b in range(n):
                if invalid[b]:continue
                positions=np.argwhere(reveals[b]);rng.shuffle(positions)
                for j,l in positions:
                    kind='llm' if j==0 else 'tool';ids=legality.candidates(kind,partial[b,j])
                    if j>0:
                        used=set()
                        for jj in range(1,k+1):
                            if jj!=j and np.all(partial[b,jj]>=0):used.update(legality.candidates('tool',partial[b,jj]).tolist())
                        if used:ids=ids[~np.isin(ids,list(used))]
                    if not len(ids):invalid[b]=True;failures['constraint_dead_end']+=1;break
                    values=np.unique(codes[ids,l]);pos=2+5*j+l;tokenids=values+model.offsets[l]
                    logits=raw[b,pos,tokenids].astype(np.float64);probs=np.exp(logits-logits.max());probs/=probs.sum()
                    choice=int(rng.choice(len(values),p=probs));v=values[choice]
                    partial[b,j,l]=v;x[b,pos]=v+model.offsets[l];score[b]+=math.log(max(probs[choice],1e-300))
        for b in range(n):
            if invalid[b]:continue
            if np.any(partial[b]<0):raise AssertionError('Unresolved MASK')
            ids=[]
            for j in range(k+1):
                options=legality.candidates('llm' if j==0 else 'tool',partial[b,j]);assert len(options)==1;ids.append(int(options[0]))
            assert len(set(ids[1:]))==k
            proposals.append({'components':[ids[0]]+sorted(ids[1:]),'reverse_path_logp':float(score[b])})
    return proposals,dict(failures)


def rank_bundles(model,q,codes,proposals,length_probs):
    """Three tool permutations, leave-one-component-out code pseudo likelihood."""
    import torch
    from model import MASK
    groups=collections.defaultdict(list)
    for i,p in enumerate(proposals):groups[len(p['components'])-1].append(i)
    scores=np.zeros(len(proposals));counts=np.zeros(len(proposals));rng=np.random.default_rng(90210)
    for k,indices in groups.items():
        tasks=[]
        for i in indices:
            original=proposals[i]['components']
            for perm in range(3):
                tools=np.array(original[1:],dtype=np.int64);rng.shuffle(tools);comp=[original[0]]+tools.tolist()
                for hidden in range(k+1):tasks.append((i,comp,hidden))
        for start in range(0,len(tasks),128):
            batch=tasks[start:start+128];comps=torch.tensor([t[1] for t in batch],device=q.device)
            x=model.serialize(comps,codes);clean=x.clone();positions=[]
            for b,(_,_,hidden) in enumerate(batch):
                pos=[2+5*hidden+l for l in range(4)];x[b,pos]=MASK;positions.append(pos)
            # Mask fraction equals one component / total components.
            sigma=-math.log(1-.999/(k+1));raw=model(x,q[None].expand(len(batch),-1),torch.full((len(batch),),sigma,device=q.device))
            for b,(i,_,_) in enumerate(batch):
                for l,pos in enumerate(positions[b]):
                    startid=model.offsets[l];size=model.code_sizes[l]
                    logp=raw[b,pos,startid:startid+size].log_softmax(-1)
                    scores[i]+=float(logp[clean[b,pos]-startid]);counts[i]+=1
    result=[]
    for i,p in enumerate(proposals):
        k=len(p['components'])-1;s=scores[i]/counts[i]+math.log(max(float(length_probs[k]),1e-30))
        result.append({**p,'pseudo_logp_per_code':scores[i]/counts[i],'length_logp':math.log(max(float(length_probs[k]),1e-30)),'ranking_score':s})
    return sorted(result,key=lambda p:(-p['ranking_score'],tuple(p['components'])))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(str(AC_ROOT)));p.add_argument('--data',type=Path,required=True);p.add_argument('--train',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=32);p.add_argument('--draws',type=int,default=64);p.add_argument('--seed',type=int,default=42);p.add_argument('--replay',action='store_true');p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    sys.path.insert(0,str(a.root));import baseline5_run_infer_rag_gpt as b5;b5.ensure_env_cuda_library()
    import torch
    from model import RVQDiffusion
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    cfg=json.loads((a.train/'config.json').read_text());assert not cfg['sanity']
    ready=json.loads((a.data/'ready.json').read_text());assert cfg['data_ready_sha256']==sha(a.data/'ready.json')
    for name,h in ready['file_hashes'].items():assert sha(a.data/name)==h,name
    for name,h in cfg['code_hashes'].items():assert sha(Path(__file__).parent/name)==h,name
    checkpoint=torch.load(a.train/'best.pt',map_location='cuda',weights_only=False)
    completed=json.loads((a.train/'completed.json').read_text());assert completed['best_sha256']==sha(a.train/'best.pt')
    audit=json.loads((a.data/'audit.json').read_text());catalog=json.loads((a.data/'catalog.json').read_text());inputs=json.loads((a.data/'test_inputs.json').read_text());queries=json.loads((a.data/'queries.json').read_text())
    if a.limit:inputs=inputs[:a.limit]
    npcodes=np.load(a.data/'codes.npy');codes=torch.tensor(npcodes,device='cuda');qfeatures=torch.tensor(np.load(a.data/'query_semantics.npy'),device='cuda')
    model=RVQDiffusion(np.load(a.data/'rvq_centroids.npy'),audit['codebook_sizes'],**cfg['architecture']).cuda();model.load_state_dict(checkpoint['model']);model.eval()
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='replay'}
    config.update({'protocol':'DDBC-Semantic-RVQ component construction; training catalog only; no retrieval snapshots','selected_step':checkpoint['step'],'checkpoint_sha256':sha(a.train/'best.pt'),'data_ready_sha256':sha(a.data/'ready.json'),'infer_code_sha256':sha(__file__),
        'ranking':'three permutations leave-one-component-out mean code logp + log predicted tool count; NOT exact likelihood','temperature':1.,'max_sampling_rounds':4,'gold_inference_access':False})
    out=a.output
    if a.replay:
        assert json.loads((out/'config.json').read_text())==config
        before=sha(out/'predictions_blind.jsonl')
    else:
        assert not out.exists(),'Choose new output directory';out.mkdir(parents=True);write(out/'config.json',config)
    predictions=[];start=time.time()
    with torch.inference_mode():
        for index,inp in enumerate(inputs):
            q=qfeatures[inp['query_index']];lp=model.length_logits(q[None]).softmax(-1)[0].cpu().numpy();unique={};failures=collections.Counter();drawn=0
            for roundno in range(4):
                seed=a.seed+10000+index*100+roundno;rng=np.random.default_rng(seed)
                lengths=rng.choice(7,a.draws,p=lp.astype(np.float64)/lp.sum(dtype=np.float64)).tolist()
                bundles,failed=draw_bundles(model,q,npcodes,catalog,lengths,a.steps,seed);failures.update(failed);drawn+=len(lengths)
                for b in bundles:
                    key=tuple(b['components'])
                    if key not in unique or b['reverse_path_logp']>unique[key]['reverse_path_logp']:unique[key]=b
                if len(unique)>=10:break
            if not unique:raise RuntimeError('No legal sampled configurations')
            ranked=rank_bundles(model,q,codes,list(unique.values()),lp);results=[]
            for b in ranked[:10]:
                llm=catalog[b['components'][0]];tools=[catalog[j] for j in b['components'][1:]];assert llm['kind']=='llm' and all(t['kind']=='tool' for t in tools)
                tokens=[t['token'] for t in tools];strict=' '.join([llm['token'],'<TOOL_SEP>']+(tokens or ['<TOOL_EMPTY>'])+['<SPECIAL_END>'])
                results.append({**b,'rank':len(results)+1,'llm_token':llm['token'],'tool_tokens':tokens,'strict_text':strict,'gen_text':strict})
            item={'sample_id':inp['sample_id'],'results':results,'length_probabilities':lp.tolist(),'sampled_draws':drawn,'unique_proposals':len(unique),'failures':dict(failures),'all_ranked_proposals':ranked}
            predictions.append(item);write(out/'blind_per_sample'/f"{inp['sample_id']}.json",item)
            print(f'{index+1}/{len(inputs)} unique={len(unique)} top10={len(results)} elapsed={time.time()-start:.1f}s',flush=True)
    (out/'predictions_blind.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions))
    # Only now read and join evaluation labels.
    assert sha(b5.DEFAULT_SOURCE/'sample_manifest.jsonl')==audit['test_manifest_sha256']
    manifest=[json.loads(s) for s in (b5.DEFAULT_SOURCE/'sample_manifest.jsonl').open()];byid={r['sample_id']:r for r in manifest}
    records=[{'ok':True,'baseline':'DDBC-Semantic-RVQ','dataset_example':byid[r['sample_id']],'query':byid[r['sample_id']]['query'],'results':r['results']} for r in predictions]
    (out/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    import baseline_top10_common as common
    common.evaluate(out,records,len(inputs))
    ev=b5.load_module(a.root/'evaluation/reference/exp3/evaluate_ranked_recall_baseline2.py','ddbc_semantic_ev')
    all_llms={c['token'] for c in catalog if c['kind']=='llm'};all_tools={c['token'] for c in catalog if c['kind']=='tool'}
    trainsets={tuple([r['components'][0]]+sorted(r['components'][1:])) for r in json.loads((a.data/'train.json').read_text())}
    extras=[]
    for rec in records:
        llm,ts=ev.get_gold(rec);gt=set(ts);pr=rec['results'][0];pt=set(pr['tool_tokens']);gc=gt|{llm};pc=pt|{pr['llm_token']}
        extras.append({'sample_id':rec['dataset_example']['sample_id'],'tool_jaccard':len(gt&pt)/len(gt|pt) if gt|pt else 1.,'component_jaccard':len(gc&pc)/len(gc|pc),
            'gold_exactly_constructible':llm in all_llms and gt<=all_tools and len(gt)<=6,
            'component_catalog_recall':(int(llm in all_llms)+len(gt&all_tools))/len(gc),
            'top1_new_combination':tuple(pr['components']) not in trainsets,
            'top10_new_combination_fraction':sum(tuple(p['components']) not in trainsets for p in rec['results'])/len(rec['results'])})
    write(out/'evaluation/additional.json',{'mean':{k:float(np.mean([r[k] for r in extras])) for k in extras[0] if k!='sample_id'},'per_sample':extras})
    write(out/'completed.json',{'predictions_sha256':sha(out/'predictions_blind.jsonl'),'results_sha256':sha(out/'results.jsonl'),'inference_seconds':time.time()-start,'n':len(inputs)})
    if a.replay:write(out/'replay_audit.json',{'byte_identical':before==sha(out/'predictions_blind.jsonl'),'before':before,'after':sha(out/'predictions_blind.jsonl')});assert before==sha(out/'predictions_blind.jsonl')
if __name__=='__main__':main()
