#!/usr/bin/env python3
"""Train-only semantic adapters; validation-selected ranking and diffusion guidance.

Never reads test targets until all selected test variants are saved. Does not alter
RVQ, the denoiser, the component vocabulary, or the original result files.
"""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse, collections, json, math, os, re, sys, time
from pathlib import Path
import numpy as np
from prepare import write, sha

ROOT=Path(str(AC_ROOT))
sys.path.insert(0,str(ROOT))
import baseline5_run_infer_rag_gpt as b5
b5.ensure_env_cuda_library()
import torch
from torch import nn
from torch.nn import functional as F
from model import RVQDiffusion
from infer import draw_bundles, rank_bundles
from guided import draw_bundles as guided_draw

class SemanticAdapter(nn.Module):
    def __init__(self,n):
        super().__init__()
        def adapter():
            m=nn.Sequential(nn.Linear(768,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,768))
            nn.init.zeros_(m[-1].weight);nn.init.zeros_(m[-1].bias)
            return m
        self.q_llm=adapter();self.q_tool=adapter();self.component=adapter()
        self.log_scale=nn.Parameter(torch.tensor([math.log(10.),math.log(10.)]))
        self.bias=nn.Parameter(torch.zeros(n))
    def forward(self,q,c,nllm):
        cc=F.normalize(c+self.component(c),dim=-1)
        ql=F.normalize(q+self.q_llm(q),dim=-1);qt=F.normalize(q+self.q_tool(q),dim=-1)
        scales=self.log_scale.exp().clamp(1,100)
        return torch.cat([scales[0]*(ql@cc[:nllm].T),scales[1]*(qt@cc[nllm:].T)],-1)+self.bias

def log_probs(logits,nllm):
    return torch.cat([logits[:,:nllm].log_softmax(-1),logits[:,nllm:].log_softmax(-1)],-1)

def targets(rows,n):
    y=np.zeros((len(rows),n),np.float32)
    for i,r in enumerate(rows):
        for c in set(r['components']):
            if c>=0:y[i,c]=1/len(r['components'])
    return torch.tensor(y,device='cuda')

def gold_tokens(r):
    text=r['target'].split('<SPECIAL_END>')[0].split('Explanation:')[0]
    llm=re.findall(r'<LLM_[^<>\n\r]+>',text)[0]
    ts=set(re.findall(r'<<[^<>\n\r]+>>|<TOOL_[^<>\n\r]+>',text))-{'<TOOL_SEP>','<TOOL_EMPTY>'}
    return llm,ts

def evaluate(proposals,rows,catalog):
    vals=[];weights=np.array([1/math.log2(i+2) for i in range(10)])
    for r in rows:
        ranked=proposals[r['query_index']];llm,gt=gold_tokens(r);gc=gt|{llm};cr=[];th=[];ch=[]
        for z in ranked[:10]:
            cs=z['components'];pl=catalog[cs[0]]['token'];pt={catalog[i]['token'] for i in cs[1:]}
            cr.append(len(gc&(pt|{pl}))/len(gc));th.append(gt<=pt);ch.append(pl==llm and gt<=pt)
        first=ranked[0]['components'];pt={catalog[i]['token'] for i in first[1:]};hit=len(gt&pt)
        rec=hit/len(gt) if gt else 1.;prec=hit/len(pt) if pt else float(not gt)
        vals.append({'top1_tool_recall':rec,'top1_component_recall':cr[0],'tool_hit@10':float(any(th)),'cr_hit@1':float(ch[0]),'cr_hit@10':float(any(ch)),
            'cr_mrr@10':next((1/(i+1) for i,v in enumerate(ch) if v),0.),'rdcr@10':sum(w*c for w,c in zip(weights,cr))/sum(weights),
            'top1_tool_precision':prec,'top1_tool_f1':2*rec*prec/(rec+prec) if rec+prec else 0.})
    return {k:float(np.mean([v[k] for v in vals])) for k in vals[0]}

def rerank(proposals,prob,length_probs,alpha):
    # Expected required-component coverage: independent marginal probabilities.
    # This scorer does not claim to model interactions; the denoiser does that.
    llm_weight=float(sum(p/(k+1) for k,p in enumerate(length_probs)))
    gain=np.array([llm_weight*prob[p['components'][0]]+(1-llm_weight)*sum(prob[c] for c in p['components'][1:]) for p in proposals])
    base=np.array([p['ranking_score'] for p in proposals])
    def z(x):return (x-x.mean())/max(float(x.std()),1e-8)
    score=base if alpha==0 else (1-alpha)*z(base)+alpha*z(gain)
    result=[{**p,'diffusion_ranking_score':float(base[i]),'semantic_expected_coverage':float(gain[i]),'ranking_score':float(score[i])} for i,p in enumerate(proposals)]
    return sorted(result,key=lambda p:(-p['ranking_score'],tuple(p['components'])))

@torch.no_grad()
def generate(model,q,codes,catalog,prior,beta,index,seed):
    lp=model.length_logits(q[None]).softmax(-1)[0].cpu().numpy();unique={};failed=collections.Counter();drawn=0
    for rnd in range(4):
        sd=seed+10000+index*100+rnd;rng=np.random.default_rng(sd)
        lengths=rng.choice(7,64,p=lp.astype(np.float64)/lp.sum(dtype=np.float64)).tolist()
        bs,fs=guided_draw(model,q,codes,catalog,lengths,32,sd,component_prior=prior,beta=beta)
        failed.update(fs);drawn+=len(lengths)
        for b in bs:
            key=tuple(b['components'])
            if key not in unique or b['reverse_path_logp']>unique[key]['reverse_path_logp']:unique[key]=b
        if len(unique)>=10:break
    assert len(unique)>=10,'Fewer than 10 unique bundles'
    ranked=rank_bundles(model,q,torch.tensor(codes,device='cuda'),list(unique.values()),lp)
    return {'length_probabilities':lp.tolist(),'all_ranked_proposals':ranked,'sampled_draws':drawn,'failures':dict(failed),'unique_proposals':len(unique)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--denoiser',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--validation-queries',type=int,default=128);p.add_argument('--epochs',type=int,default=30);a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'config.json').exists(),'Use a fresh directory'
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    ready=json.loads((a.data/'ready.json').read_text())
    for name,h in ready['file_hashes'].items():assert sha(a.data/name)==h
    tr=json.loads((a.data/'train.json').read_text());va=json.loads((a.data/'validation.json').read_text());cat=json.loads((a.data/'catalog.json').read_text())
    assert not {r['query_index'] for r in tr}&{r['query_index'] for r in va}
    nllm=sum(c['kind']=='llm' for c in cat);assert all(c['kind']=='llm' for c in cat[:nllm])
    q=torch.tensor(np.load(a.data/'query_semantics.npy'),device='cuda');c=torch.tensor(np.load(a.data/'component_semantics.npy'),device='cuda')
    codes=np.load(a.data/'codes.npy');audit=json.loads((a.data/'audit.json').read_text());cfg=json.loads((a.denoiser/'config.json').read_text())
    den=RVQDiffusion(np.load(a.data/'rvq_centroids.npy'),audit['codebook_sizes'],**cfg['architecture']).cuda()
    den.load_state_dict(torch.load(a.denoiser/'best.pt',map_location='cuda',weights_only=False)['model']);den.eval()
    net=SemanticAdapter(len(cat)).cuda();opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.01)
    ty=targets(tr,len(cat));vy=targets(va,len(cat));tq=torch.tensor([r['query_index'] for r in tr],device='cuda');vq=torch.tensor([r['query_index'] for r in va],device='cuda')
    conf={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    conf.update({'trainable_scorer_parameters':sum(p.numel() for p in net.parameters()),'scorer':'residual semantic adapters, type-normalized weighted positive cross entropy','selection':'minimum full validation weighted CE; rank/guidance chosen by validation RDCR@10','train_queries':len(set(tq.tolist())),'train_rows':len(tr),'validation_rows':len(va),'guidance_grid':[0,.5,1.],'ranking_alpha_grid':[0,.25,.5,.75,1.],
      'lr':3e-4,'weight_decay':.01,'batch_size':128,'max_epochs':a.epochs,'early_stop_patience':5,'seed':a.seed,'steps':32,'draws':64,'data_sha256':sha(a.data/'ready.json'),'denoiser_sha256':sha(a.denoiser/'best.pt'),'code_sha256':{f:sha(Path(__file__).parent/f) for f in ['optimize.py','guided.py','infer.py','model.py']},'test_gold_access_before_predictions':False})
    write(out/'config.json',conf);history=[];best=float('inf');bad=0;start=time.time()
    @torch.no_grad()
    def val_loss():
        net.eval();total=0
        for ix in range(0,len(va),128):total+=float(-(vy[ix:ix+128]*log_probs(net(q[vq[ix:ix+128]],c,nllm),nllm)).sum())
        return total/len(va)
    init=val_loss();write(out/'initial_validation.json',{'weighted_ce':init})
    for epoch in range(1,a.epochs+1):
        net.train();order=torch.randperm(len(tr),device='cuda');total=0
        for ix in range(0,len(tr),128):
            ids=order[ix:ix+128];loss=-(ty[ids]*log_probs(net(q[tq[ids]],c,nllm),nllm)).sum(-1).mean()
            assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();total+=float(loss)*len(ids)
        vl=val_loss();record={'epoch':epoch,'train_ce':total/len(tr),'validation_ce':vl,'elapsed_seconds':time.time()-start};history.append(record);print(json.dumps(record),flush=True)
        if vl<best-1e-6:
            best=vl;bad=0;torch.save({'state_dict':net.state_dict(),'epoch':epoch,'validation_ce':vl},out/'semantic_best.pt')
        else:bad+=1
        write(out/'training_history.json',history)
        if bad>=5:break
    ck=torch.load(out/'semantic_best.pt',map_location='cuda',weights_only=False);net.load_state_dict(ck['state_dict']);net.eval()
    @torch.no_grad()
    def probability(qid):return log_probs(net(q[qid:qid+1],c,nllm),nllm)[0].exp().cpu().numpy()
    val_ids=sorted({r['query_index'] for r in va});rng=np.random.default_rng(20260916);rng.shuffle(val_ids);val_ids=sorted(val_ids[:a.validation_queries]);val_rows=[r for r in va if r['query_index'] in set(val_ids)]
    write(out/'validation_subset.json',{'query_indices':val_ids,'n_rows':len(val_rows),'selection':'uniform query sample seed 20260916, includes unrepresentable targets'})
    probs={qid:probability(qid) for qid in val_ids};cached={};grid=[]
    for beta in conf['guidance_grid']:
        cache={}
        for index,qid in enumerate(val_ids):
            item=generate(den,q[qid],codes,cat,probs[qid],beta,index,a.seed);cache[qid]=item
            write(out/'validation_proposals'/f'beta_{beta}'/f'{qid}.json',item)
            if index%16==0:print(f'validation beta={beta} query={index+1}/{len(val_ids)} elapsed={time.time()-start:.1f}',flush=True)
        cached[beta]=cache
        for alpha in conf['ranking_alpha_grid']:
            ranked={qid:rerank(cache[qid]['all_ranked_proposals'],probs[qid],cache[qid]['length_probabilities'],alpha) for qid in val_ids}
            entry={'beta':beta,'alpha':alpha,'metrics':evaluate(ranked,val_rows,cat)};grid.append(entry);print(json.dumps(entry),flush=True)
        write(out/'validation_grid.json',grid)
    # Prefer simpler variants on exact ties; the no-change baseline is eligible.
    choose=lambda rr:max(rr,key=lambda r:(r['metrics']['rdcr@10'],-r['beta'],-r['alpha']))
    rank_selection=choose([r for r in grid if r['beta']==0]);selection=choose(grid)
    frozen={'rerank_only':rank_selection,'overall':selection,'semantic_checkpoint_epoch':ck['epoch'],'semantic_checkpoint_sha256':sha(out/'semantic_best.pt'),'validation_grid_sha256':sha(out/'validation_grid.json'),'frozen_before_test_inference':True}
    write(out/'selection_frozen.json',frozen);print('SELECTION '+json.dumps(frozen),flush=True)
    # Test inputs contain query indices and IDs, never targets. Reuse original
    # blind proposals for the ranking ablation; selected guidance generates anew.
    inputs=json.loads((a.data/'test_inputs.json').read_text());assert not {r['query_index'] for r in inputs}&({r['query_index'] for r in tr}|{r['query_index'] for r in va})
    old={r['sample_id']:r for r in (json.loads(x) for x in (a.original/'predictions_blind.jsonl').open())}
    variants={'rerank_only':rank_selection,'guided_selected':selection};predictions={name:[] for name in variants}
    with torch.inference_mode():
        for index,inp in enumerate(inputs):
            qid=inp['query_index'];prior=probability(qid)
            guided=None
            if selection['beta']!=0:guided=generate(den,q[qid],codes,cat,prior,selection['beta'],index,a.seed)
            for name,s in variants.items():
                item=old[inp['sample_id']] if s['beta']==0 else guided
                ranked=rerank(item['all_ranked_proposals'],prior,item['length_probabilities'],s['alpha']);results=[]
                for z in ranked[:10]:
                    llm=cat[z['components'][0]]['token'];tokens=[cat[i]['token'] for i in z['components'][1:]]
                    strict=' '.join([llm,'<TOOL_SEP>']+(tokens or ['<TOOL_EMPTY>'])+['<SPECIAL_END>'])
                    results.append({**z,'rank':len(results)+1,'llm_token':llm,'tool_tokens':tokens,'strict_text':strict,'gen_text':strict})
                predictions[name].append({'sample_id':inp['sample_id'],'results':results,'all_ranked_proposals':ranked,'sampled_draws':item['sampled_draws'],'unique_proposals':item['unique_proposals'],'failures':item['failures'],'length_probabilities':item['length_probabilities']})
            if index%20==0:print(f'test inference {index+1}/{len(inputs)}',flush=True)
    hashes={}
    for name,pp in predictions.items():
        dest=out/name;dest.mkdir();(dest/'predictions_blind.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pp));hashes[name]=sha(dest/'predictions_blind.jsonl')
    write(out/'blind_predictions_complete.json',hashes)
    # First test target access, after both fixed variants are written.
    assert sha(b5.DEFAULT_SOURCE/'sample_manifest.jsonl')==audit['test_manifest_sha256']
    manifest={r['sample_id']:r for r in (json.loads(x) for x in (b5.DEFAULT_SOURCE/'sample_manifest.jsonl').open())}
    import baseline_top10_common as common
    metrics={}
    for name,pp in predictions.items():
        records=[{'ok':True,'baseline':'DDBC-Semantic-'+name,'dataset_example':manifest[r['sample_id']],'query':manifest[r['sample_id']]['query'],'results':r['results']} for r in pp]
        dest=out/name;(dest/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));metrics[name]=common.evaluate(dest,records,len(inputs))
        independent=evaluate({i:r['results'] for i,r in enumerate(pp)},[{'query_index':i,'target':manifest[r['sample_id']]['target']} for i,r in enumerate(pp)],cat)
        assert all(abs(independent[k]-metrics[name]['mean'][k])<1e-12 for k in independent)
        for r in pp:
            assert len(r['results'])==10;assert len({tuple(z['components']) for z in r['results']})==10
            for z in r['results']:
                ids=z['components'];assert cat[ids[0]]['kind']=='llm';assert all(cat[i]['kind']=='tool' for i in ids[1:]);assert len(ids)==len(set(ids));assert len(ids)<=7
        write(dest/'audit.json',{'independent_nine_metrics_match':True,'all_outputs_legal':True,'predictions_sha256':hashes[name],'results_sha256':sha(dest/'results.jsonl'),'selection_sha256':sha(out/'selection_frozen.json')})
    write(out/'completed.json',{'elapsed_seconds':time.time()-start,'selected':frozen,'metrics':metrics,'scorer_trainable_parameters':conf['trainable_scorer_parameters']})

if __name__=='__main__':main()
