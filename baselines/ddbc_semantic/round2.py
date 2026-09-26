#!/usr/bin/env python3
"""Second optimization: exact feasibility lookahead and a whole-bundle critic.
All choices are made on validation; old test predictions are only read after
selection is frozen, and test labels only after all blind predictions are saved.
"""
import argparse,collections,json,math,sys,time
from pathlib import Path
import numpy as np
from optimize import ROOT,SemanticAdapter,log_probs,evaluate,write,sha,torch,b5,RVQDiffusion
from critic_v2 import BundleCritic,pad_components
from guided import draw_bundles as original_draw
from sampling_v2 import draw_bundles as matching_draw
from torch.nn import functional as F

MODES=['previous','matching_constant','matching_annealed','matching_annealed_lengthmix']
GAMMAS=[0.,.25,.5,1.]

@torch.no_grad()
def generate(model,q,codes,cat,prior,mode,index,seed,draws=64):
    lp=model.length_logits(q[None]).softmax(-1)[0].cpu().numpy();sampling_lp=lp.astype(np.float64)
    if mode.endswith('lengthmix'):
        uniform=np.array([0]+[1/6]*6);sampling_lp=.8*sampling_lp+.2*uniform
    sampling_lp/=sampling_lp.sum();unique={};stats=collections.Counter();drawn=0
    for rnd in range(4):
        sd=seed+10000+index*100+rnd;rng=np.random.default_rng(sd);lengths=rng.choice(7,draws,p=sampling_lp).tolist()
        if mode=='previous':bs,st=original_draw(model,q,codes,cat,lengths,32,sd,prior,1.)
        else:bs,st=matching_draw(model,q,codes,cat,lengths,32,sd,prior,'constant' if mode=='matching_constant' else 'annealed')
        stats.update(st);drawn+=len(lengths)
        for b in bs:
            key=tuple(b['components'])
            if key not in unique or b['reverse_path_logp']>unique[key]['reverse_path_logp']:unique[key]=b
        if len(unique)>=10 or draws<64:break
    assert unique
    if draws==64:assert len(unique)>=10
    return {'all_ranked_proposals':list(unique.values()),'length_probabilities':lp.tolist(),'sampling_length_probabilities':sampling_lp.tolist(),'sampled_draws':drawn,'unique_proposals':len(unique),'failures':{k:v for k,v in stats.items() if k=='constraint_dead_end'},'sampler_diagnostics':dict(stats)}

def coverage_labels(proposals,rows):
    truth=[(set(c for c in r['components'] if c>=0),len(r['components']),-1 not in r['components']) for r in rows]
    labels=[]
    for p in proposals:
        predicted=set(p['components']);labels.append([sum(len(predicted&s)/n for s,n,_ in truth)/len(truth),sum(known and s<=predicted for s,_,known in truth)/len(truth)])
    return np.array(labels,np.float32)

def semantic_gain(props,prior,lp):
    w=sum(p/(k+1) for k,p in enumerate(lp))
    return np.array([w*prior[p['components'][0]]+(1-w)*sum(prior[i] for i in p['components'][1:]) for p in props])

def rank_items(props,prior,lp,critic_scores,gamma):
    base=semantic_gain(props,prior,lp)
    def z(v):return (v-v.mean())/max(float(v.std()),1e-8)
    score=(1-gamma)*z(base)+gamma*z(np.asarray(critic_scores))
    result=[{**p,'semantic_expected_coverage':float(base[i]),'critic_expected_component_recall':float(critic_scores[i]),'ranking_score':float(score[i])} for i,p in enumerate(props)]
    return sorted(result,key=lambda x:(-x['ranking_score'],tuple(x['components'])))

@torch.no_grad()
def critic_scores(net,q,c,probs,qid,props):
    net.eval();result=[]
    for ix in range(0,len(props),256):
        bb=props[ix:ix+256];ids=pad_components([p['components'] for p in bb]);logits=net(q[qid:qid+1].expand(len(bb),-1),c,ids,probs[qid:qid+1].expand(len(bb),-1));result.extend(logits[:,0].sigmoid().cpu().tolist())
    return result

def to_prediction(sample_id,item,ranked,cat):
    results=[]
    for p in ranked[:10]:
        ids=p['components'];llm=cat[ids[0]]['token'];tokens=[cat[i]['token'] for i in ids[1:]];strict=' '.join([llm,'<TOOL_SEP>']+(tokens or ['<TOOL_EMPTY>'])+['<SPECIAL_END>'])
        results.append({**p,'rank':len(results)+1,'llm_token':llm,'tool_tokens':tokens,'strict_text':strict,'gen_text':strict})
    return {'sample_id':sample_id,'results':results,'all_ranked_proposals':ranked,**{k:item[k] for k in ['length_probabilities','sampled_draws','unique_proposals','failures']},'sampler_diagnostics':item.get('sampler_diagnostics',{})}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--denoiser',type=Path,required=True);ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--epochs',type=int,default=20);a=ap.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True);assert not (out/'config.json').exists(),'Choose a new output directory'
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    ready=json.loads((a.data/'ready.json').read_text())
    for f,h in ready['file_hashes'].items():assert sha(a.data/f)==h
    oldcfg=json.loads((a.previous/'config.json').read_text());oldsel=json.loads((a.previous/'selection_frozen.json').read_text());assert oldsel['overall']['beta']==1 and oldsel['overall']['alpha']==1
    assert sha(a.previous/'semantic_best.pt')==oldsel['semantic_checkpoint_sha256'];assert sha(a.denoiser/'best.pt')==oldcfg['denoiser_sha256']
    for f,h in oldcfg['code_sha256'].items():assert sha(Path(__file__).parent/f)==h
    cat=json.loads((a.data/'catalog.json').read_text());nllm=sum(x['kind']=='llm' for x in cat);tr=json.loads((a.data/'train.json').read_text());va=json.loads((a.data/'validation.json').read_text())
    tg=collections.defaultdict(list);vg=collections.defaultdict(list)
    for r in tr:tg[r['query_index']].append(r)
    for r in va:vg[r['query_index']].append(r)
    assert not set(tg)&set(vg)
    q=torch.tensor(np.load(a.data/'query_semantics.npy'),device='cuda');c=torch.tensor(np.load(a.data/'component_semantics.npy'),device='cuda');codes=np.load(a.data/'codes.npy');audit=json.loads((a.data/'audit.json').read_text());dc=json.loads((a.denoiser/'config.json').read_text())
    den=RVQDiffusion(np.load(a.data/'rvq_centroids.npy'),audit['codebook_sizes'],**dc['architecture']).cuda();den.load_state_dict(torch.load(a.denoiser/'best.pt',map_location='cuda',weights_only=False)['model']);den.eval()
    sem=SemanticAdapter(len(cat)).cuda();sem.load_state_dict(torch.load(a.previous/'semantic_best.pt',map_location='cuda',weights_only=False)['state_dict']);sem.eval()
    with torch.no_grad():
        probs=torch.cat([log_probs(sem(q[i:i+64],c,nllm),nllm).exp() for i in range(0,len(q),64)],0)
        lp_all=den.length_logits(q).softmax(-1).cpu().numpy()
    priors=probs.cpu().numpy();critic=BundleCritic().cuda();parameter_count=sum(p.numel() for p in critic.parameters())
    valids=json.loads((a.previous/'validation_subset.json').read_text())['query_indices'];valrows=[r for r in va if r['query_index'] in set(valids)]
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()};config.update({'modes':MODES,'critic_blend_grid':GAMMAS,'draws':64,'diffusion_steps':32,'critic_parameters':parameter_count,'critic_train_queries':len(tg),'critic_lr':3e-4,'critic_weight_decay':.01,'critic_batch_queries':32,'critic_candidates_per_query':16,'critic_checkpoint_selection':'minimum MSE on previous validation proposals; labels averaged over each query targets','final_selection':'maximum validation RDCR@10; old baseline eligible, simpler wins ties','validation_query_indices':valids,'fresh_confirmation_queries':128,'confirmation_rule':'compare discovery-selected configuration against previous baseline on disjoint validation queries; keep new only if RDCR improves','test_labels_before_predictions':False,'data_sha256':sha(a.data/'ready.json'),'denoiser_sha256':sha(a.denoiser/'best.pt'),'semantic_sha256':sha(a.previous/'semantic_best.pt'),'code_sha256':{f:sha(Path(__file__).parent/f) for f in ['round2.py','sampling_v2.py','critic_v2.py','optimize.py','guided.py','model.py']}})
    write(out/'config.json',config);start=time.time();rng=np.random.default_rng(100042)
    # Train-only synthetic negatives and positives; never used as inference proposals.
    pool={};alltrain=sorted(tg)
    for qid in alltrain:
        rows=tg[qid];prior=priors[qid];tools=np.arange(nllm,len(cat));tp=prior[nllm:].astype(float);tp/=tp.sum();llp=prior[:nllm].astype(float);llp/=llp.sum();props={}
        for j in range(16):
            r=rows[int(rng.integers(len(rows)))];gold=r['components'];llm=gold[0] if j<6 or j>=13 else int(rng.choice(nllm,p=llp));selected=list(gold[1:])
            if 2<=j<6:
                selected=[t for t in selected if rng.random()<.5]
                nadd=int(rng.integers(0,4))
                selected+=rng.choice(tools,nadd,replace=False,p=tp).tolist()
            elif 6<=j<11:
                k=int(rng.choice(7,p=lp_all[qid].astype(float)/lp_all[qid].sum(dtype=float)));temp=[.6,1.,2.][j%3];p2=tp**(1/temp);p2/=p2.sum();selected=rng.choice(tools,k,replace=False,p=p2).tolist()
            elif 11<=j<13:
                other=tg[alltrain[int(rng.integers(len(alltrain)))]];selected=other[int(rng.integers(len(other)))]['components'][1:]
            elif j>=13:
                selected+=rng.choice(tools,int(rng.integers(1,4)),replace=False,p=tp).tolist()
            selected=list(dict.fromkeys(selected))[:6];ids=[int(llm)]+sorted(int(i) for i in selected);props[tuple(ids)]={'components':ids}
        pool[qid]=list(props.values())
    real_ids=rng.choice(alltrain,96,replace=False).tolist();write(out/'real_negative_queries.json',real_ids)
    for index,qid in enumerate(real_ids):
        item=generate(den,q[qid],codes,cat,priors[qid],'previous',index,12042,draws=16);pool[qid].extend(item['all_ranked_proposals']);write(out/'training_diffusion_proposals'/f'{qid}.json',item)
        if index%16==0:print(f'train diffusion negatives {index+1}/96 elapsed={time.time()-start:.1f}',flush=True)
    trainids=[];trainqid=[];trainlabels=[];ranges={}
    for qid in alltrain:
        ps=list({tuple(p['components']):p for p in pool[qid]}.values());begin=len(trainids);trainids.extend(p['components'] for p in ps);trainqid.extend([qid]*len(ps));trainlabels.extend(coverage_labels(ps,tg[qid]));ranges[qid]=(begin,len(trainids))
    ids_tensor=pad_components(trainids);labels_tensor=torch.tensor(np.array(trainlabels),device='cuda');qids_tensor=torch.tensor(trainqid,device='cuda')
    write(out/'training_data_audit.json',{'candidate_count':len(trainids),'query_count':len(alltrain),'diffusion_query_count':len(real_ids),'all_queries_train_only':set(trainqid)<=set(tg),'maximum_tools':max(len(x)-1 for x in trainids),'candidate_tensors_sha256_source':'data/checkpoint hashes and fixed synthesis seed in source'})
    valcache={qid:json.loads((a.previous/'validation_proposals/beta_1.0'/f'{qid}.json').read_text()) for qid in valids}
    vprops=[];vqueries=[];vlabels=[]
    for qid,item in valcache.items():
        ps=item['all_ranked_proposals'];vprops.extend(p['components'] for p in ps);vqueries.extend([qid]*len(ps));vlabels.extend(coverage_labels(ps,vg[qid]))
    vids=pad_components(vprops);vqids=torch.tensor(vqueries,device='cuda');vlabels=torch.tensor(np.array(vlabels),device='cuda')
    opt=torch.optim.AdamW(critic.parameters(),lr=3e-4,weight_decay=.01);best=float('inf');bad=0;history=[]
    @torch.no_grad()
    def val_mse():
        critic.eval();total=0
        for ix in range(0,len(vids),256):
            iq=vqids[ix:ix+256];pred=critic(q[iq],c,vids[ix:ix+256],probs[iq])[:,0].sigmoid();total+=float(((pred-vlabels[ix:ix+256,0])**2).sum())
        return total/len(vids)
    write(out/'initial_validation.json',{'critic_mse':val_mse()})
    for epoch in range(1,a.epochs+1):
        critic.train();order=rng.permutation(alltrain);total=0;nb=0
        for ix in range(0,len(order),32):
            which=[]
            for qid in order[ix:ix+32]:
                lo,hi=ranges[int(qid)];which.extend(rng.integers(lo,hi,size=16).tolist())
            ii=torch.tensor(which,device='cuda');iq=qids_tensor[ii];logits=critic(q[iq],c,ids_tensor[ii],probs[iq]);target=labels_tensor[ii];pred=logits[:,0].sigmoid()
            mse=F.mse_loss(pred,target[:,0]);aux=F.binary_cross_entropy_with_logits(logits[:,1],target[:,1]);z=logits[:,0].view(-1,16);y=target[:,0].view(-1,16);delta=y[:,:8]-y[:,8:];pair=(F.softplus(-delta.sign()*(z[:,:8]-z[:,8:]))*delta.abs()).mean();loss=mse+.1*aux+.1*pair
            assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.);opt.step();total+=float(loss);nb+=1
        v=val_mse();r={'epoch':epoch,'train_loss':total/nb,'validation_mse':v,'elapsed_seconds':time.time()-start};history.append(r);write(out/'critic_history.json',history);print(json.dumps(r),flush=True)
        if v<best-1e-7:best=v;bad=0;torch.save({'state_dict':critic.state_dict(),'epoch':epoch,'validation_mse':v},out/'critic_best.pt')
        else:bad+=1
        if bad>=4:break
    ck=torch.load(out/'critic_best.pt',map_location='cuda',weights_only=False);critic.load_state_dict(ck['state_dict']);critic.eval();grid=[]
    def grid_for_cache(cache,rows,mode,gammas):
        scores={qid:critic_scores(critic,q,c,probs,qid,item['all_ranked_proposals']) for qid,item in cache.items()};ret=[]
        for gamma in gammas:
            ranked={qid:rank_items(item['all_ranked_proposals'],priors[qid],item['length_probabilities'],scores[qid],gamma) for qid,item in cache.items()}
            result={'mode':mode,'gamma':gamma,'metrics':evaluate(ranked,rows,cat)};ret.append(result);print(json.dumps(result),flush=True)
        return ret
    for mode in MODES:
        if mode=='previous':cache=valcache
        else:
            cache={}
            for index,qid in enumerate(valids):
                item=generate(den,q[qid],codes,cat,priors[qid],mode,index,a.seed);cache[qid]=item;write(out/'validation_proposals'/mode/f'{qid}.json',item)
                if index%32==0:print(f'validation {mode} {index+1}/{len(valids)} elapsed={time.time()-start:.1f}',flush=True)
        grid.extend(grid_for_cache(cache,valrows,mode,GAMMAS));write(out/'validation_grid.json',grid)
    choose=lambda rr:max(rr,key=lambda r:(r['metrics']['rdcr@10'],-MODES.index(r['mode']),-r['gamma']))
    selected=choose(grid);critic_only=choose([r for r in grid if r['mode']=='previous']);sampling_only=choose([r for r in grid if r['gamma']==0]);baseline=next(r for r in grid if r['mode']=='previous' and r['gamma']==0)
    # Fresh validation queries were not used to fit/choose critic or discovery grid.
    remaining=sorted(set(vg)-set(valids));crng=np.random.default_rng(20260917);crng.shuffle(remaining);confirmids=sorted(remaining[:128]);confirmrows=[r for r in va if r['query_index'] in set(confirmids)]
    write(out/'confirmation_subset.json',{'query_indices':confirmids,'rows':len(confirmrows),'disjoint_from_discovery':not set(confirmids)&set(valids)})
    confirmations=[]
    for variant in [baseline,selected] if (selected['mode'],selected['gamma'])!=('previous',0.) else [baseline]:
        mode=variant['mode'];cache={}
        for index,qid in enumerate(confirmids):
            item=generate(den,q[qid],codes,cat,priors[qid],mode,index,43);cache[qid]=item;write(out/'confirmation_proposals'/mode/f'{qid}.json',item)
            if index%32==0:print(f'confirmation {mode} {index+1}/{len(confirmids)} elapsed={time.time()-start:.1f}',flush=True)
        confirmations.extend(grid_for_cache(cache,confirmrows,mode,[variant['gamma']]))
    accepted=len(confirmations)>1 and confirmations[1]['metrics']['rdcr@10']>confirmations[0]['metrics']['rdcr@10']
    final=selected if accepted else baseline
    selection={'discovery_selected':selected,'confirmation':confirmations,'confirmation_accepted':accepted,'final':final,'critic_only':critic_only,'sampling_only':sampling_only,'critic_epoch':ck['epoch'],'critic_sha256':sha(out/'critic_best.pt'),'validation_grid_sha256':sha(out/'validation_grid.json'),'frozen_before_test':True}
    write(out/'selection_frozen.json',selection);print('SELECTION '+json.dumps(selection),flush=True)
    inputs=json.loads((a.data/'test_inputs.json').read_text());assert not {r['query_index'] for r in inputs}&(set(tg)|set(vg));old={r['sample_id']:r for r in (json.loads(x) for x in (a.previous/'guided_selected/predictions_blind.jsonl').open())}
    variants={'critic_only':critic_only,'sampling_only':sampling_only,'selected':final};predictions={name:[] for name in variants}
    with torch.inference_mode():
        for index,inp in enumerate(inputs):
            qid=inp['query_index'];cache={'previous':old[inp['sample_id']]};sc={}
            for mode in dict.fromkeys(v['mode'] for v in variants.values()):
                if mode!='previous':cache[mode]=generate(den,q[qid],codes,cat,priors[qid],mode,index,a.seed)
                sc[mode]=critic_scores(critic,q,c,probs,qid,cache[mode]['all_ranked_proposals'])
            for name,variant in variants.items():
                item=cache[variant['mode']];ranked=rank_items(item['all_ranked_proposals'],priors[qid],item['length_probabilities'],sc[variant['mode']],variant['gamma']);predictions[name].append(to_prediction(inp['sample_id'],item,ranked,cat))
            if index%20==0:print(f'test inference {index+1}/{len(inputs)} elapsed={time.time()-start:.1f}',flush=True)
    hashes={}
    for name,pp in predictions.items():
        dest=out/name;dest.mkdir();(dest/'predictions_blind.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in pp));hashes[name]=sha(dest/'predictions_blind.jsonl')
    write(out/'blind_predictions_complete.json',hashes)
    # First test target access, after all blind predictions have been written.
    assert sha(b5.DEFAULT_SOURCE/'sample_manifest.jsonl')==audit['test_manifest_sha256'];manifest={r['sample_id']:r for r in (json.loads(x) for x in b5.DEFAULT_SOURCE.joinpath('sample_manifest.jsonl').open())}
    import baseline_top10_common as common
    reports={}
    for name,pp in predictions.items():
        dest=out/name;records=[{'ok':True,'baseline':'DDBC-Adapt-v2-'+name,'dataset_example':manifest[r['sample_id']],'query':manifest[r['sample_id']]['query'],'results':r['results']} for r in pp]
        (dest/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));reports[name]=common.evaluate(dest,records,len(inputs))
        independent=evaluate({i:r['results'] for i,r in enumerate(pp)},[{'query_index':i,'target':manifest[r['sample_id']]['target']} for i,r in enumerate(pp)],cat);assert all(abs(v-reports[name]['mean'][k])<1e-12 for k,v in independent.items())
        for r in pp:
            assert len(r['results'])==10 and len({tuple(z['components']) for z in r['results']})==10
            for z in r['all_ranked_proposals']:
                ids=z['components'];assert cat[ids[0]]['kind']=='llm';assert len(ids)<=7 and len(ids)==len(set(ids));assert all(cat[i]['kind']=='tool' for i in ids[1:])
        write(dest/'audit.json',{'all_outputs_legal':True,'all_top10_unique':True,'independent_nine_metrics_match':True,'predictions_sha256':hashes[name],'results_sha256':sha(dest/'results.jsonl')})
    write(out/'completed.json',{'elapsed_seconds':time.time()-start,'selection':selection,'metrics':reports,'critic_parameters':parameter_count})

if __name__=='__main__':main()
