#!/usr/bin/env python3
"""Training-only fixed-agent-catalog masked diffusion. See README.md for scope."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse, collections, hashlib, importlib.util, json, math, os, random, sys, time
from pathlib import Path
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def write(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m; spec.loader.exec_module(m); return m


def prepare(root,out):
    import baseline5_run_infer_rag_gpt as b5
    ev=load_module(root/'evaluation/reference/exp3/evaluate_ranked_recall_baseline2.py','ddbc_ev')
    td=root/'outputs/text2bundle_top10_disjoint_train_seed42'
    source=td/'training_excluding_table1.jsonl'
    raw=[json.loads(s) for s in source.open()]
    manifest,refs=b5.reference_inputs(b5.DEFAULT_SOURCE)
    tq={r['query'].strip() for r in manifest}; ti={str(r['qid']) for r in manifest}
    assert not any(r['query'].strip() in tq or str(r['qid']) in ti for r in raw)
    trids=set((td/'train_qids.txt').read_text().splitlines()); vaids=set((td/'val_qids.txt').read_text().splitlines())
    assert not trids&vaids
    train=[r for r in raw if str(r['qid']) in trids]; val=[r for r in raw if str(r['qid']) in vaids]
    assert len(train)+len(val)==len(raw)
    assert not {r['query'].strip() for r in train}&{r['query'].strip() for r in val}
    def key(r):
        llm,ts=ev.parse_agent_text(r['target'])
        if not llm: raise ValueError('Invalid training target')
        return (llm,tuple(sorted(set(ts))))
    keys=sorted({key(r) for r in train}); ids={k:i for i,k in enumerate(keys)}
    groups=collections.defaultdict(set)
    for r in train: groups[r['query'].strip()].add(ids[key(r)])
    queries=list(groups); bundles=[sorted(groups[q]) for q in queries]
    catalog=[{'id':i,'llm_token':k[0],'tool_tokens':list(k[1])} for i,k in enumerate(keys)]
    write(out/'catalog.json',catalog)
    write(out/'train_bundles.json',[{'query':q,'agent_ids':b} for q,b in zip(queries,bundles)])
    write(out/'split_audit.json',{'source_sha256':digest(source),'test_manifest_sha256':digest(b5.DEFAULT_SOURCE/'sample_manifest.jsonl'),
        'train_qids_sha256':digest(td/'train_qids.txt'),'val_qids_sha256':digest(td/'val_qids.txt'),
        'train_rows':len(train),'validation_rows':len(val),'train_queries':len(queries),'catalog_size':len(keys),
        'bundle_sizes':dict(collections.Counter(map(len,bundles))),'train_test_qid_overlap':0,'train_test_query_overlap':0,
        'train_validation_query_overlap':0,'validation_rows_in_catalog':sum(key(r) in ids for r in val),
        'catalog_source':'training targets only; exact LLM + unordered exact tool-token set',
        'test_context_used':False,'test_gold_used_in_generation':False})
    return queries,bundles,catalog,manifest,ev


def build_model(torch,agent_features,dim=128):
    nn=torch.nn
    class Denoiser(nn.Module):
        def __init__(self):
            super().__init__(); self.n=len(agent_features)
            self.register_buffer('features',torch.tensor(agent_features,dtype=torch.float32))
            self.agent_proj=nn.Linear(agent_features.shape[1],dim,bias=False)
            self.residual=nn.Embedding(self.n,dim); nn.init.normal_(self.residual.weight,std=.02)
            self.mask=nn.Parameter(torch.randn(dim)*.02)
            self.query=nn.Linear(agent_features.shape[1],dim)
            self.time=nn.Sequential(nn.Linear(1,dim),nn.SiLU(),nn.Linear(dim,dim))
            self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,4,dim*4,.1,batch_first=True),2,enable_nested_tensor=False)
            self.norm=nn.LayerNorm(dim); self.bias=nn.Parameter(torch.zeros(self.n))
        def forward(self,x,q,t,lengths):
            emb=self.agent_proj(self.features)+self.residual.weight
            h=emb[x.clamp(max=self.n-1)]
            h=torch.where((x==self.n)[...,None],self.mask,h)
            h=h+self.query(q)[:,None,:]+self.time(t[:,None])[:,None,:]
            pad=torch.arange(x.shape[1],device=x.device)[None,:]>=lengths[:,None]
            h=self.encoder(h,src_key_padding_mask=pad)
            return self.norm(h)@emb.T/math.sqrt(dim)+self.bias
    return Denoiser()


def train(args,out,queries,bundles,catalog):
    import torch,joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    # All fitted text statistics are training-only. No tuned external encoder.
    descriptions=[' '.join([a['llm_token']]+a['tool_tokens']).replace('_',' ').replace('&',' ') for a in catalog]
    vectorizer=TfidfVectorizer(max_features=40000,ngram_range=(1,2),sublinear_tf=True)
    mat=vectorizer.fit_transform(queries+descriptions)
    svd=TruncatedSVD(n_components=256,random_state=args.seed)
    features=normalize(svd.fit_transform(mat)).astype('float32')
    joblib.dump((vectorizer,svd),out/'text_encoder.joblib')
    q=torch.from_numpy(features[:len(queries)]).to(args.device); af=features[len(queries):]
    np.save(out/'agent_features.npy',af)
    model=build_model(torch,af).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    noise=load_module(Path(__file__).parent/'upstream/noise_schedule.py','ddbc_official_noise').LogLinearNoise()
    width=max(map(len,bundles)); n=len(catalog)
    target=torch.full((len(bundles),width),n,dtype=torch.long,device=args.device)
    lengths=torch.tensor(list(map(len,bundles)),device=args.device)
    for i,b in enumerate(bundles): target[i,:len(b)]=torch.tensor(b,device=args.device)
    history=[]; start=time.time()
    for epoch in range(args.epochs):
        model.train(); total=0.; steps=0
        for ix in torch.randperm(len(bundles),device=args.device).split(args.batch_size):
            x=target[ix]; le=lengths[ix]
            active=torch.arange(width,device=args.device)[None,:]<le[:,None]
            # Randomize real set members only. Padding remains outside the bundle.
            order=torch.rand(x.shape,device=args.device).masked_fill(~active,2).argsort(1)
            x=x.gather(1,order)
            t=torch.rand(len(ix),device=args.device)*.999+.001
            sigma,dsigma=noise(t); m=-torch.expm1(-sigma)
            mask=(torch.rand(x.shape,device=args.device)<m[:,None])&active
            xt=x.masked_fill(mask,n)
            logits=model(xt,q[ix],sigma,le)
            ce=torch.nn.functional.cross_entropy(logits.transpose(1,2),x.clamp(max=n-1),reduction='none')
            # SUBS: unmasked symbols copy exactly; their NLL is zero.
            loss=((ce*mask).sum(1)/le*(dsigma/torch.expm1(sigma))).mean()
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            total+=loss.item(); steps+=1
        history.append({'epoch':epoch+1,'loss':total/steps,'elapsed_seconds':time.time()-start})
        print(json.dumps(history[-1]),flush=True); write(out/'history.json',history)
    torch.save({'state_dict':model.state_dict(),'seed':args.seed,'epochs':args.epochs},out/'model.pt')
    return model,vectorizer,svd


def sample_sets(torch,model,q,length_probs,steps,draws,seed):
    """Absorbing-mask reverse kernel, random-order legal no-duplicate sampling."""
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device=q.device; n=model.n; width=len(length_probs)
    lengths=torch.multinomial(torch.tensor(length_probs,device=device),draws,replacement=True)+1
    x=torch.full((draws,width),n,device=device,dtype=torch.long)
    active=torch.arange(width,device=device)[None,:]<lengths[:,None]
    qs=q[None,:].expand(draws,-1); logscore=torch.zeros(draws,device=device)
    for step in range(steps,0,-1):
        t=step/steps; sigma=-math.log(max(1-.999*t,1e-7))
        logits=model(x,qs,torch.full((draws,),sigma,device=device),lengths)
        reveal=(torch.rand(x.shape,device=device)<1/step)&(x==n)&active
        # These constraints change the original independent sampler, explicitly documented.
        for pos in torch.randperm(width,device=device).tolist():
            rows=reveal[:,pos].nonzero().flatten()
            if not len(rows): continue
            scores=logits[rows,pos].clone()
            previous=x[rows]; ri,ci=(previous<n).nonzero(as_tuple=True)
            scores[ri,previous[ri,ci]]=-torch.inf
            probs=scores.softmax(-1)
            chosen=torch.multinomial(probs,1).squeeze(1)
            logscore[rows]+=probs.gather(1,chosen[:,None]).squeeze(1).log()
            x[rows,pos]=chosen
    assert not ((x==n)&active).any()
    return [row[:int(le)] for row,le in zip(x.cpu().tolist(),lengths.cpu().tolist())],logscore.cpu().tolist()


def infer(args,out,model,vectorizer,svd,bundles,catalog,manifest,ev):
    import torch
    from sklearn.preprocessing import normalize
    model.eval(); counts=collections.Counter(map(len,bundles)); probs=[counts[k]/len(bundles) for k in range(1,max(counts)+1)]
    # Generation receives query/sample ID only. Gold joins after all predictions exist.
    inputs=[{'query':r['query'],'sample_id':r['sample_id']} for r in manifest]
    features=normalize(svd.transform(vectorizer.transform([r['query'].strip() for r in inputs]))).astype('float32')
    predictions=[]
    with torch.inference_mode():
        for index,(inp,f) in enumerate(zip(inputs,features)):
            sets,scores=sample_sets(torch,model,torch.tensor(f,device=args.device),probs,args.steps,args.draws,args.seed+10000+index)
            freq=collections.Counter(i for b in sets for i in b)
            # Marginal inclusion frequency; mean sampled reverse-path score is only a tie break.
            tie=collections.defaultdict(list)
            for b,s in zip(sets,scores):
                assert len(b)==len(set(b))
                for i in b: tie[i].append(s/len(b))
            ranked=sorted(freq,key=lambda i:(-freq[i],-sum(tie[i])/len(tie[i]),i))[:10]
            results=[]
            for i in ranked:
                a=catalog[i]; text=' '.join([a['llm_token'],'<TOOL_SEP>']+(a['tool_tokens'] or ['<TOOL_EMPTY>'])+['<SPECIAL_END>'])
                results.append({**a,'rank':len(results)+1,'strict_text':text,'gen_text':text,'inclusion_frequency':freq[i]/args.draws})
            predictions.append({'sample_id':inp['sample_id'],'results':results,'sampled_agent_sets':sets,'sampled_reverse_path_scores':scores})
            print(f'inference {index+1}/{len(inputs)} unique={len(freq)} returned={len(results)}',flush=True)
    (out/'predictions_blind.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions))
    records=[{'ok':True,'baseline':'DDBC-Adapt-ID','query':gold['query'],'dataset_example':gold,'results':pred['results']} for gold,pred in zip(manifest,predictions)]
    (out/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    import baseline_top10_common as common
    common.evaluate(out,records,len(manifest))
    extra=[]; coverage=[]
    for r in records:
        gllm,gtools=ev.get_gold(r); gold=set(gtools); p=r['results'][0]; pred=set(p['tool_tokens'])
        extra.append({'sample_id':r['dataset_example']['sample_id'],'tool_jaccard':len(gold&pred)/len(gold|pred) if gold|pred else 1.0,
            'component_jaccard':len((gold|{gllm})&(pred|{p['llm_token']}))/len(gold|pred|{gllm,p['llm_token']})})
        coverage.append({'sample_id':r['dataset_example']['sample_id'],
            'exact_agent_in_catalog':any(gllm==a['llm_token'] and gold==set(a['tool_tokens']) for a in catalog),
            'complete_recall_possible':any(gllm==a['llm_token'] and gold<=set(a['tool_tokens']) for a in catalog),
            'max_component_recall':max(len((gold|{gllm})&(set(a['tool_tokens'])|{a['llm_token']}))/len(gold|{gllm}) for a in catalog)})
    write(out/'evaluation/additional_metrics.json',{k:sum(r[k] for r in extra)/len(extra) for k in ['tool_jaccard','component_jaccard']})
    write(out/'evaluation/catalog_coverage.json',{'mean':{k:sum(r[k] for r in coverage)/len(coverage) for k in ['exact_agent_in_catalog','complete_recall_possible','max_component_recall']},'per_sample':coverage})
    write(out/'evaluation/jaccard_per_sample.json',extra)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(str(AC_ROOT)))
    p.add_argument('--output',type=Path,required=True); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--lr',type=float,default=0.001); p.add_argument('--steps',type=int,default=16)
    p.add_argument('--draws',type=int,default=64); p.add_argument('--device',default='cuda')
    p.add_argument('--infer-only',action='store_true'); args=p.parse_args()
    sys.path.insert(0,str(args.root))
    import baseline5_run_infer_rag_gpt as b5
    b5.ensure_env_cuda_library()
    import torch,joblib
    torch.set_num_threads(4); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    out=args.output
    if not args.infer_only and (out/'config.json').exists(): raise ValueError('Output already exists; choose a new path')
    out.mkdir(parents=True,exist_ok=True)
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    cfg.update({'code_sha256':digest(__file__),'torch':torch.__version__,'numpy':np.__version__,
        'upstream_commit':'e1f4afc59121662e626ffefde79316a5e0dde044','official_noise_sha256':digest(Path(__file__).parent/'upstream/noise_schedule.py'),
        'checkpoint_selection':'fixed 30 epochs by default, no test/validation optimization',
        'query_encoder':'training-only word TFIDF(1,2) max40000 + SVD256 L2',
        'architecture':{'dim':128,'layers':2,'heads':4,'dropout':.1,'position_encoding':False},
        'ranking':'64 sampled variable-size sets; inclusion frequency, mean path score tie break',
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    if args.infer_only:
        saved=json.loads((out/'config.json').read_text())
        for k in ['seed','steps','draws','code_sha256']:
            assert cfg[k]==saved[k],f'Changed replay setting: {k}'
    else: write(out/'config.json',cfg)
    queries,bundles,catalog,manifest,ev=prepare(args.root,out)
    if args.infer_only:
        vectorizer,svd=joblib.load(out/'text_encoder.joblib'); af=np.load(out/'agent_features.npy')
        model=build_model(torch,af).to(args.device)
        model.load_state_dict(torch.load(out/'model.pt',map_location=args.device,weights_only=False)['state_dict'])
    else: model,vectorizer,svd=train(args,out,queries,bundles,catalog)
    infer(args,out,model,vectorizer,svd,bundles,catalog,manifest,ev)
    write(out/'completed.json',{'checkpoint_sha256':digest(out/'model.pt'),'predictions_sha256':digest(out/'predictions_blind.jsonl'),'results_sha256':digest(out/'results.jsonl')})

if __name__=='__main__': main()
