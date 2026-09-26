#!/usr/bin/env python3
"""Train-only component registry and frozen semantic features; no test gold input."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse,collections,hashlib,json,os,re,sys,time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def readl(path):return [json.loads(x) for x in open(path)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(str(AC_ROOT)));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.root));import baseline5_run_infer_rag_gpt as b5
    b5.ensure_env_cuda_library()
    import torch,numpy as np
    import baseline3_train_text2bundle_agent_adapted as b3
    from transformers import AutoTokenizer, AutoModel
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import joblib
    torch.set_num_threads(4);np.random.seed(42);torch.manual_seed(42)
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'ready.json').exists():raise ValueError('Prepared dataset already exists')
    td=a.root/'outputs/text2bundle_top10_disjoint_train_seed42';source=td/'training_excluding_table1.jsonl'
    raw=readl(source);tr=set((td/'train_qids.txt').read_text().splitlines());va=set((td/'val_qids.txt').read_text().splitlines())
    manifest=readl(b5.DEFAULT_SOURCE/'sample_manifest.jsonl')
    testq={r['query'].strip() for r in manifest};testid={str(r['qid']) for r in manifest}
    assert not tr&va
    assert not any(r['query'].strip() in testq or str(r['qid']) in testid for r in raw)
    train=[r for r in raw if str(r['qid']) in tr];val=[r for r in raw if str(r['qid']) in va]
    assert len(train)+len(val)==len(raw)
    assert not {r['query'].strip() for r in train}&{r['query'].strip() for r in val}
    def parse(r):
        text=str(r['target']).split('<SPECIAL_END>')[0].split('Explanation:')[0]
        llm,tokens,tools=b3.parse_gold_agent(text)
        surfaces={b3.canonical_tool(t):t for t in tokens}
        return llm,tools,surfaces
    surfaces=collections.defaultdict(collections.Counter);descs={};llmdesc={};allowed_train=[];excluded=collections.Counter()
    for r in train:
        llm,ts,ss=parse(r)
        if not llm or len(ts)>6:excluded['missing_llm' if not llm else 'over_six_tools']+=1;continue
        allowed_train.append((r,llm,ts));
        for t,s in ss.items():surfaces[t][s]+=1
        for t,d in b3.parse_intrinsic_tool_descriptions(str(r.get('context',''))).items():
            if len(d)>len(descs.get(t,'')):descs[t]=d
        for line in str(r.get('context','')).splitlines():
            m=re.search(r'token=(<LLM_[^<>]+>)',line)
            if not m:continue
            chunks=[c.strip() for c in line.split('|') if c.strip().startswith(('strengths=','desc='))]
            d='; '.join(chunks)
            if len(d)>len(llmdesc.get(m[1],'')):llmdesc[m[1]]=d[:1200]
    llms=sorted({llm for _,llm,_ in allowed_train});tools=sorted({t for _,_,ts in allowed_train for t in ts})
    catalog=[]
    for tok in llms:
        catalog.append({'id':len(catalog),'kind':'llm','identity':tok,'token':tok,'text':'Language model: '+tok[5:-1].replace('_',' ')+'. '+llmdesc.get(tok,'')})
    for t in tools:
        surface=sorted(surfaces[t],key=lambda s:(-surfaces[t][s],s))[0]
        catalog.append({'id':len(catalog),'kind':'tool','identity':t,'token':surface,'text':b3.tool_to_text(t,descs.get(t,''))})
    mapping={(r['kind'],r['identity']):r['id'] for r in catalog};queries=[];qmap={}
    def qi(q):
        q=q.strip()
        if q not in qmap:qmap[q]=len(queries);queries.append(q)
        return qmap[q]
    def record(r):
        llm,ts,_=parse(r)
        comps=[mapping.get(('llm',llm),-1)]+[mapping.get(('tool',t),-1) for t in ts]
        return {'qid':str(r['qid']),'query_index':qi(r['query']),'components':comps,'target':r['target'],'num_tools':len(ts)}
    train_records=[];seen=set()
    for r,llm,ts in allowed_train:
        rec=record(r);k=(rec['query_index'],tuple(sorted(rec['components'])))
        if k not in seen:train_records.append(rec);seen.add(k)
    n_train_queries=len(queries)
    val_records=[record(r) for r in val]
    # Only manifest query and identifying fields are allowed across this boundary.
    test_inputs=[{'sample_id':r['sample_id'],'qid':str(r['qid']),'query_index':qi(r['query'])} for r in manifest]
    write(out/'catalog.json',catalog);write(out/'queries.json',queries);write(out/'train.json',train_records);write(out/'validation.json',val_records);write(out/'test_inputs.json',test_inputs)
    encoder_path=Path('/root/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2')
    revision=(encoder_path/'refs/main').read_text().strip();snapshot=encoder_path/'snapshots'/revision
    tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    encoder=AutoModel.from_pretrained(str(snapshot),local_files_only=True).cuda().eval()
    maxlen=384
    def encode_full(texts,label):
        chunks=[];owners=[];lengths=[]
        usable=maxlen-tokenizer.num_special_tokens_to_add(pair=False)
        for i,text in enumerate(texts):
            ids=tokenizer.encode(text,add_special_tokens=False,truncation=False)
            if not ids:ids=[tokenizer.unk_token_id]
            for start in range(0,len(ids),usable):
                part=ids[start:start+usable];chunks.append(tokenizer.build_inputs_with_special_tokens(part));owners.append(i);lengths.append(len(part))
        print(label,'texts',len(texts),'chunks',len(chunks),flush=True)
        all_embeddings=[]
        with torch.inference_mode():
            for start in range(0,len(chunks),128):
                tok=tokenizer.pad({'input_ids':chunks[start:start+128]},padding=True,return_tensors='pt')
                tok={k:v.cuda() for k,v in tok.items()};hidden=encoder(**tok).last_hidden_state
                mask=tok['attention_mask'][...,None];pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                all_embeddings.append(torch.nn.functional.normalize(pooled,dim=1).cpu().numpy())
                if start%2048==0:print(label,'encoded',min(start+128,len(chunks)),'/',len(chunks),flush=True)
        es=np.concatenate(all_embeddings)
        emb=np.zeros((len(texts),es.shape[1]),dtype=np.float32);weight=np.zeros(len(texts))
        for e,i,n in zip(es,owners,lengths):emb[i]+=e*n;weight[i]+=n
        emb/=np.maximum(weight[:,None],1);emb/=np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-12)
        return emb,{'texts':len(texts),'chunks':len(chunks),'long_texts':sum(v>1 for v in collections.Counter(owners).values()),'max_tokens':maxlen,'chunking':'nonoverlap token windows, token-count weighted normalized mean'}
    cf,cstats=encode_full([x['text'] for x in catalog],'components')
    qf,qstats=encode_full(queries,'queries');np.save(out/'component_semantics.npy',cf);np.save(out/'query_semantics.npy',qf)
    del encoder;torch.cuda.empty_cache()
    pca=PCA(n_components=128,random_state=42);features=pca.fit_transform(cf).astype('float32');joblib.dump(pca,out/'pca.joblib')
    residual=features.copy();codes=[];centroids=[];errors=[]
    for level in range(3):
        km=KMeans(n_clusters=128,n_init=3,max_iter=100,random_state=42+level)
        labels=km.fit_predict(residual);centers=km.cluster_centers_.astype('float32')
        residual-=centers[labels];codes.append(labels);centroids.append(centers);errors.append(float(np.mean(np.sum(residual**2,axis=1))))
        print('RVQ level',level,'residual_mse',errors[-1],flush=True)
    codes=np.stack(codes,axis=1);used=collections.Counter();dedup=[]
    for code in codes:
        k=tuple(code);dedup.append(used[k]);used[k]+=1
    code_sizes=[128]*3+[max(dedup)+1];codes=np.column_stack([codes,dedup]).astype('int64')
    assert len({tuple(r) for r in codes})==len(catalog)
    np.save(out/'codes.npy',codes);np.save(out/'rvq_centroids.npy',np.stack(centroids))
    np.save(out/'component_pca.npy',features)
    audit={'source_sha256':sha(source),'train_qids_sha256':sha(td/'train_qids.txt'),'val_qids_sha256':sha(td/'val_qids.txt'),'test_manifest_sha256':sha(b5.DEFAULT_SOURCE/'sample_manifest.jsonl'),
        'train_test_qid_overlap':0,'train_test_query_overlap':0,'train_val_query_overlap':0,'raw_train_rows':len(train),'train_unique_query_bundles':len(train_records),
        'train_queries':n_train_queries,'validation_rows':len(val_records),'validation_fully_representable_rows':sum(-1 not in r['components'] and r['num_tools']<=6 for r in val_records),
        'excluded_train_rows':dict(excluded),'llms':len(llms),'tools':len(tools),'tool_descriptions_available':sum(t in descs for t in tools),'llm_descriptions_available':sum(t in llmdesc for t in llms),
        'codebook_sizes':code_sizes,'unique_code_tuples':len(catalog),'semantic_collision_groups':sum(v>1 for v in used.values()),
        'rvq_residual_squared_error':errors,'pca_retained_variance':float(pca.explained_variance_ratio_.sum()),
        'encoder':'sentence-transformers/all-mpnet-base-v2','encoder_revision':revision,'encoder_finetuned_here':False,'encoder_pooling':'official model-card attention-masked mean + L2; AutoModel avoids missing cached pooling config','encoder_path':str(snapshot),
        'component_encoding':cstats,'query_encoding':qstats,'query_embeddings_include_val_test':True,'encoder_is_frozen':True,
        'all_fitted_registry_pca_rvq_training_only':True,'metadata_source':'training context intrinsic descriptions only; no scores/ranks or target explanations',
        'test_gold_in_preparation':False,'train_tool_count_histogram':dict(collections.Counter(r['num_tools'] for r in train_records))}
    write(out/'audit.json',audit)
    hashes={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='ready.json'}
    write(out/'ready.json',{'file_hashes':hashes,'prepare_code_sha256':sha(__file__)})
    print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()
