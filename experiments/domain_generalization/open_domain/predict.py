"""Frozen single-skill-bundle adaptation; gold is read only after prediction."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse, collections, fcntl, hashlib, importlib.util, json, math, os
from pathlib import Path
import random, sys, time, traceback, urllib.request

ROOT = Path(str(AC_ROOT))
LLM = '<LLM_GPT-4>'

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def signature(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()

def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x, indent=2, ensure_ascii=False)); t.replace(p)

def request(url, payload=None):
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
    with op.open(req, timeout=180) as f: return json.load(f)

def load(path, name):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s)
    sys.modules[name] = m; s.loader.exec_module(m); return m

def evaluate(out, data, config_hash):
    qs = json.loads((data/'questions/merge.json').read_text())
    gold = json.loads((data/'rankings/merge.json').read_text())['rankings']
    groups = collections.defaultdict(list)
    methods = ['generator', 'ours_generator10_critic', 'critic_all50']
    for qid, q in qs.items():
        r = json.loads((out/'per_sample'/f'{qid}.json').read_text())
        assert r['config_hash'] == config_hash and r['query'] == q['input']
        assert len(gold[qid]) == 1
        for method in methods:
            ranked = r['rankings'][method]
            assert len(ranked) == 10 and len(set(ranked)) == 10
            rank = ranked.index(gold[qid][0])+1 if gold[qid][0] in ranked else 0
            metrics = {'Hit@1':int(rank==1), 'Hit@5':int(0<rank<=5), 'Hit@10':int(rank>0), 'MRR@10':1/rank if rank else 0}
            labels = ['all1000', 'all/difficulty/'+q['metadata']['difficulty']]
            if q['metadata']['judge_verdict']=='ok':
                labels += ['judge_ok776', 'ok/difficulty/'+q['metadata']['difficulty'], 'ok/category/'+q['metadata']['category']]
            for label in labels: groups[(label,method)].append(metrics)
    result = {}
    for (group,method), vals in groups.items():
        result.setdefault(group,{})[method] = {'n':len(vals), **{k:sum(x[k] for x in vals)/len(vals) for k in vals[0]}}
    write(out/'metrics.json', {'scale':'fractions, not percent', 'results':result})
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ROOT/'outputs/OPENClaw50_V10_CRITICV3_FROZEN')
    ap.add_argument('--limit',type=int,default=0); ap.add_argument('--preflight',action='store_true'); args=ap.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    data=ROOT/'dataset_opendomain'; modeldir=ROOT/'checkpoints/generative_v10_compact'
    sys.path.insert(0,str(ROOT/'training/generator'))
    from compact_context import build_prompt, parse_context, components
    sota=load(ROOT/'inference/run_infer_v13_batch_eval_sota.py','skill_sota')
    tok=sota.load_tokenizer_peft_aware(str(modeldir))
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    catalog=json.loads((data/'agents/merge.json').read_text())
    questions=json.loads((data/'questions/merge.json').read_text())
    # The candidate order depends only on the public catalog, never query metadata.
    ids=sorted(catalog); random.Random(42).shuffle(ids)
    tools={aid:'<<'+catalog[aid]['C']['source_slug']+'>>' for aid in ids}
    assert len(set(tools.values()))==50
    context='cf_retrieved_llm:\n'+LLM+' [1]\nsemantic_retrieved_tool:\n'+'\n'.join(tools.values())+'\nevidence:\n'
    context+='[1] token='+LLM+' | desc=General-purpose language model; fixed conditioning for every skill.\n'
    for i,aid in enumerate(ids,2):
        desc=' '.join(catalog[aid]['C']['description'].split())
        assert not any(x in desc for x in ['<<','>>','<LLM_','<TOOL_'])
        context+=f'[{i}] token={tools[aid]} | desc={desc}\n'
    assert set(parse_context(context)[1])==set(tools.values())
    health=request('http://127.0.0.1:8013/health')
    assert health['status']=='ok' and health['model']['serialization_version']=='critic-compact-v3'
    assert health['model']['checkpoint_dir']==str(AC_ROOT / 'checkpoints/bundle_critic_terra_v3')
    config={'version':'frozen-single-skill-v1','model':str(modeldir),'critic':health['model'], 'fixed_llm':LLM,
            'candidate_order':ids,'catalog_size':50,'generator_shortlist':10,'final_ranking':'critic_raw descending on generator top10',
            'generator_score':'mean log probability of complete skill phrase and SPECIAL_END, conditioned on fixed LLM and TOOL_SEP',
            'scope':'single skill bundle; no training, no LLM selection, no multi-skill composition, no retrieval filtering',
            'source_hashes':{str(p):digest(p) for p in [Path(__file__),ROOT/'inference/run_infer_v13_batch_eval_sota.py',ROOT/'training/generator/compact_context.py',modeldir/'training_provenance.json',modeldir/'adapter_config.json',modeldir/'tokenizer.json',data/'agents/merge.json',data/'questions/merge.json',data/'rankings/merge.json']}}
    fingerprint=signature(config)
    if (out/'config.json').exists(): assert json.loads((out/'config.json').read_text())==config,'Config changed; use separate output directory'
    else: write(out/'config.json',config)
    write(out/'catalog.json', [{ 'agent_id':aid,'token':tools[aid],**catalog[aid]['C']} for aid in ids])
    (out/'context.txt').write_text(context)
    lengths=[]
    for q in questions.values():
        prompt,meta=build_prompt(context,q['input'],tok)
        assert set(components(prompt.split('### User Query:')[0])[1])==set(tools.values())
        lengths.append(meta['prompt_tokens'])
    write(out/'preflight.json',{'queries':len(questions),'candidates':50,'all_candidate_ids_preserved':True,'prompt_max':max(lengths),'prompt_mean':sum(lengths)/len(lengths),'critic_health':health,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
    print(f'PREFLIGHT OK {len(questions)} queries, 50 candidates; max prompt={max(lengths)}',flush=True)
    if args.preflight:return
    import torch
    random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model=sota._load_model_peft_aware(str(modeldir),token=False)
    assert not model.training
    model.requires_grad_(False)
    phrases=[tools[aid]+' <SPECIAL_END>' for aid in ids]; reverse={p:aid for p,aid in zip(phrases,ids)}
    selected=list(questions.items()); selected=selected[:args.limit] if args.limit else selected
    started=time.time(); done=0
    for qid,q in selected:
        path=out/'per_sample'/f'{qid}.json'
        if path.exists():
            old=json.loads(path.read_text()); assert old['config_hash']==fingerprint and old['query']==q['input'];done+=1;continue
        t=time.time()
        # Only q.input reaches models. Metadata and gold never enter prediction.
        query=q['input']; prompt,meta=build_prompt(context,query,tok)
        prefix=prompt+LLM+' <TOOL_SEP>'
        enc=tok(prefix,return_tensors='pt',add_special_tokens=True)
        enc={k:v.to(model.device) for k,v in enc.items()}
        with torch.inference_mode():
            scored=sota.rank_phrases_trie_constrained(model,tok,prefix_ids=enc['input_ids'],prefix_mask=enc['attention_mask'],candidate_phrases=phrases,leading_space=True,top_k=50,score_batch_size=8)
        assert len(scored)==50 and len({x[0] for x in scored})==50
        gen=[{'agent_id':reverse[p],'sum_logprob':float(total),'mean_logprob':float(avg),'token_count':len(tokens)} for p,tokens,total,avg in scored]
        assert all(math.isfinite(x['mean_logprob']) for x in gen)
        gen.sort(key=lambda x:(-x['mean_logprob'],x['agent_id']))
        payload={'query':query,'evidence_context':context,'candidates':[{'id':aid,'llm':LLM,'tools':[tools[aid]]} for aid in ids],'top_k':50}
        response=request('http://127.0.0.1:8013/v1/rerank',payload)
        cr={x['id']:float(x['raw_score']) for x in response['ranking']}
        assert set(cr)==set(ids) and all(math.isfinite(v) for v in cr.values())
        shortlist=[x['agent_id'] for x in gen[:10]]
        rankings={'generator':shortlist,'ours_generator10_critic':sorted(shortlist,key=lambda aid:(-cr[aid],aid)), 'critic_all50':sorted(ids,key=lambda aid:(-cr[aid],aid))[:10]}
        write(path,{'query_id':qid,'query':query,'config_hash':fingerprint,'rankings':rankings,'generator_scores':gen,'critic_scores':cr,'prompt':prompt,'prompt_stats':meta,'elapsed_sec':time.time()-t})
        done+=1
        write(out/'progress.json',{'status':'running','completed':done,'expected':len(selected),'last_query':qid,'last_sec':time.time()-t,'elapsed_sec':time.time()-started})
        print(f'OK {done}/{len(selected)} {qid} {time.time()-t:.2f}s',flush=True)
    if not args.limit:
        metrics=evaluate(out,data,fingerprint)
        print(json.dumps({k:metrics[k] for k in ['all1000','judge_ok776']},indent=2),flush=True)
    write(out/'progress.json',{'status':'complete','completed':done,'expected':len(selected),'elapsed_sec':time.time()-started,'evaluation_complete':not bool(args.limit)})

if __name__=='__main__':
    main()
