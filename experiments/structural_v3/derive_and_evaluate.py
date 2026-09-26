from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
#!/usr/bin/env python3
import argparse,copy,csv,json,math,re,statistics,sys
from pathlib import Path
from run_variants import ROOT,EXP,WEIGHTS,MANIFEST_SHA,load_inputs,write,base
LABELS=['ToolR@1','Tool-Hit@10','CompR@1','CR-Hit@1','CR-Hit@10','CR-MRR@10','RDCR@10','SUCR@10','OGR@10','Oracle-RDCR@10']
FIELDS=['top1_tool_recall','tool_hit@10','top1_component_recall','top1_complete_recall','cr_hit@10','cr_mrr@10','rdcr@10']
ORDER=['Free','Greedy','Beam','Critic','Beam+Critic','Critic+Cartesian']
def key(llm,tools):return llm,tuple(sorted(set(tools)-{'<TOOL_EMPTY>','<TOOL_SEP>'}))
def parse(text):
    text=text.split('<SPECIAL_END>')[0].split('Explanation:')[0]
    llms=re.findall(r'<LLM_[^<>\n\r]+>',text)
    tools=(set(re.findall(r'<<[^<>\n\r]+>>',text))|set(re.findall(r'<TOOL_[^<>\n\r]+>',text)))-{'<TOOL_EMPTY>','<TOOL_SEP>'}
    return (llms[0] if llms else ''),tools

def derive(final_mode):
    manifest,source_rows=load_inputs()
    saved_source=ROOT/'outputs/table1_ours_v10_compact_fixedtest100'
    cache=ROOT/'outputs/OURS_V10_CARTESIAN_CRITIC_TERRA_V3'
    cfg=json.loads((saved_source/'ours_config.json').read_text())
    cachecfg=json.loads((cache/'config.json').read_text())
    assert cachecfg['critic_model']['weights_sha256']==WEIGHTS
    assert cachecfg['source_sha256']==base.file_sha(saved_source/'results.jsonl')
    assert cfg['settings']['beam_min_size']==1 and cfg['settings']['beam_max_size']==2
    assert cfg['adapter_weights_sha256']=='97da97b8d7dde0a21792df9968112fd972da69677fc2324cae1a5af347047e37'
    assert cfg['compact_context']['frozen_test_sha256']==MANIFEST_SHA
    byid={r['dataset_example']['sample_id']:r for r in base.read_lines(saved_source/'results.jsonl')}
    collections={v:[] for v in ['Beam','Critic','Critic+Cartesian']}
    sota=base.load_module(ROOT/'inference/run_infer_v13_batch_eval_sota.py','derive_sota')
    config={'final_ranking':final_mode,'source_config_sha256':base.sha(cfg),
        'source_results_sha256':base.file_sha(saved_source/'results.jsonl'),
        'critic_cache_config_sha256':base.sha(cachecfg),'critic_model':cachecfg['critic_model'],
        'source_search':'generator-only saved complete pool before final cap, no old critic selection reused',
        'cartesian_toolsets':'unique toolsets from current v3 Critic top10',
        'cartesian_llms':'same retrieved top10','runner_sha256':base.file_sha(__file__),
        'tie_break':'critic groups: descending raw score then canonical LLM/tool key; beam: descending generator score then generator average then node id'}
    for sample,src in zip(manifest,source_rows):
        sid=sample['sample_id'];r=byid[sid]
        assert r['ok'] and r['dataset_example']==sample
        assert r['context']==src['context'] and r['config_sha256']==base.sha(cfg)
        trace=r['generation']['search_trace']
        assert trace['search_critic_enabled'] is False
        assert all(e['stage']=='final_rerank' and not e.get('error') for e in r['generation']['critic_api_events'])
        pool=trace['final_rerank_pool_before_cap'];assert len(pool)==trace['final_rerank_candidate_count']
        scored=json.loads((cache/'scored_cartesian_candidates'/(sid+'.json')).read_text())
        assert scored['config_sha256']==base.sha(cachecfg)
        products={key(p['llm_token'],p['tool_tokens']):copy.deepcopy(p) for p in scored['products']}
        assert len(products)==len(scored['products'])
        nodes=[]
        for p in pool:
            assert p['is_complete']
            n={k:copy.deepcopy(p[k]) for k in ['node_id','llm','tools','generator_avg_logprob','generator_logprob','generator_token_count']}
            n['critic_raw_v3']=products[key(n['llm'],n['tools'])]['critic_raw']
            assert math.isfinite(n['critic_raw_v3'])
            n['generator_only_score']=n['generator_avg_logprob']-0.1*len(n['tools'])
            nodes.append(n)
        # Reconstruct the pre-cap completed pool; never reuse original selected top10.
        beam=sorted(nodes,key=lambda n:(-n['generator_only_score'],-n['generator_avg_logprob'],n['node_id']))[:10]
        if final_mode=='raw':
            critic=sorted(nodes,key=lambda n:(-n['critic_raw_v3'],key(n['llm'],n['tools'])))[:10]
        else:
            avg=statistics.fmean(n['generator_avg_logprob'] for n in nodes);std=statistics.pstdev(n['generator_avg_logprob'] for n in nodes)
            ca=statistics.fmean(n['critic_raw_v3'] for n in nodes);cs=statistics.pstdev(n['critic_raw_v3'] for n in nodes)
            for n in nodes:n['hybrid_score']=.5*((n['critic_raw_v3']-ca)/cs if cs>=1e-8 else 0)+.15*((n['generator_avg_logprob']-avg)/std if std>=1e-8 else 0)-.1*len(n['tools'])
            critic=sorted(nodes,key=lambda n:(-n['hybrid_score'],-n['critic_raw_v3'],-n['generator_avg_logprob']))[:10]
        selected_tools={key(n['llm'],n['tools'])[1] for n in critic}
        llms,_=sota.extract_candidates_from_context(src['context']['text']);llms=list(dict.fromkeys(llms))[:10]
        assert len(llms)==10
        cartkeys={key(l,t) for t in selected_tools for l in llms}
        assert cartkeys <= set(products)
        cart=sorted([products[k] for k in cartkeys],key=lambda n:(-n['critic_raw'],key(n['llm_token'],n['tool_tokens'])))[:10]
        for variant,selected in [('Beam',beam),('Critic',critic),('Critic+Cartesian',cart)]:
            results=[]
            for rank,n in enumerate(selected,1):
                llm=n.get('llm',n.get('llm_token'));tools=n.get('tools',n.get('tool_tokens'))
                results.append({'rank':rank,'llm_token':llm,'tool_tokens':tools,
                    'strict_text':' '.join([llm,'<TOOL_SEP>',*tools,'<SPECIAL_END>']),
                    'score_details':n})
            record={'ok':True,'variant':variant,'dataset_example':sample,'results':results,
                    'config_sha256':base.sha(config),'available_pool_count':len(cartkeys) if variant=='Critic+Cartesian' else len(nodes),
                    'generation':{'search_critic_enabled':False,'final_critic_reranking':variant!='Beam'},
                    'provenance':{'source_sample':str(saved_source/'per_sample'/(sid+'.json')),
                        'v3_score_cache':str(cache/'scored_cartesian_candidates'/(sid+'.json')),
                        'v3_cache_sha256':base.file_sha(cache/'scored_cartesian_candidates'/(sid+'.json'))}}
            write(EXP/variant/'per_sample'/(sid+'.json'),record);collections[variant].append(record)
        write(EXP/'shared_completed_pools'/(sid+'.json'),{'sample_id':sid,'nodes':nodes})
    for variant,rows in collections.items():
        out=EXP/variant;write(out/'config.json',config)
        (out/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        (out/'sample_manifest.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in manifest))
        write(out/'status.json',{'state':'complete','completed':len(rows),'total':100})
    print('[DERIVED]',list(collections),flush=True)

def metrics(record):
    gl,gt=parse(record['dataset_example']['target']);assert gl
    gc=gt|{gl}; union=set();tr=[];cr=[];th=[];ch=[];seen=set()
    for p in record['results'][:10]:
        l=p.get('llm_token','');t=set(p.get('tool_tokens',[]))-{'<TOOL_EMPTY>','<TOOL_SEP>'}
        if not l and not t:l,t=parse(p.get('gen_text',''))
        k=key(l,t);assert k not in seen;seen.add(k)
        comps=t|({l} if l else set());union|=comps
        tr.append(len(gt&t)/len(gt) if gt else 1.)
        cr.append(len(gc&comps)/len(gc));th.append(gt<=t);ch.append(gl==l and gt<=t)
    discounts=[1/math.log2(r+1) for r in range(1,11)]
    rd=lambda vals:sum(v*d for v,d in zip(vals,discounts))/sum(discounts)
    raw=[tr[0] if tr else 0,float(any(th)),cr[0] if cr else 0,float(ch[0]) if ch else 0,float(any(ch)),
         next((1/(i+1) for i,h in enumerate(ch) if h),0),rd(cr),len(gc&union)/len(gc),max(cr,default=0),rd(sorted(cr,reverse=True))]
    assert raw[7]+1e-12>=raw[8] and raw[9]+1e-12>=raw[6]
    return dict(zip(LABELS,raw))

def evaluate(require_all=False):
    manifest,_=load_inputs()
    evaluator=base.load_module(ROOT/'evaluation/reference/exp3/evaluate_ranked_recall_baseline4.py','structural_evaluator')
    summaries={};missing=[]
    for variant in ORDER:
        out=EXP/variant
        if not (out/'results.jsonl').exists():missing.append(variant);continue
        rows=base.read_lines(out/'results.jsonl');assert len(rows)==100
        cfg=json.loads((out/'config.json').read_text())
        result=[];maxdiff=0.;unparseable=[]
        for sample,r in zip(manifest,rows):
            assert r['ok'] and r['dataset_example']==sample and r['config_sha256']==base.sha(cfg)
            assert [p['rank'] for p in r['results']]==list(range(1,len(r['results'])+1))
            m=metrics(r)
            if evaluator.get_candidates(r):
                original=evaluator.evaluate_one(r)
                maxdiff=max(maxdiff,max(abs(m[k]-original[f]) for k,f in zip(LABELS,FIELDS)))
            else:
                assert all(v==0 for v in m.values()),m;unparseable.append(sample['sample_id'])
            result.append({'sample_id':sample['sample_id'],**m})
        assert maxdiff<1e-12,maxdiff
        raw={k:statistics.fmean(r[k] for r in result) for k in LABELS}
        report={'n':100,'raw':raw,'percent':{k:v*100 for k,v in raw.items()},'official_evaluator_max_difference':maxdiff,
                'unparseable_samples_counted_as_zero':unparseable,'candidate_count_min':min(len(r['results']) for r in rows),
                'candidate_count_max':max(len(r['results']) for r in rows)}
        write(out/'evaluation/metrics.json',report);write(out/'evaluation/per_sample.json',result)
        summaries[variant]=report
    write(EXP/'SUMMARY.json',{'complete':not missing,'missing':missing,'variants':summaries})
    with (EXP/'results_percent.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['Variant']+LABELS)
        for v,r in summaries.items():w.writerow([v]+[f'{r["percent"][k]:.4f}' for k in LABELS])
    print(json.dumps({v:r['percent'] for v,r in summaries.items()},indent=2),flush=True)
    if require_all:assert not missing,missing
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--derive',action='store_true');ap.add_argument('--final_mode',choices=['raw','hybrid'],default='raw');ap.add_argument('--require_all',action='store_true');a=ap.parse_args()
    if a.derive:derive(a.final_mode)
    evaluate(a.require_all)
