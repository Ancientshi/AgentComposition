#!/usr/bin/env python3
"""Reuse saved Ours tool bundles, cross with retrieved LLMs, critic-rank only.

No generator weights/tokenizer/search/retrieval are invoked. Gold is attached
only after candidate construction and scoring, for evaluation and diagnostics.
The original hybrid score is deliberately not transferred to new LLMs.
"""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse
import json
import math
from pathlib import Path
import statistics
import time
from collections import Counter
import sys
sys.path.insert(0,str(AC_ROOT))
import baseline5_run_infer_rag_gpt as io

ROOT=Path(str(AC_ROOT))
DEFAULT_SOURCE=ROOT/'outputs/table1_ours_seed42_n100_beam1-2_critic05_finalonly'
DEFAULT_OUTPUT=ROOT/'outputs/OURS_CARTESIAN_LLM10_CRITIC_RERANK_FROM_RECORDS_seed42_n100'
VIEWS={
    'FINAL10_TOOLSETS_X_TOP10_LLM':'Ours final-10 toolsets x LLM10 / critic-only',
    'COMPLETED_TOOLSETS_X_TOP10_LLM':'Ours completed toolsets x LLM10 / critic-only',
    'CONTROL_ORIGINAL_FINAL10_CRITIC_ONLY':'Ours original final-10 / critic-only control',
    'CONTROL_ORIGINAL_COMPLETED_CRITIC_ONLY':'Ours original completed pool / critic-only control',
}

def bundle_key(tools):
    return tuple(sorted(set(t for t in tools if t!='<TOOL_EMPTY>')))

def candidate_key(llm,tools):
    return llm,bundle_key(tools)

def build_expansion(original_final,original_completed,llms):
    """Inputs intentionally exclude targets/query labels; retain tool order."""
    llms=list(dict.fromkeys(llms))
    toolsets={}
    for item in original_completed:
        tools=list(dict.fromkeys(t for t in item['tools'] if t!='<TOOL_EMPTY>'))
        toolsets.setdefault(bundle_key(tools),tools)
    final_keys={bundle_key(r['tool_tokens']) for r in original_final}
    if not final_keys.issubset(toolsets):
        raise ValueError('Final toolset absent from saved completed pool')
    products=[]
    for ti,(key,tools) in enumerate(toolsets.items()):
        for li,llm in enumerate(llms):
            products.append({'id':f'tools_{ti:03d}_llm_{li:02d}', 'llm_token':llm,
                'tool_tokens':tools,'retrieval_llm_rank':li+1,'source_toolset_index':ti,
                'from_original_final10_toolset':key in final_keys})
    original_final_keys={candidate_key(r['llm_token'],r['tool_tokens']) for r in original_final}
    original_completed_keys={candidate_key(r['llm'],r['tools']) for r in original_completed}
    product_keys={candidate_key(r['llm_token'],r['tool_tokens']) for r in products}
    if not original_completed_keys.issubset(product_keys):
        raise ValueError('An original LLM is absent from retrieved Top10')
    if len(product_keys)!=len(products): raise ValueError('Duplicate Cartesian candidates')
    return products,original_final_keys,original_completed_keys

def select_view(products,name,original_final_keys,original_completed_keys):
    if name=='FINAL10_TOOLSETS_X_TOP10_LLM':
        selected=[r for r in products if r['from_original_final10_toolset']]
    elif name=='COMPLETED_TOOLSETS_X_TOP10_LLM': selected=list(products)
    elif name=='CONTROL_ORIGINAL_FINAL10_CRITIC_ONLY':
        selected=[r for r in products if candidate_key(r['llm_token'],r['tool_tokens']) in original_final_keys]
    elif name=='CONTROL_ORIGINAL_COMPLETED_CRITIC_ONLY':
        selected=[r for r in products if candidate_key(r['llm_token'],r['tool_tokens']) in original_completed_keys]
    else: raise ValueError(name)
    # Exact ties use retrieval LLM rank, then saved toolset order, never labels.
    return sorted(selected,key=lambda r:(-r['critic_raw'],r['retrieval_llm_rank'],r['source_toolset_index']))

def evaluate_views(output,manifest,scored_records,evaluator,sota):
    summary={}
    fields=['top1_tool_recall','tool_hit@10','top1_component_recall','top1_complete_recall',
            'cr_hit@10','cr_mrr@10','rdcr@10']
    for name,label in VIEWS.items():
        records=[]; evaluations=[]; diagnostics=[]
        directory=output/name
        for sample,saved in zip(manifest,scored_records):
            assert saved['sample_id']==sample['sample_id']
            final_keys={(r[0],tuple(r[1])) for r in saved['original_final_keys']}
            completed_keys={(r[0],tuple(r[1])) for r in saved['original_completed_keys']}
            pool=select_view(saved['products'],name,final_keys,completed_keys)
            results=[]
            for rank,candidate in enumerate(pool[:10],1):
                text=sota.build_strict_text(llm_token=candidate['llm_token'],tool_tokens=candidate['tool_tokens'],
                    tool_sep_token='<TOOL_SEP>',end_token='<SPECIAL_END>',tool_empty_token='<TOOL_EMPTY>')
                results.append({'rank':rank,**candidate,'strict_text':text,'gen_text':text})
            record={'ok':True,'baseline':label,'dataset_example':sample,'results':results,
                'generation':{'num_return_sequences':len(results),'results':results},
                'reranking':'critic raw score only; no transferred generator scores or length penalty',
                'available_pool_count':len(pool),'config_sha256':saved['config_sha256']}
            io.write_json(directory/'per_sample'/f"{sample['sample_id']}.json",record)
            ev=evaluator.evaluate_one(record)
            if ev['num_candidates']!=len(results): raise ValueError('Evaluator dropped a candidate')
            records.append(record); evaluations.append(ev)
            gold_llm=ev['gold_llm']; gold_tools=set(ev['gold_tools'])
            diagnostics.append({'sample_id':sample['sample_id'],'available_pool_count':len(pool),
                'available_llm_count':len({r['llm_token'] for r in pool}),
                'top10_llm_count':len({r['llm_token'] for r in results}),
                'gold_llm_available':any(r['llm_token']==gold_llm for r in pool),
                'gold_llm_in_top10':any(r['llm_token']==gold_llm for r in results),
                'complete_agent_available':any(r['llm_token']==gold_llm and gold_tools.issubset(r['tool_tokens']) for r in pool),
                'complete_agent_in_top10':bool(ev['cr_hit@10'])})
        means={k:statistics.fmean(e[k] for e in evaluations) for k in fields}
        report={'n':len(records),'mean':means,'candidate_counts':dict(Counter(len(r['results']) for r in records)),
            'all_have_ten':all(len(r['results'])==10 for r in records),
            'gold_source':'dataset_example.target only','missing_rank_policy':'zero',
            'diagnostics':{k:statistics.fmean(r[k] for r in diagnostics) for k in diagnostics[0] if k!='sample_id'}}
        io.write_json(directory/'evaluation/metrics.json',report)
        io.write_json(directory/'evaluation/coverage_per_sample.json',diagnostics)
        (directory/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        (directory/'sample_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest))
        values=[f'{means[k]:.4f}' if k=='cr_mrr@10' else f'{means[k]*100:.2f}' for k in fields]
        (directory/'evaluation/table1_row.tex').write_text(label+' & '+' & '.join(values)+r' \\'+'\n')
        summary[name]=report
        print('[RESULT]',name,' & '.join(values),flush=True)
    io.write_json(output/'COMPARISON_SUMMARY.json',summary)
    return summary

def main():
    io.ensure_env_cuda_library()
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    ap.add_argument('--output',type=Path,default=None)
    ap.add_argument('--critic_url',default='http://127.0.0.1:8013')
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--check_only',action='store_true')
    args=ap.parse_args()
    if args.limit<0: ap.error('limit must be nonnegative')
    output=args.output or Path(str(DEFAULT_OUTPUT)+(f'__SMOKE_{args.limit}' if args.limit else ''))
    if output.resolve()==args.source.resolve(): raise ValueError('Source cannot be overwritten')
    original=io.load_module(ROOT/'inference/run_infer_v13_table1_ours.py','cartesian_original')
    sota=io.load_module(ROOT/'inference/run_infer_v13_batch_eval_sota.py','cartesian_sota')
    evaluator=io.load_module(ROOT/'evaluation/reference/exp3/evaluate_ranked_recall_baseline2.py','cartesian_evaluator')
    manifest=io.read_jsonl(args.source/'sample_manifest.jsonl')
    rows=io.read_jsonl(args.source/'results.jsonl')
    by_id={r['dataset_example']['sample_id']:r for r in rows}
    if len(by_id)!=len(rows) or len(rows)!=len(manifest): raise ValueError('Source sample mismatch')
    if args.limit: manifest=manifest[:args.limit]
    source_config=json.loads((args.source/'ours_config.json').read_text())
    health=original.critic_health(args.critic_url)
    if health['model'].get('serialization_version')!='critic-compact-v3' or health['model'].get('checkpoint_dir')!=str(AC_ROOT / 'checkpoints/bundle_critic_terra_v3'):
        raise ValueError('Expected isolated v3 critic service')
    config={'experiment':'OURS CARTESIAN LLM10 CRITIC RERANK FROM RECORDS',
        'source':str(args.source.resolve()),'source_sha256':io.file_digest(args.source/'results.jsonl'),
        'source_code_sha256':io.file_digest(__file__),'critic_model':health['model'],'critic_url':args.critic_url,
        'sample_count':len(manifest),'limit':args.limit,'llm_pool':'same saved retrieved Top10',
        'tool_pool':'same saved final_rerank_pool_before_cap, unordered toolset dedup',
        'ranking':'critic raw score descending; retrieval LLM rank then saved toolset index break exact ties',
        'generator_rescored':False,'generator_score_reused':False,'search_rerun':False,
        'retrieval_rerun':False,'length_penalty':0,'views':list(VIEWS)}
    fingerprint=io.digest(config)
    cp=output/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config: raise ValueError('Resume configuration changed')
    prepared=[]
    for sample in manifest:
        row=by_id[sample['sample_id']]
        if sample!=row['dataset_example'] or not row.get('ok'): raise ValueError('Invalid source row')
        if row.get('config_sha256')!=original.sha(source_config): raise ValueError('Source row config mismatch')
        llms,_=sota.extract_candidates_from_context(row['context']['text'])
        llms=list(dict.fromkeys(llms))[:10]
        if len(llms)!=10: raise ValueError('Expected ten distinct retrieved LLMs')
        products,final_keys,completed_keys=build_expansion(row['results'],
            row['generation']['search_trace']['final_rerank_pool_before_cap'],llms)
        prepared.append((sample,row,products,final_keys,completed_keys))
    print('[CHECK]',len(prepared),'samples; critic candidates min/max/total',
          min(len(p[2]) for p in prepared),max(len(p[2]) for p in prepared),sum(len(p[2]) for p in prepared),flush=True)
    if args.check_only: return
    io.write_json(cp,config)
    scored=[]
    for index,(sample,row,products,final_keys,completed_keys) in enumerate(prepared,1):
        sid=sample['sample_id']; path=output/'scored_cartesian_candidates'/f'{sid}.json'
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['config_sha256']!=fingerprint: raise ValueError('Cached scoring config mismatch')
            scored.append(saved); print('[RESUME]',index,sid,flush=True); continue
        # CriticClient's key omits query: instantiate a fresh client PER SAMPLE.
        client=sota.BundleCriticClient(args.critic_url,timeout=600,required=True,verbose=False)
        nodes=[sota.SearchNode(node_id=r['id'],parent_id='',depth=len(r['tool_tokens']),
            llm=r['llm_token'],tools=tuple(r['tool_tokens']),suffix_ids=[],generator_logprob=0,
            generator_token_count=0,last_action='<SPECIAL_END>',is_complete=True,
            termination_reason='recombined_from_saved_toolset') for r in products]
        started=time.time()
        client.score_nodes(query=row['query'],nodes=nodes,evidence_context=row['context']['text'],stage='final_rerank')
        for candidate,node in zip(products,nodes):
            if node.critic_raw is None or not math.isfinite(node.critic_raw): raise ValueError('Missing/nonfinite critic score')
            candidate['critic_raw']=node.critic_raw; candidate['critic_sigmoid']=node.critic_sigmoid
        saved={'sample_id':sid,'config_sha256':fingerprint,'products':products,
            'original_final_keys':sorted(final_keys),'original_completed_keys':sorted(completed_keys),
            'critic_events':client.events,'elapsed_sec':time.time()-started}
        io.write_json(path,saved)
        scored.append(saved)
        io.write_json(output/'run_status.json',{'completed_samples':len(scored),'expected_samples':len(manifest),'state':'scoring'})
        print(f'[CRITIC ONLY] {index}/{len(prepared)} {sid}; candidates={len(nodes)}; {time.time()-started:.2f}s',flush=True)
    evaluate_views(output,manifest,scored,evaluator,sota)
    io.write_json(output/'run_status.json',{'completed_samples':len(scored),'expected_samples':len(manifest),'state':'complete'})

if __name__=='__main__': main()
