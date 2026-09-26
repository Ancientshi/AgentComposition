#!/usr/bin/env python3
"""Exact final-only reranking of frozen V10/V3 completed pools; no inference."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import contextlib,csv,fcntl,hashlib,json,math,statistics,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT = AC_ROOT
sys.path[:0]=[str(ROOT/'training/generator'),str(ROOT/'experiments/structural_v3')]
import run_infer_table1_compact as base
from derive_and_evaluate import metrics,LABELS,FIELDS
SOURCE=ROOT/'outputs/beam_v10_criticv3_D_validation_20260921'
OUT=ROOT/'outputs/beam_critic_v10_cross_validation_20260921'
BOUNDS=[(1,2),(2,4),(3,5),(4,7),(6,10)]
LAMBDAS=[0.,.5,1.,1.5,2.]
GEN_SHA='97da97b8d7dde0a21792df9968112fd972da69677fc2324cae1a5af347047e37'
CRITIC_SHA='d7eb8069dc30484f97c1b4679b3228bcbc386b10fde2a195e947d5a41fc1c23f'
write=base.write_json

def key(n):return n['llm'],tuple(sorted(set(n['tools'])-{'<TOOL_EMPTY>','<TOOL_SEP>'}))
def zscores(vals):
    mean=sum(vals)/len(vals);std=math.sqrt(sum((v-mean)**2 for v in vals)/len(vals))
    return [(v-mean)/std if std>=1e-8 else 0. for v in vals]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    byid={r['sample_id']:r for r in manifest};assert len(manifest)==len(byid)==510
    evaluator=base.load_module(ROOT/'evaluation/reference/exp3/evaluate_ranked_recall_baseline4.py','cross_eval')
    sota=base.load_module(ROOT/'inference/run_infer_v13_batch_eval_sota.py','cross_sota')
    config={'bounds':BOUNDS,'critic_weights':LAMBDAS,'n':len(manifest),
            'positive_weight_formula':'lambda*z(critic_raw) + 0.15*z(generator_avg_logprob) - 0.1*len(tools)',
            'zero_weight_formula':'generator_avg_logprob - 0.1*len(tools)',
            'normalization':'population standard deviation within entire pre-cap completed pool; std<1e-8 gives zero',
            'positive_tie_break':'descending score, critic raw, generator average; canonical key for exact ties',
            'zero_tie_break':'descending generator-only score, generator average, ascending node ID; no critic',
            'note':'lambda=0 is the original generator-only baseline, not the continuous hybrid lambda=0 endpoint',
            'search_critic_enabled':False,'pool':'full pre-cap completed candidates, never just old top10',
            'source':str(SOURCE),'source_manifest_sha256':base.sha(manifest),'generator_sha256':GEN_SHA,
            'critic_sha256':CRITIC_SHA,'runner_sha256':base.file_sha(__file__),'sota_sha256':base.file_sha(sota.__file__),
            'primary_selection_metric':'RDCR@10','legacy_pure_critic_result':'separate control; not a lambda=0.5 cell'}
    if (OUT/'config.json').exists():assert json.loads((OUT/'config.json').read_text())==config
    write(OUT/'config.json',config);write(OUT/'source_data_audit.json',json.loads((SOURCE/'data_audit.json').read_text()))
    aggregate=[];audit=[]
    for bi,(lo,hi) in enumerate(BOUNDS):
        source=SOURCE/f'beam_{lo}_{hi}';configs=[]
        for d in ([source] if bi<2 else [SOURCE/f'beam_{lo}_{hi}_shard{s}of2' for s in range(2)]):
            c=json.loads((d/'config.json').read_text());configs.append(c)
            assert c['generator_sha256']==GEN_SHA and c['critic_model']['weights_sha256']==CRITIC_SHA
            assert c['settings']['beam_min_size']==lo and c['settings']['beam_max_size']==hi
            assert c['sota_sha256']==config['sota_sha256']
        fingerprints={base.sha(c) for c in configs}
        cells={l:OUT/f'beam_{lo}_{hi}_lambda_{l:g}' for l in LAMBDAS}
        for d in cells.values():d.mkdir(exist_ok=True)
        per={l:[] for l in LAMBDAS};oldmetrics=[];seen=[];poolcounts=[];diff=0.;scorediff=0.;digest=hashlib.sha256()
        with contextlib.ExitStack() as stack:
            handles={l:stack.enter_context((d/'results.jsonl').open('w')) for l,d in cells.items()}
            f=stack.enter_context((source/'results.jsonl').open('rb'))
            for line in f:
                digest.update(line);r=json.loads(line);sample=r['dataset_example'];sid=sample['sample_id'];seen.append(sid)
                assert r['ok'] and r['config_sha256'] in fingerprints
                expected_sample={k:v for k,v in byid[sid].items() if k!='context'}
                assert sample==expected_sample
                trace=r['generation']['search_trace'];assert trace['search_critic_enabled'] is False
                assert all(e['stage']=='final_rerank' and not e.get('error') for e in r['generation']['critic_api_events'])
                pool=trace['final_rerank_pool_before_cap'];assert len(pool)==trace['final_rerank_candidate_count']
                assert pool and len({key(n) for n in pool})==len(pool)
                assert all(n['is_complete'] and math.isfinite(n['critic_raw']) and math.isfinite(n['generator_avg_logprob']) for n in pool)
                pure=sorted(pool,key=lambda n:(-n['critic_raw'],key(n)))[:10]
                assert [(p['llm_token'],tuple(sorted(set(p['tool_tokens'])))) for p in r['results']]==[key(n) for n in pure]
                oldmetrics.append(metrics(r));poolcounts.append(len(pool))
                cz=zscores([n['critic_raw'] for n in pool]);gz=zscores([n['generator_avg_logprob'] for n in pool])
                for lam in LAMBDAS:
                    nodes=[SimpleNamespace(**n) for n in pool]
                    if lam==0:sota.assign_generator_only_scores(nodes,length_penalty=.1)
                    else:sota.assign_search_scores(nodes,mode='hybrid',critic_weight=lam,generator_weight=.15,length_penalty=.1)
                    scores=[n['generator_avg_logprob']-.1*len(n['tools']) if lam==0 else lam*c+.15*g-.1*len(n['tools']) for n,c,g in zip(pool,cz,gz)]
                    scorediff=max(scorediff,max(abs(n.search_score-s) for n,s in zip(nodes,scores)));assert scorediff<1e-12
                    if lam==0:indices=sorted(range(len(pool)),key=lambda i:(-scores[i],-pool[i]['generator_avg_logprob'],pool[i]['node_id']))[:10]
                    else:indices=sorted(range(len(pool)),key=lambda i:(-scores[i],-pool[i]['critic_raw'],-pool[i]['generator_avg_logprob'],key(pool[i])))[:10]
                    results=[]
                    for rank,i in enumerate(indices,1):
                        n=pool[i];results.append({'rank':rank,'node_id':n['node_id'],'llm_token':n['llm'],'tool_tokens':n['tools'],
                            'strict_text':' '.join([n['llm'],'<TOOL_SEP>',*n['tools'],'<SPECIAL_END>']),
                            'final_score':scores[i],'critic_raw':n['critic_raw'],'generator_avg_logprob':n['generator_avg_logprob'],
                            'critic_z':cz[i],'generator_z':gz[i]})
                    record={'ok':True,'dataset_example':sample,'results':results,'beam':[lo,hi],'lambda':lam,
                            'available_pool_count':len(pool),'config_sha256':base.sha(config),'source_record_config_sha256':r['config_sha256']}
                    m=metrics(record);ref=evaluator.evaluate_one(record)
                    diff=max(diff,max(abs(m[k]-ref[f]) for k,f in zip(LABELS,FIELDS)));assert diff<1e-12
                    per[lam].append({'sample_id':sid,**m});handles[lam].write(json.dumps(record,ensure_ascii=False)+'\n')
                if len(seen)%100==0:
                    write(OUT/'status.json',{'state':'running','completed_beams':bi,'current_beam':[lo,hi],'current_queries':len(seen),'total_queries':510})
                    print('[PROGRESS]',lo,hi,len(seen),flush=True)
        assert seen==[r['sample_id'] for r in manifest]
        old=json.loads((source/'metrics.json').read_text())['raw']
        pure_diff=max(abs(statistics.fmean(m[k] for m in oldmetrics)-old[k]) for k in LABELS);assert pure_diff<1e-12
        for lam,rows in per.items():
            raw={k:statistics.fmean(r[k] for r in rows) for k in LABELS}
            result={'beam':[lo,hi],'lambda':lam,'n':len(rows),'raw':raw,'percent':{k:v*100 for k,v in raw.items()}}
            write(cells[lam]/'metrics.json',result);write(cells[lam]/'per_sample_metrics.json',rows);aggregate.append(result)
        audit.append({'beam':[lo,hi],'n':len(seen),'source_results_sha256':digest.hexdigest(),'source_config_sha256':sorted(fingerprints),
                      'pool_size_min':min(poolcounts),'pool_size_max':max(poolcounts),'source_pure_D_reproduction_difference':pure_diff,
                      'official_score_max_difference':scorediff,'official_evaluator_max_difference':diff})
        write(OUT/'audit.json',audit);print('[BEAM COMPLETE]',lo,hi,flush=True)
    best=max(aggregate,key=lambda r:r['raw']['RDCR@10'])
    write(OUT/'SUMMARY.json',{'state':'complete','cells':aggregate,'best_by_RDCR':{'beam':best['beam'],'lambda':best['lambda'],'RDCR@10':best['percent']['RDCR@10']}})
    with (OUT/'results_percent.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['beam','lambda','n']+LABELS)
        for r in aggregate:w.writerow([str(tuple(r['beam'])),r['lambda'],r['n']]+[r['percent'][k] for k in LABELS])
    lines=['# V10 + critic V3: beam × critic-weight sensitivity','',
           f"All 25 cells complete, 510 validation queries per cell. Best observed RDCR@10: beam {tuple(best['beam'])}, lambda {best['lambda']:g}, {best['percent']['RDCR@10']:.4f}%.",'',
           '## RDCR@10 (%)','', '| Critic weight | (1,2) | (2,4) | (3,5) | (4,7) | (6,10) |','|---|---:|---:|---:|---:|---:|']
    for lam in LAMBDAS:lines.append('| '+f'{lam:g}'+' | '+' | '.join(f"{next(r for r in aggregate if r['beam']==list(b) and r['lambda']==lam)['percent']['RDCR@10']:.4f}" for b in BOUNDS)+' |')
    lines+=['','For positive lambda, score = lambda × z(critic raw) + 0.15 × z(generator average log probability) − 0.1 × tool count. Z-scores use the full completed pool before any final cap.',
            'At lambda=0, use the original generator-only baseline: generator average log probability − 0.1 × tool count. This is explicitly a separate no-critic baseline, not the continuous endpoint of the hybrid formula.',
            'Critic never affects search. All completed pools and V3 scores are reused exactly; every cell selects a fresh top10 from the full pool. Original pure-critic D results are retained separately and are not identified with lambda=0.5.',
            'All source D rankings and metrics were reproduced. Score calculations and seven metrics agree with original implementations within 1e-12. Fixed K=10, missing ranks zero; all 510 queries included.',
            'Validation excludes generator train, fixed test100, critic train and critic validation query/qid overlap. One highest-ranked available reference per query was fixed before inference. Historical cached contexts may contain earlier gold injection. These validation scores are not test100 scores.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'status.json',{'state':'complete','cells':25,'queries_per_cell':510})
    print('[COMPLETE]',json.dumps({'beam':best['beam'],'lambda':best['lambda'],'RDCR':best['percent']['RDCR@10']}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(OUT/'status.json',{'state':'failed','error':repr(exc)});raise
