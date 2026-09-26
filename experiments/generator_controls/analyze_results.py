"""Independent metric audit, paired uncertainty and exposure-stratified reports."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import collections,csv,hashlib,json,math,pathlib,random,statistics
P=pathlib.Path;ROOT=P(__file__).resolve().parent;RESULT=ROOT/'results'
LABELS={'baseline4_rag_llama_top10_seed42_n100':'RAG-Llama','baseline5_rag_gpt_5_4_2026_03_05_top10_seed42_n100':'RAG-GPT-5.4','baseline5_rag_gpt_5_4_mini_2026_03_17_top10_seed42_n100':'RAG-GPT-5.4-mini','baseline5_rag_gpt_5_4_nano_2026_03_17_top10_seed42_n100':'RAG-GPT-5.4-nano','baseline5_rag_gpt_5_6_luna_top10_seed42_n100':'RAG-Luna','baseline5_rag_gpt_5_6_sol_top10_seed42_n100':'RAG-Sol','baseline5_rag_gpt_5_6_terra_top10_seed42_n100':'RAG-Terra','baseline_siliconflow_rag_Pro_moonshotai_Kimi_K2_6_top10_seed42_n100':'RAG-Kimi','baseline_siliconflow_rag_Qwen_Qwen3_8_27B_top10_seed42_n100':'RAG-Qwen','baseline_siliconflow_rag_deepseek_ai_DeepSeek_V4_Pro_top10_seed42_n100':'RAG-DeepSeek','history_full_retrieval':'History×LLM / retrieval order','history_full_v4':'History×LLM + V4','history_legal_retrieval':'History(1–6 tools)×LLM / retrieval order','history_legal_v4':'History(1–6 tools)×LLM + V4','ours_no_expansion_v4':'Ours + V4 / no expansion','ours_cartesian_v4':'Ours + V4 / Cartesian','ours_legacy_seeds_v4':'Legacy upstream toolsets + V4'}
def label(n):
 for suffix,tail in [('_original',' / original'),('_v4',' + V4')]:
  if n.endswith(suffix) and n[:-len(suffix)] in LABELS:return LABELS[n[:-len(suffix)]]+tail
 return LABELS.get(n,n)
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def key(c):return c['llm'],tuple(sorted(set(c['tools'])))
def independent(pred):
 import re
 target=pred['dataset_example']['target'].split('<SPECIAL_END>')[0].split('Explanation:')[0]
 lm=re.findall(r'<LLM_[^<>\n\r]+>',target)[0];ts=set(re.findall(r'<<[^<>\n\r]+>>|<TOOL_[^<>\n\r]+>',target))-{'<TOOL_SEP>','<TOOL_EMPTY>'};gc=ts|{lm}
 tr=[];cr=[];cp=[];tp=[];cf=[];th=[];ch=[];exact=[];eb=[]
 for c in pred['results'][:10]:
  tools=set(c['tools']);comp=tools|{c['llm']};n=len(gc&comp)
  tr.append(len(tools&ts)/len(ts) if ts else 1.);cr.append(n/len(gc));cp.append(n/len(comp));tp.append(len(tools&ts)/len(tools) if tools else float(not ts));cf.append(2*n/(len(gc)+len(comp)));th.append(ts<=tools);ch.append(c['llm']==lm and ts<=tools);exact.append(c['llm']==lm and ts==tools);eb.append(ts==tools)
 w=[1/math.log2(i+2) for i in range(10)];discount=lambda xs:sum(a*b for a,b in zip(xs,w))/sum(w)
 return {'ToolR@1':tr[0],'Tool-Hit@10':float(any(th)),'CompR@1':cr[0],'CR-Hit@1':float(ch[0]),'CR-Hit@10':float(any(ch)),'CR-MRR@10':next((1/(i+1) for i,x in enumerate(ch) if x),0.),'RDCR@10':discount(cr),'RDCP@10':discount(cp),'RDCF1@10':discount(cf),'CompP@1':cp[0],'ToolP@1':tp[0],'Exact-Config-Hit@10':float(any(exact)),'Exact-Bundle-Hit@10':float(any(eb))}
def aggregate(rs):
 return {'n':len(rs),'metrics':{k:statistics.fmean(r['metrics'][k] for r in rs) for k in rs[0]['metrics']} if rs else {},'mean_pool_count':statistics.fmean(r['pool_count'] for r in rs) if rs else None}
def bootstrap(diffs,indices):
 n=len(diffs);bs=sorted(sum(diffs[i] for i in ids)/n*100 for ids in indices)
 return {'delta_pp':statistics.fmean(diffs)*100,'ci95_pp':[bs[249],bs[9749]],'improved':sum(x>1e-12 for x in diffs),'declined':sum(x< -1e-12 for x in diffs),'unchanged':sum(abs(x)<=1e-12 for x in diffs)}
def main():
 ps=rows(RESULT/'predictions.jsonl');es=rows(ROOT/'exposure/per_query.jsonl');by=collections.defaultdict(dict)
 for p in ps:assert p['qid'] not in by[p['method']];by[p['method']][p['qid']]=p
 assert all(len(v)==100 for v in by.values())
 maxerr=0.;sortedn=0
 for p in ps:
  assert len(p['results'])==10 and len({key(c) for c in p['results']})==10
  indep=independent(p);maxerr=max(maxerr,max(abs(indep[k]-v) for k,v in p['metrics'].items()))
  if p['method'].endswith('_v4'):
   assert p['results']==sorted(p['results'],key=lambda c:(-c['score'],key(c)));sortedn+=1
 assert maxerr<1e-12
 invariants={}
 for name in LABELS:
  if name+'_original' not in by:continue
  a=by[name+'_original'];b=by[name+'_v4'];assert set(a)==set(b)
  for q in a:
   assert {key(c) for c in a[q]['results']}=={key(c) for c in b[q]['results']}
   assert a[q]['metrics']['CR-Hit@10']==b[q]['metrics']['CR-Hit@10'];assert a[q]['metrics']['Tool-Hit@10']==b[q]['metrics']['Tool-Hit@10']
  invariants[name]=True
 scopes=['generator_train_targets','pipeline_train_supervised','pipeline_train_conservative','pipeline_train_valid_conservative']
 subsets={'all':[e['qid'] for e in es]}
 for scope in scopes:
  for unit in ['bundle','configuration']:
   subsets[scope+'_unseen_'+unit]=[e['qid'] for e in es if not e['exposure'][scope][unit+'_seen']]
 strict='pipeline_train_valid_conservative'
 subsets['strict_unseen_bundle_known_components']=[e['qid'] for e in es if not e['exposure'][strict]['bundle_seen'] and e['all_target_components_seen_in_task_training']]
 subsets['strict_unseen_configuration_known_components']=[e['qid'] for e in es if not e['exposure'][strict]['configuration_seen'] and e['all_target_components_seen_in_task_training']]
 subsets['strict_unseen_bundle_retrievable']=[e['qid'] for e in es if not e['exposure'][strict]['bundle_seen'] and e['exact_bundle_in_retrieved_top5']]
 subsets['strict_unseen_bundle_not_retrieved']=[e['qid'] for e in es if not e['exposure'][strict]['bundle_seen'] and not e['exact_bundle_in_retrieved_top5']]
 subsets['strict_unseen_bundle_known_components_not_retrieved']=[e['qid'] for e in es if not e['exposure'][strict]['bundle_seen'] and e['all_target_components_seen_in_task_training'] and not e['exact_bundle_in_retrieved_top5']]
 subsets['strict_unseen_bundle_nonempty']=[e['qid'] for e in es if not e['exposure'][strict]['bundle_seen'] and len(e['target']['tools'])>=1]
 subsets['strict_unseen_multitool_bundle_known_components']=[e['qid'] for e in es if not e['exposure'][strict]['bundle_seen'] and e['all_target_components_seen_in_task_training'] and len(e['target']['tools'])>=2]
 subsets['strict_unseen_multitool_bundle_known_components_not_retrieved']=[e['qid'] for e in es if not e['exposure'][strict]['bundle_seen'] and e['all_target_components_seen_in_task_training'] and len(e['target']['tools'])>=2 and not e['exact_bundle_in_retrieved_top5']]
 subsets['historical_bundle_absent_top5']=[e['qid'] for e in es if not e['exact_bundle_in_retrieved_top5']]
 subsets['strict_seen_bundle']=[e['qid'] for e in es if e['exposure'][strict]['bundle_seen']]
 subreports={s:{name:aggregate([rs[q] for q in ids]) for name,rs in by.items()} for s,ids in subsets.items()}
 write(ROOT/'subgroups.json',{'query_ids':subsets,'results':subreports})
 sensitivity={}
 for name in LABELS:
  if name+'_original' in by:
   ids=[q for q,r in by[name+'_original'].items() if all(1<=len(c['tools'])<=6 for c in r['results'])]
   sensitivity[name]={'queries':ids,'n':len(ids),'original':aggregate([by[name+'_original'][q] for q in ids]),'v4':aggregate([by[name+'_v4'][q] for q in ids])}
 write(ROOT/'rag_in_range_sensitivity.json',sensitivity)
 names=sorted(by);summary={name:aggregate(list(by[name].values())) for name in names}
 with (ROOT/'all_results_percent.csv').open('w') as f:
  w=csv.writer(f);metrics=list(ps[0]['metrics']);w.writerow(['Method','N','Mean pool candidates']+metrics)
  for n in names:w.writerow([label(n),100,summary[n]['mean_pool_count']]+[summary[n]['metrics'][k]*100 for k in metrics])
 # Pairing is fixed by qid; one shared resampling plan across all full-test comparisons.
 qs=sorted(subsets['all']);rng=random.Random(20260921);indices=[[rng.randrange(100) for _ in range(100)] for _ in range(10000)]
 pairs=[('history_full_v4','history_full_retrieval'),('history_legal_v4','history_legal_retrieval'),('ours_cartesian_v4','ours_no_expansion_v4'),('ours_cartesian_v4','history_full_v4'),('ours_cartesian_v4','history_legal_v4')]
 pairs += [(name+'_v4',name+'_original') for name in LABELS if name+'_original' in by]
 pairs += [('ours_cartesian_v4',name+'_v4') for name in LABELS if name+'_original' in by]
 ci={}
 for a,b in pairs:
  ci[a+' minus '+b]={'n':100,'metrics':{k:bootstrap([by[a][q]['metrics'][k]-by[b][q]['metrics'][k] for q in qs],indices) for k in ['RDCR@10','RDCP@10','CR-Hit@10','CR-MRR@10','Exact-Config-Hit@10']}}
 write(ROOT/'paired_bootstrap.json',{'resamples':10000,'seed':20260921,'interval':'paired query percentile 95%; unadjusted descriptive intervals, no multiple-comparison correction','comparisons':ci})
 # Check the existing V4 historical view rather than substituting it for current V4 seeds.
 oldpath=ROOT.parent/'training/critic/stage2/evaluation/summary.json'
 if not oldpath.exists():oldpath=P(str(AC_ROOT / 'training/critic/stage2/evaluation/summary.json'))
 old=json.loads(oldpath.read_text())['results']['upstream_top10_toolsets']['v4_raw']['metrics']
 legacy=summary['ours_legacy_seeds_v4']['metrics'];reproduction={k:{'old':v,'new':legacy[k],'difference':legacy[k]-v} for k,v in old.items()}
 write(ROOT/'legacy_v4_reproduction.json',reproduction)
 write(ROOT/'FINAL_AUDIT.json',{'passed':True,'queries':100,'method_groups':len(by),'ranked_results':len(ps),'independent_metric_max_error':maxerr,'canonical_v4_sorted_groups':sortedn,'rag_candidate_and_hit_invariance':invariants,'checkpoint_sha256':json.loads((RESULT/'config.json').read_text())['checkpoint_sha256'],'legacy_v4_max_metric_difference':max(abs(x['difference']) for x in reproduction.values())})
 print(json.dumps({'groups':len(by),'max_error':maxerr,'subgroups':{k:len(v) for k,v in subsets.items()},'selected':{n:summary[n] for n in ['history_full_v4','history_legal_v4','ours_no_expansion_v4','ours_cartesian_v4']}},indent=2))
if __name__=='__main__':main()
