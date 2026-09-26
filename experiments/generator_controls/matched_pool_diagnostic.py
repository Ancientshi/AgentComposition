from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json,pathlib,statistics,random,importlib.util
from analyze_results import independent,key,bootstrap
P=pathlib.Path;ROOT=P(__file__).resolve().parent
src=ROOT.parent/'training/generator/final_results/outputs/table1_ours_v10_compact_fixedtest100/results.jsonl'
if not src.exists():src=ROOT.parent/'outputs/table1_ours_v10_compact_fixedtest100/results.jsonl'
histpath=ROOT.parent/'dual_cf_component_ret_20260921/cached_inputs.json'
if not histpath.exists():histpath=ROOT.parent/'outputs/component_ret_dual_cf_cartesian_seed42_n100_20260921/cached_inputs.json'
history={r['qid']:r for r in json.load(open(histpath))['rows']}
records=[]
for line in src.open():
 r=json.loads(line);q=r['dataset_example']['qid'];h=history[q];llms=[x['token'] for x in h['llms'][:10]]
 gen={}
 for n in r['generation']['search_trace']['final_rerank_pool_before_cap']:
  t=tuple(sorted(set(n['tools'])));s=n['generator_avg_logprob']-.1*len(t);gen[t]=max(gen.get(t,-float('inf')),s)
 gs=sorted(gen,key=lambda t:(-gen[t],t));hs=list(dict.fromkeys(tuple(sorted(set(b['tool_tokens']))) for b in h['bundles'][:5] if 1<=len(set(b['tool_tokens']))<=6));b=min(len(hs),len(gs));assert b>=1
 cached=json.load(open(ROOT/'results/scores'/f'{q}.json'));scores={key(c):s for c,s in zip(cached['candidates'],cached['scores'])}
 for name,ts in [('matched_history_v4',hs[:b]),('matched_generator_v4',gs[:b])]:
  cs=[{'llm':l,'tools':list(t)} for t in ts for l in llms];assert len(cs)==10*b and len({key(c) for c in cs})==len(cs)
  pool=[{**c,'score':scores[key(c)]} for c in cs];assert all(c['score'] is not None for c in pool)
  selected=sorted(pool,key=lambda c:(-c['score'],key(c)))[:10]
  pred={'qid':q,'method':name,'dataset_example':r['dataset_example'],'results':selected,'pool_count':len(pool),'bundle_count':b,'selected_seed_toolsets':ts}
  pred['metrics']=independent(pred);records.append(pred)
by={n:{r['qid']:r for r in records if r['method']==n} for n in ['matched_history_v4','matched_generator_v4']}
qs=sorted(by['matched_history_v4']);assert len(qs)==100
for q in qs:assert by['matched_history_v4'][q]['pool_count']==by['matched_generator_v4'][q]['pool_count']
rng=random.Random(20260921);indices=[[rng.randrange(100) for _ in range(100)] for _ in range(10000)]
report={'status':'post-hoc diagnostic, not primary or a tuned replacement','n':100,'policy':'Equal B unique toolsets per query × same LLM10; historical retrieval order versus best generator-only path score. No V4 or reference used for seed preselection. Same frozen V4 final scores.','equal_final_pool_counts':True,'total_inference_cost_matched':False,'results':{n:{'mean_pool_count':statistics.fmean(r['pool_count'] for r in rs.values()),'metrics':{k:statistics.fmean(r['metrics'][k] for r in rs.values()) for k in records[0]['metrics']}} for n,rs in by.items()},'generator_minus_history_ci':{k:bootstrap([by['matched_generator_v4'][q]['metrics'][k]-by['matched_history_v4'][q]['metrics'][k] for q in qs],indices) for k in ['RDCR@10','RDCP@10','CR-Hit@10','Exact-Config-Hit@10']}}
(ROOT/'matched_pool_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));(ROOT/'matched_pool_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
