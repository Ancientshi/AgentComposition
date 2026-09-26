from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json,statistics,hashlib,math,sys,re
from pathlib import Path
ROOT=Path(str(AC_ROOT));OUT=ROOT/'outputs/SkillBench_FROZEN_97';D=ROOT/'datasets/SkillBench'
NORMALIZED='--extract-identifiers' in sys.argv
SUFFIX='_identifier_extraction' if NORMALIZED else ''
def token(value,allowed):
 if isinstance(value,str) and value in allowed:return value
 if NORMALIZED and isinstance(value,str):
  found=re.findall(r'<<[^<>\n]+>>|<LLM_[^<>\n]+>',value)
  if len(found)==1 and found[0] in allowed:return found[0]
 return None
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
x=read(ROOT/'experiments/domain_generalization/frozen/inputs.json');cs={c['token']:c['id'] for c in x['components']};ls={c['token']:c['id'] for c in x['llms']}
agents=read(D/'agents/merge.json');labels=read(D/'canonical/rankings/merge.json')['rankings'];qs=x['queries'];summary={}
methods=['ours','gpt-5.4-2026-03-05','gpt-5.4-mini-2026-03-17','gpt-5.4-nano-2026-03-17']
if '--gpt-only' in sys.argv:methods=methods[1:]
for method in methods:
 rows=[];invalid=0;invalid_llm=0;missing=0;format_errors=0;usage={};lengths={}
 for q in qs:
  sid=q['sample_id'];r=read(OUT/method/'per_sample'/f'{sid}.json');assert r['sample_id']==sid
  if method!='ours':
   assert r['api']['returned_model']==method
   for k,v in (r['api'].get('usage') or {}).items():
    if isinstance(v,(float,int)):usage[k]=usage.get(k,0)+v
  gold=[(agents[a]['M'].get('name'),set(agents[a]['T']['tools'])) for a in labels[sid]]
  assert any(m for m,t in gold),'No knownLLM positive; fullconfig metric denominator needs exclusion'
  missing+=max(0,10-len(r.get('results') or []));format_errors+=int(bool(r.get('format_error')))
  predicted=[];seen=set()
  for c in (r.get('results') or [])[:10]:
   try:
    assert isinstance(c,dict)
    m=token(c.get('llm',''),ls);ts=c['tools'];assert isinstance(ts,list) and 1<=len(ts)<=10
    ts=[token(t,cs) for t in ts];assert all(t in cs for t in ts) and len(set(ts))==len(ts)
    key=(str(c.get('llm')) if m is None else m,tuple(sorted(ts)));assert key not in seen;seen.add(key)
    if m is None:invalid_llm+=1
    predicted.append((ls.get(m),{cs[t] for t in ts}));lengths[len(ts)]=lengths.get(len(ts),0)+1
   except (AssertionError,KeyError,TypeError):predicted.append(None);invalid+=1
  predicted += [None]*(10-len(predicted));e={'sample_id':sid}
  def exact(p,g,full=False):return bool(p) and p[1]==g[1] and (not full or bool(g[0]) and p[0]==g[0])
  def covers(p,g,full=False):return bool(p) and g[1]<=p[1] and (not full or bool(g[0]) and p[0]==g[0])
  for label,fn,full in [('SetExact',exact,False),('AgentExact',exact,True),('SetCoverage',covers,False),('AgentCoverage',covers,True)]:
   ranks=[i+1 for i,p in enumerate(predicted) if any(fn(p,g,full) for g in gold)]
   for k in [1,5,10]:e[f'{label}-Hit@{k}']=float(bool(ranks) and ranks[0]<=k)
   e[f'{label}-MRR@10']=1/ranks[0] if ranks else 0
  rec=[max((len(p[1]&g[1])+int(p[0]==g[0]))/(len(g[1])+1) for g in gold if g[0]) if p else 0 for p in predicted]
  e['Legacy-RDCR@10']=sum(v/math.log2(i+2) for i,v in enumerate(rec))/sum(1/math.log2(i+2) for i in range(10))
  p=predicted[0]
  if p:
   def f1(g):return 2*len(p[1]&g[1])/(len(p[1])+len(g[1]))
   best=max(gold,key=f1);e['Set-F1@1']=f1(best);e['Set-Precision@1']=len(p[1]&best[1])/len(p[1]);e['Set-Recall@1']=len(p[1]&best[1])/len(best[1])
   e['Legacy-ToolR@1']=max(len(p[1]&g[1])/len(g[1]) for g in gold)
   e['Legacy-CompR@1']=max((len(p[1]&g[1])+int(p[0]==g[0]))/(len(g[1])+1) for g in gold if g[0])
  else:
   for k in ['Set-F1@1','Set-Precision@1','Set-Recall@1','Legacy-ToolR@1','Legacy-CompR@1']:e[k]=0
  rows.append(e)
 metrics={k:statistics.fmean(r[k] for r in rows) for k in rows[0] if k!='sample_id'}
 summary[method]={'n':len(rows),'metrics':metrics,'invalid_component_or_duplicate_slots':invalid,'invalid_llm_slots_with_valid_sets':invalid_llm,'missing_slots':missing,'format_error_responses':format_errors,'predicted_set_lengths':lengths,'usage':usage}
 write(OUT/method/('evaluation'+SUFFIX+'.json'),{'summary':summary[method],'per_sample':rows})
if '--gpt-only' in sys.argv:
 write(OUT/('GPT_ONLY_metrics'+SUFFIX+'.json'),summary);print(json.dumps(summary,indent=2));sys.exit(0)
write(OUT/('metrics'+SUFFIX+'.json'),summary)
lines=['# SkillBench — frozen inference comparison','', 'Protocol: '+('secondary exact delimited identifier extraction; no spelling repair, inference, or extra API calls' if NORMALIZED else 'primary literal identifier matching; malformed LLM fields do not invalidate valid component sets')+'.', '', '97 unique queries from 1,000 successful trajectories. No training. All methods share 19 exact LLM identifiers and 175 components. Each query has up to 10 equally successful observed configurations as positives. Invalid/missing/duplicate ranks count as misses.','', '| Method | Set exact Hit@1 | Set exact Hit@10 | Agent exact Hit@10 | Set F1@1 | Set coverage Hit@10 | Agent coverage Hit@10 |','|---|---:|---:|---:|---:|---:|---:|']
for m,r in summary.items():
 met=r['metrics'];lines.append('| '+m+' | '+' | '.join(f'{100*met[k]:.2f}%' for k in ['SetExact-Hit@1','SetExact-Hit@10','AgentExact-Hit@10','Set-F1@1','SetCoverage-Hit@10','AgentCoverage-Hit@10'])+' |')
lines+=['','Ours: frozen v10 generator, beam1–2, completed sets crossed with all19 LLMs, frozen criticv3 raw-score reranking. Maximum set size expanded6→10; original critic serialization and512-token budget retained. GPT: first returned response, no formatting retries, temperature0, reasoning none.','', 'This is an empirical closed-inventory cross-domain diagnostic. Gold configurations are observed successful components, not proven minimal configurations or counterfactually validated recommendations. The97 query variants belong to79 tasks and are not97 independent tasks; no confidence claim is made. Exact LLM aliases are preserved, and unresolved LLM gold labels are excluded only from full-agent matching. Shared generic harness actions can raise set-overlap scores. All gold scores equal1.0; top10 selection uses deterministic hashes, not a measured performance ordering.']
(OUT/('RESULTS'+SUFFIX+'.md')).write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
