"""Score the frozen query manifest, refusing to call unattempted rows model failures."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
from metrics_core import evaluate,skills_candidates
KEYS=['ToolR@1','Tool-Hit@10','CompR@1','CR-Hit@1','CR-Hit@10','CR-MRR@10','RDCR@10','RDCP@10']
MODELS={'GPT-5.4':'gpt-5.4-2026-03-05','GPT-5.4-mini':'gpt-5.4-mini-2026-03-17','GPT-5.4-nano':'gpt-5.4-nano-2026-03-17','GPT-5.6-Luna':'gpt-5.6-luna','GPT-5.6-Terra':'gpt-5.6-terra','GPT-5.6-Sol':'gpt-5.6-sol','DeepSeek-V4-Pro':'deepseek-ai/DeepSeek-V4-Pro','Qwen3.8-27B':'Qwen/Qwen3.8-27B','Kimi-K2.6':'Pro/moonshotai/Kimi-K2.6','Ours':'ours'}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();out=a.directory
 protocol=read(out/'protocol.json');ids=protocol['test_ids'];assert len(ids)==len(set(ids))==19
 gold=read(out/'gold.json');catalog=read(out/'catalog_map.json');assert set(ids)==set(gold)
 zero=evaluate([],gold[ids[0]]);assert all(zero[k]==0 for k in KEYS)
 known=next(g for g in gold[ids[0]] if g['llm']);perfect=evaluate([known]*10,[known]);assert all(abs(perfect[k]-1)<1e-12 for k in KEYS)
 one=evaluate([known],[known]);assert abs(one['RDCR@10']-1/sum(1/math.log2(r+1) for r in range(1,11)))<1e-12
 results={};rows=[];incomplete={};audit={};provenance={};recommendations=[]
 for label,model in MODELS.items():
  files={sid:out/model/'per_sample'/f'{sid}.json' for sid in ids}
  missing=[sid for sid,p in files.items() if not p.exists()]
  if missing:incomplete[label]={'missing':missing,'available':19-len(missing)};continue
  records=[];counts={'n':19,'empty_raw_responses':0,'no_scorable_configuration_queries':0,'format_error_queries':0,'generation_error_queries':0,'transport_error_queries':0,'invalid_or_missing_slots':0,'valid_tools_with_invalid_llm_slots':0};hashes={}
  for sid,p in files.items():
   r=read(p);assert r['sample_id']==sid
   if label=='Ours':assert r['checkpoint_sha256']==protocol['checkpoint_sha256']
   elif not r.get('transport_error'):assert r['api']['returned_model']==model
   cs=skills_candidates(r,catalog['components'],catalog['llms']);assert len(cs)==10
   met={k:evaluate(cs,gold[sid])[k] for k in KEYS};records.append({'sample_id':sid,**met})
   rows.append({'method':label,'sample_id':sid,**met})
   counts['empty_raw_responses']+=int(not r.get('results'))
   counts['no_scorable_configuration_queries']+=int(not any(c is not None for c in cs))
   counts['format_error_queries']+=int(bool(r.get('format_error')))
   counts['generation_error_queries']+=int(bool(r.get('generation_error')))
   counts['transport_error_queries']+=int(bool(r.get('transport_error')))
   counts['invalid_or_missing_slots']+=sum(c is None for c in cs)
   counts['valid_tools_with_invalid_llm_slots']+=sum(c is not None and c['llm'] is None for c in cs)
   hashes[sid]=sha(p);recommendations.append({'method':label,'sample_id':sid,'raw_results':r.get('results',[]),'scored_candidates':cs,'metrics':met,'generation_error':r.get('generation_error'),'format_error':r.get('format_error'),'transport_error':r.get('transport_error')})
  raw={k:statistics.fmean(r[k] for r in records) for k in KEYS};percent={k:100*v for k,v in raw.items()}
  hit_counts={k:sum(r[k] for r in records) for k in ['Tool-Hit@10','CR-Hit@1','CR-Hit@10']}
  for k,n in hit_counts.items():assert float(n).is_integer() and abs(percent[k]-100*n/19)<1e-10
  results[label]={'n':19,'raw':raw,'percent':percent,'hit_counts':hit_counts};audit[label]=counts;provenance[label]=hashes
 write(out/'metrics.json',results);write(out/'evaluation_audit.json',{'expected_n':19,'all_complete':not incomplete,'complete_methods':list(results),'incomplete_methods':incomplete,'query_sets_equal_for_complete_methods':True,'per_method':audit,'test_ids':ids,'test_manifest_sha256':protocol['test_manifest_sha256'],'evaluator_core_sha256':sha(Path(__file__).with_name('metrics_core.py')),'evaluator_sha256':sha(Path(__file__))})
 write(out/'prediction_sha256.json',provenance);write(out/'per_query_metrics.json',rows)
 (out/'recommendations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in recommendations))
 with (out/'results_percent.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['Method','N']+KEYS);w.writeheader();w.writerows({'Method':m,'N':19,**v['percent']} for m,v in results.items())
 lines=['# SkillsBench: fixed 19-query evaluation','','All eight metrics use the 0–100 scale. Every completed method uses the same 19 queries; invalid, duplicate and missing ranks remain in place and score zero. Valid tool sets can receive partial credit when the LLM identifier is invalid.','', '| Method | N | '+' | '.join(KEYS)+' |','|---|---:|'+'---:|'*len(KEYS)]
 for m,v in results.items():lines.append('| '+m+' | 19 | '+' | '.join(f'{v["percent"][k]:.2f}' for k in KEYS)+' |')
 lines+=['','Train: 63 tasks / 78 queries / 506 resolved pairs. Test: 16 tasks / 19 queries / 109 resolved pairs. One additional unresolved-backbone reference is retained for tool-only matching.','', 'Ours uses the existing final epoch-3 checkpoint (step 192), no critic, beam (1,2), and generator-average-logprob final ranking. The checkpoint and scoring rules were fixed before this run. Seven evaluation queries served as validation queries in the earlier experiment, so this is not an untouched test set.','', 'Cached baseline responses are reused only after exact prompt and model-identifier checks. No response is selected based on quality; no formatting repair calls are made.']
 if incomplete:lines+=['','INCOMPLETE METHODS (no 19-query score is reported):',json.dumps(incomplete,ensure_ascii=False,indent=2)]
 (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 (out/'table_rows.tex').write_text('% All eight metrics scaled 0--100; fixed n=19.\n'+'\n'.join(m+' & '+' & '.join(f'{v["percent"][k]:.2f}' for k in KEYS)+r' \\' for m,v in results.items())+'\n')
 print(json.dumps({'results':results,'incomplete':{k:v['available'] for k,v in incomplete.items()},'audit':audit},indent=2))
if __name__=='__main__':main()
