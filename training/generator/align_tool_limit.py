"""One-time finalization of prepared data; preserve split and frozen test."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import pathlib,json,collections,statistics
from compact_context import components,query_key
from prepare_sft import sha_file,read_rows
root=pathlib.Path(str(AC_ROOT / 'datasets/generative_v10_compact'));rp=root/'preparation_report.json';report=json.loads(rp.read_text())
assert 'tool_limit_finalization' not in report
previous=dict(report['outputs_sha256']);removed=[];stats=collections.defaultdict(list)
for split in ['train','valid']:
 p=root/f'sft_{split}.jsonl';tmp=p.with_suffix('.tmp');groups=set();n=0
 with tmp.open('w') as f:
  for _,x in read_rows(p):
   if len(components(x['target'])[1])>6:removed.append(x);continue
   f.write(json.dumps(x,ensure_ascii=False)+'\n');groups.add(query_key(x['query']));n+=1
   stats[split+'_prompt'].append(x['compression']['prompt_tokens']);stats[split+'_total'].append(len(x['input_ids']));stats[split+'_supervised'].append(sum(t!=-100 for t in x['labels']))
 tmp.replace(p);report['counts'][split+'_rows']=n;report['unique_queries'][split]=len(groups);report['outputs_sha256'][split]=sha_file(p)
byfile=collections.defaultdict(set)
for x in removed:byfile[x['provenance']['cached_context_source']].add(x['provenance']['source_line'])
removed_stale=0
for name,lines in byfile.items():
 for line,x in read_rows(name):
  if line in lines:removed_stale+=not x['completion'].startswith(x['target'])
with (root/'excluded.jsonl').open('a') as f:
 for x in removed:f.write(json.dumps({'reason':'tool_count_limit','source':x['provenance']['cached_context_source'],'line':x['provenance']['source_line'],'qid':x['qid'],'query_hash':x['query_hash']})+'\n')
report['counts']['excluded_tool_count_limit']=len(removed);report['counts']['stale_completion_rebuilt']-=removed_stale
report['budgets']['max_tools']=6
report['token_stats']={k:{'min':min(v),'median':statistics.median(v),'max':max(v),'mean':statistics.mean(v)} for k,v in stats.items()}
report['tool_limit_finalization']={'max_tools':6,'removed_rows':len(removed),'previous_outputs_sha256':previous,'script_sha256':sha_file(__file__)}
report['first256_token_comparison']['note']='Comparison from initial preparation before final tool-count filtering.'
rp.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({'removed':len(removed),'counts':report['counts']}))
