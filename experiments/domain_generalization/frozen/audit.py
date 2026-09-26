from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import hashlib,json,collections
from pathlib import Path
R=Path(str(AC_ROOT));O=R/'outputs/SkillBench_FROZEN_97';D=R/'datasets/SkillBench';W=R/'experiments/domain_generalization/frozen'
def read(p):return json.loads(p.read_text())
x=read(W/'inputs.json');prepared=read(O/'prepared.json');models=['ours']+read(O/'config.json')['models'];ids=[q['sample_id'] for q in x['queries']]
assert len(ids)==len(set(ids))==97
assert len({q['query'] for q in x['queries']})==97
assert len(x['components'])==175 and len(x['llms'])==19
assert all(q['query'] in p['prompt'] and p['meta']['prompt_tokens']<=7600 for q,p in zip(x['queries'],prepared))
assert len({v['token'] for v in x['llms']+x['components']})==194
checks=read(D/'checksums.json')
for p,h in checks.items():assert hashlib.sha256((D/p).read_bytes()).hexdigest()==h,p
report={'dataset_files_verified':len(checks),'queries':97,'components':175,'known_llm_identifiers':19,'results':{}}
for m in models:
 files=list((O/m/'per_sample').glob('*.json'));assert {p.stem for p in files}==set(ids),(m,len(files))
 item={'responses':len(files),'invalid_format_responses':0,'missing_slots':0}
 for q,p in zip(x['queries'],prepared):
  r=read(O/m/'per_sample'/f"{q['sample_id']}.json");assert r['sample_id']==q['sample_id']
  item['missing_slots']+=max(0,10-len(r['results']))
  if m!='ours':
   assert r['api']['returned_model']==m
   assert p['context'] in r['messages'][1]['content'] and q['query'] in r['messages'][1]['content']
   assert len(r['messages'])==2
   if r.get('format_error'):item['invalid_format_responses']+=1
  else:
   gen=read(O/'ours/generation'/f"{q['sample_id']}.json")
   assert gen['prompt']['text']==p['prompt']
   trace=gen['generation']['search_trace'];assert not trace['search_critic_enabled'] and trace['final_critic_reranking']
   assert len(gen['generation']['critic_api_events'])==1
   sets={tuple(sorted(set(c['tools']))) for c in trace['final_rerank_pool_before_cap']}
   products={(c['llm'],tuple(sorted(c['tools']))) for c in r['all_candidates']}
   assert products=={(m['token'],ts) for m in x['llms'] for ts in sets}
   assert r['results']==sorted(r['all_candidates'],key=lambda c:-c['critic_raw'])[:10]
 report['results'][m]=item
report['status']='PASS'
report['runner_sha256']=hashlib.sha256((W/'run.py').read_bytes()).hexdigest()
(O/'AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
