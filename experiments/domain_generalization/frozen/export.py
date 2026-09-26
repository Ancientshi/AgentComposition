from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json,hashlib
from pathlib import Path
R=Path(str(AC_ROOT));O=R/'outputs/SkillBench_FROZEN_97';W=R/'experiments/domain_generalization/frozen'
x=json.loads((W/'inputs.json').read_text());cs={c['token']:c['id'] for c in x['components']};ls={c['token']:c['id'] for c in x['llms']};methods=['ours']+json.loads((O/'config.json').read_text())['models']
for m in methods:
 rows=[]
 for q in x['queries']:
  r=json.loads((O/m/'per_sample'/f"{q['sample_id']}.json").read_text());pred=[];seen=set()
  for rank,c in enumerate(r['results'][:10],1):
   try:
    lm=c['llm'];ts=c['tools'];assert isinstance(ts,list) and 1<=len(ts)<=10 and len(ts)==len(set(ts)) and all(t in cs for t in ts)
    key=(lm,tuple(sorted(ts)));assert key not in seen;seen.add(key)
    pred.append({'rank':rank,'M':{'name':ls[lm]} if lm in ls else {},'T':{'tools':sorted(cs[t] for t in ts)},'valid_set':True,'valid':lm in ls,'raw_llm_field':lm})
   except (AssertionError,KeyError,TypeError):pred.append({'rank':rank,'valid':False,'raw':c})
  rows.append({'query_id':q['sample_id'],'query':q['query'],'predictions':pred})
 (O/m/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
sourcefiles=[W/'run.py',W/'evaluate.py',W/'audit.py',W/'export.py',W/'serve_extended.py',W/'inputs.json',R/'inference/run_infer_v13_batch_eval_sota.py',R/'training/generator/run_infer_table1_compact.py',R/'training/generator/compact_context.py',Path(str(AC_ROOT / 'training/critic/stage1/compact_input.py'))]
(O/'CODE_PROVENANCE.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sourcefiles},indent=2))
print('Exported original M/T prediction schema for all four methods')
