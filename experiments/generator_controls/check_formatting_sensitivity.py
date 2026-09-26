from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json,pathlib
P=pathlib.Path(str(AC_ROOT / 'experiments/generator_controls/exposure'))
norm=lambda s:' '.join(s.replace('<Tool_','<TOOL_').split())
rs=[json.loads(x) for x in (P/'per_query.jsonl').read_text().splitlines()]
out={}
for name in ['generator_train_targets','pipeline_train_supervised','pipeline_train_conservative','pipeline_train_valid_conservative','partii_full_dataset_catalog']:
 idx=json.loads((P/(name+'.json')).read_text());bs={tuple(sorted({norm(t) for t in b})) for b in idx['bundles']};cs={(norm(l),tuple(sorted({norm(t) for t in b}))) for l,b in idx['configurations']}
 ub=[];uc=[]
 for r in rs:
  g=r['target'];b=tuple(sorted({norm(t) for t in g['tools']}));c=(norm(g['llm']),b)
  if b not in bs:ub.append(r['qid'])
  if c not in cs:uc.append(r['qid'])
 out[name]={'unseen_bundle_n':len(ub),'unseen_configuration_n':len(uc),'unseen_bundle_qids':ub,'unseen_configuration_qids':uc}
(P/'formatting_sensitivity.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:{n:v for n,v in r.items() if n.endswith('_n')} for k,r in out.items()},indent=2))
