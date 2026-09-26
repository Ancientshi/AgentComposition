"""User-requested development evaluation on validation, not held-out test evidence."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import os,sys,json,subprocess,statistics,fcntl
from pathlib import Path
from collections import defaultdict
W=Path(__file__).resolve().parent;R=W.parent
METHODS=['A_frozen','D_target_n3_s42','D_target_n7_s42','D_target_n15_s42','B_explicit_instruction','C_source_n15_s42']
def read(p):return json.loads(p.read_text())
def report():
 manifest=read(W/'data/manifest.json');x=read(R/'experiments/domain_generalization/frozen/inputs.json');cs={c['token']:c['id'] for c in x['components']};ls={c['token']:c['id'] for c in x['llms']};skills={c['id'] for c in x['components'] if c['type']=='skill'};agents=read(R/'datasets/SkillBench/agents/merge.json');rank=read(R/'datasets/SkillBench/canonical/rankings/merge.json')['rankings'];qs=read(R/'datasets/SkillBench/canonical/questions/merge.json');result={}
 for name in METHODS:
  out=W/'predictions'/name/'val'
  if not (out/'COMPLETE.json').exists():continue
  rows=[]
  for sid in manifest['query_ids']['val']:
   raw=read(out/'per_sample'/f'{sid}.json');gold=[(agents[a]['M'].get('name'),set(agents[a]['T']['tools'])) for a in rank[sid]];pred=[];seen=set()
   for p in raw['results'][:10]:
    try:
     ts=p['tools'];assert isinstance(ts,list) and 1<=len(ts)<=10 and len(set(ts))==len(ts) and all(t in cs for t in ts);key=(p['llm'],tuple(sorted(ts)));assert key not in seen;seen.add(key);pred.append((ls.get(p['llm']),{cs[t] for t in ts}))
    except (AssertionError,TypeError,KeyError):pred.append(None)
   pred+=[None]*(10-len(pred));top=pred[0];sg=[g&skills for _,g in gold if g&skills];ps=top[1]&skills if top else set()
   rows.append({'task_id':qs[sid]['metadata']['task_id'],'sample_id':sid,'Set-F1@1':max(2*len(top[1]&g)/(len(top[1])+len(g)) for _,g in gold) if top else 0.,'Skill-F1@1':max(2*len(ps&g)/(len(ps)+len(g)) for g in sg) if sg else None,'Tool-Hit@10':float(any(p and g<=p[1] for p in pred for _,g in gold)),'Legal@1':float(bool(top) and top[0] is not None)})
  bytask=defaultdict(list)
  for row in rows:bytask[row['task_id']].append(row)
  fields=['Set-F1@1','Skill-F1@1','Tool-Hit@10','Legal@1'];macro={}
  for k in fields:
   vals=[statistics.mean(r[k] for r in rs if r[k] is not None) for rs in bytask.values() if any(r[k] is not None for r in rs)];macro[k]=statistics.mean(vals) if vals else None
  result[name]={'tasks':len(bytask),'queries':len(rows),'task_macro':macro,'per_sample':rows}
 p=W/'PROVISIONAL_VALIDATION.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'caveat':'Validation already used for checkpoint selection; development evidence only. Not independent test performance.','methods':result},indent=2));tmp.replace(p)
 print(json.dumps({k:v['task_macro'] for k,v in result.items()},indent=2),flush=True)
def main():
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 for name in METHODS:
  if not (W/'predictions'/name/'val'/'COMPLETE.json').exists():
   with (W/'logs'/f'provisional.{name}.log').open('w') as log:subprocess.run([sys.executable,'-u',str(W/'infer.py'),'--job',name,'--split','val'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  report()
 print('PROVISIONAL VALIDATION COMPLETE',flush=True)
if __name__=='__main__':main()
