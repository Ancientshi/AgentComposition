"""Freeze task splits and capped positives before running any new test inference."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json, random, hashlib, re, sys
from pathlib import Path
from collections import defaultdict
R=Path(str(AC_ROOT)); W=R/'experiments/domain_generalization/quickadapt'; D=W/'data'
sys.path.insert(0,str(R/'training/generator'))
from compact_context import TASK, encode_supervision, query_key
from transformers import AutoTokenizer

def read(p): return json.loads(p.read_text())
def write(p,x): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def lines(p):
 with p.open() as f:
  for s in f:
   if s.strip(): yield json.loads(s)
def jsonl(p,rows): p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 assert not D.exists(), 'Refusing to replace frozen experiment data'
 D.mkdir(parents=True)
 src=R/'datasets/SkillBench';qs=read(src/'canonical/questions/merge.json');rank=read(src/'canonical/rankings/merge.json')['rankings'];agents=read(src/'agents/merge.json')
 prepared={q['sample_id']:q for q in read(R/'outputs/SkillBench_FROZEN_97/prepared.json')};x=read(R/'experiments/domain_generalization/frozen/inputs.json')
 tasks=sorted({v['metadata']['task_id'] for v in qs.values()});assert len(tasks)==79 and len(qs)==97
 random.Random(20260919).shuffle(tasks);splits={'train':tasks[:15],'val':tasks[15:25],'test':tasks[25:]}
 ids={s:sorted(k for k,v in qs.items() if v['metadata']['task_id'] in ts) for s,ts in splits.items()}
 for a,b in [('train','val'),('train','test'),('val','test')]:
  assert not set(splits[a])&set(splits[b]);assert not {query_key(qs[k]['input']) for k in ids[a]}&{query_key(qs[k]['input']) for k in ids[b]}
 # Audit normalized exact overlap against all v10 train AND validation queries.
 target_queries={query_key(v['input']) for v in qs.values()};overlap=[];source_groups=defaultdict(list);source_paths=[]
 for split in ['train','valid']:
  p=R/f'dataset/generative_v10_compact/sft_{split}.jsonl';source_paths.append(p)
  for row in lines(p):
   if query_key(row['query']) in target_queries: overlap.append({'qid':row['qid'],'split':split})
   if split=='train':source_groups[row['qid']].append(row)
 assert not overlap, 'Source/target exact query overlap: resolve before proceeding'
 assert all(q['prompt'].startswith('### Task:\n'+TASK+'\n\n### Context:\n') for q in prepared.values())
 tok=AutoTokenizer.from_pretrained(R/'checkpoints/generative_v10_compact',local_files_only=True)
 lm={z['id']:z['token'] for z in x['llms']};tool={z['id']:z['token'] for z in x['components']}
 def target_rows(selected,max_queries,max_targets):
  rows=[]
  for task in selected:
   sids=sorted(k for k in qs if qs[k]['metadata']['task_id']==task)[:max_queries]
   for sid in sids:
    seen=set();n=0
    for aid in rank[sid]:
     ag=agents[aid]
     if not ag['M']:continue
     target=lm[ag['M']['name']]+' <TOOL_SEP> '+' '.join(sorted(tool[t] for t in ag['T']['tools']))+' <SPECIAL_END>'
     if target in seen:continue
     seen.add(target);q=prepared[sid];assert q['query']==qs[sid]['input']
     rows.append({'task_id':task,'sample_id':sid,'agent_id':aid,'query':q['query'],'target':target,'prompt':q['prompt'],**encode_supervision(q['prompt'],target,tok,8192)});n+=1
     if n>=max_targets:break
  return rows
 val=target_rows(splits['val'],10000,10000);jsonl(D/'val.jsonl',val)
 jobs=[];subsets={}
 for seed in [42,43,44]:
  order=splits['train'].copy();random.Random(seed).shuffle(order);subsets[str(seed)]={}
  for n in [3,7,15]:
   name=f'D_target_n{n}_s{seed}';rows=target_rows(order[:n],2,2);assert len({r['task_id'] for r in rows})==n
   jsonl(D/f'{name}.jsonl',rows);subsets[str(seed)][str(n)]=order[:n]
   jobs.append({'name':name,'group':'D','seed':seed,'tasks':n,'queries':len({r['sample_id'] for r in rows}),'pairs':len(rows)})
  # Source rehearsal control: same template already; no target supervised answers.
  order_src=sorted(source_groups);random.Random(seed).shuffle(order_src);sr=[]
  for qid in order_src[:15]:
   for row in source_groups[qid][:2]:sr.append({**row,'task_id':'source:'+str(qid),'sample_id':'source:'+str(qid)})
  name=f'C_source_n15_s{seed}';jsonl(D/f'{name}.jsonl',sr);jobs.append({'name':name,'group':'C','seed':seed,'tasks':15,'queries':15,'pairs':len(sr)})
 manifest={'split_seed':20260919,'task_ids':splits,'query_ids':ids,'subsets':subsets,'jobs':jobs,'test_manifest':[prepared[k] for k in ids['test']],'val_manifest':[prepared[k] for k in ids['val']]}
 write(D/'manifest.json',manifest)
 write(D/'audit.json',{'tasks':{s:len(v) for s,v in splits.items()},'queries':{s:len(v) for s,v in ids.items()},'v10_exact_normalized_query_overlap':overlap,'near_duplicate_or_pretraining_contamination':'not established by exact-match audit','outer_template_identical_to_v10':True,'validation_pairs':len(val),'input_hashes':{str(p):sha(p) for p in source_paths+[src/'canonical/questions/merge.json',src/'canonical/rankings/merge.json',R/'outputs/SkillBench_FROZEN_97/prepared.json']},'data_hashes':{p.name:sha(p) for p in D.glob('*.jsonl')}})
 print(json.dumps(read(D/'audit.json'),indent=2),flush=True)
if __name__=='__main__':main()
