"""Read-only task-training exposure audit; no ranking and no label-based selection."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import collections,hashlib,importlib.util,itertools,json,pathlib,random,re,sys
P=pathlib.Path;R=P(str(AC_ROOT));B=P(str(AC_ROOT));OUT=R/'experiments/generator_controls/exposure'
sys.path.insert(0,str(B/'training/critic/stage1'));from compact_input import compact_inventory
sys.path.insert(0,str(B/'training/critic/stage2'));from core import parse_target

def rows(p):
 with P(p).open() as f:
  for l in f:
   if l.strip():yield json.loads(l)
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def norm(q):return ' '.join(q.split())
def token(t):return re.sub(r'^<Tool_','<TOOL_',t)
def key(c,canonical=False):
 fun=token if canonical else (lambda x:x)
 return fun(c['llm']),tuple(sorted({fun(t) for t in c['tools'] if t not in {'<TOOL_EMPTY>','<TOOL_SEP>'}}))
def index():return {'config':set(),'bundle':set(),'components':set(),'query':set(),'qid':set(),'n':0}
def add(idx,c,r=None):
 k=key(c);idx['config'].add(k);idx['bundle'].add(k[1]);idx['components'].update(k[1]);idx['components'].add(k[0]);idx['n']+=1
 if r is not None:
  if r.get('query'):idx['query'].add(norm(r['query']))
  if r.get('qid'):idx['qid'].add(r['qid'])
def main():
 OUT.mkdir(parents=True,exist_ok=True);sources={};indexes={};
 def source(p):sources[str(p)]=sha(p);return rows(p)
 for split in ['train','valid']:
  tar=index();prompt=index();n=0
  for r in source(R/f'dataset/generative_v10_compact/sft_{split}.jsonl'):
   add(tar,parse_target(r['target']),r);inv=compact_inventory(r['prompt']);n+=1
   for ts in inv['bundles']:
    prompt['bundle'].add(tuple(sorted(set(ts))));prompt['n']+=1
   prompt['components'].update(inv['tools']);prompt['components'].update(inv['llms'])
  indexes['generator_'+split+'_targets']=tar;indexes['generator_'+split+'_prompt_bundles']=prompt
  print('generator',split,n,flush=True)
 for split in ['train','valid']:
  allidx=index();actual=index()
  for r in source(B/f'critic_v3_work/data/cases_{split}.jsonl'):
   for c in r['candidates']:add(allidx,c,r)
   if split=='train':
    labelpath=B/'training/critic/stage1/data/labels'/f"{r['query_hash']}.json";label=json.loads(labelpath.read_text());scores=label['scores'];sources[str(labelpath)]=sha(labelpath)
    cs=[{**c,'score':scores[c['id']]/100} for c in r['candidates']]
    groups=collections.defaultdict(list);rng=random.Random(r['query_hash'])
    for i,a in enumerate(cs):
     for b in cs[i+1:]:
      if abs(a['score']-b['score'])<.05:continue
      pos,neg=(a,b) if a['score']>b['score'] else (b,a)
      kind='llm' if set(a['tools'])==set(b['tools']) else ('tools' if a['llm']==b['llm'] else 'both')
      groups[kind].append((pos,neg))
    used={}
    for kind,limit in [('llm',64),('tools',32),('both',32)]:
     rng.shuffle(groups[kind])
     for pos,neg in groups[kind][:limit]:used[key(pos)]=pos;used[key(neg)]=neg
    for c in used.values():add(actual,c,r)
   else:
    for c in r['candidates']:add(actual,c,r)
  indexes['critic_v3_'+split+'_all_candidates']=allidx;indexes['critic_v3_'+split+'_used_candidates']=actual
  allidx=index();actual=index()
  for r in source(B/f'critic_v4_work/data/cases_{split}.jsonl'):
   for c in r['candidates']:add(allidx,c,r)
   ids={p[k] for p in r['pairs'] for k in ['positive','negative']} if split=='train' else set(range(len(r['candidates'])))
   for i in ids:add(actual,r['candidates'][i],r)
  indexes['critic_v4_'+split+'_all_candidates']=allidx;indexes['critic_v4_'+split+'_used_candidates']=actual
  print('critic',split,flush=True)
 def union(names):
  idx=index()
  for name in names:
   for k in idx:
    if k=='n':idx[k]+=indexes[name][k]
    else:idx[k]|=indexes[name][k]
  return idx
 indexes['critic_inherited_train_used']=union(['critic_v3_train_used_candidates','critic_v4_train_used_candidates'])
 indexes['pipeline_train_supervised']=union(['generator_train_targets','critic_inherited_train_used'])
 indexes['pipeline_train_conservative']=union(['generator_train_targets','generator_train_prompt_bundles','critic_v3_train_all_candidates','critic_v4_train_all_candidates'])
 indexes['pipeline_train_valid_conservative']=union(['pipeline_train_conservative','generator_valid_targets','generator_valid_prompt_bundles','critic_v3_valid_all_candidates','critic_v4_valid_all_candidates'])
 # Reference catalog, a dataset catalog rather than an asserted deployed index.
 catalog=index();p=R/'datasets/PartII/agents/merge.json';sources[str(p)]=sha(p)
 agents=json.loads(p.read_text())
 for a in agents.values():
  ts=a.get('T',{}).get('tools',[])
  ts=[t if t.startswith('<') and t.endswith('>') else '<'+t+'>' for t in ts]
  catalog['bundle'].add(tuple(sorted(set(ts))))
 indexes['partii_full_dataset_catalog']=catalog
 test=list(source(R/'datasets/generative_v10_compact/test_manifest.frozen.jsonl'))
 hist={r['qid']:r for r in json.loads((R/'outputs/component_ret_dual_cf_cartesian_seed42_n100_20260921/cached_inputs.json').read_text())['rows']}
 canonical={}
 for name,idx in indexes.items():
  canonical[name]={'config':{(token(l),tuple(sorted({token(t) for t in ts}))) for l,ts in idx['config']},'bundle':{tuple(sorted({token(t) for t in ts})) for ts in idx['bundle']}}
 records=[]
 for r in test:
  gold=parse_target(r['target']);k=key(gold);kc=key(gold,True);exposure={}
  for name,idx in indexes.items():
   exposure[name]={'bundle_seen':k[1] in idx['bundle'],'configuration_seen':k in idx['config'],'bundle_seen_wrapper_normalized':kc[1] in canonical[name]['bundle'],'configuration_seen_wrapper_normalized':kc in canonical[name]['config'],'query_seen':norm(r['query']) in idx['query'],'qid_seen':r['qid'] in idx['qid']}
  retrieved={tuple(sorted(set(b['tool_tokens']))) for b in hist[r['qid']]['bundles'][:5]}
  components=indexes['pipeline_train_conservative']['components'];known=set(k[1])|{k[0]}
  records.append({'qid':r['qid'],'target':gold,'exposure':exposure,'all_target_components_seen_in_task_training':known<=components,'unseen_components':sorted(known-components),'exact_bundle_in_retrieved_top5':k[1] in retrieved,'complete_bundle_in_retrieved_top5':any(set(k[1])<=set(ts) for ts in retrieved),'exact_bundle_in_retrieved_top5_wrapper_normalized':kc[1] in {tuple(sorted({token(t) for t in ts})) for ts in retrieved}})
 counts={name:{'unique_bundles':len(idx['bundle']),'unique_configurations':len(idx['config']),'records_added':idx['n'],'test_bundle_unseen':sum(not r['exposure'][name]['bundle_seen'] for r in records),'test_configuration_unseen':sum(not r['exposure'][name]['configuration_seen'] for r in records),'test_bundle_unseen_wrapper_normalized':sum(not r['exposure'][name]['bundle_seen_wrapper_normalized'] for r in records),'test_configuration_unseen_wrapper_normalized':sum(not r['exposure'][name]['configuration_seen_wrapper_normalized'] for r in records),'test_query_overlap':sum(r['exposure'][name]['query_seen'] for r in records),'test_qid_overlap':sum(r['exposure'][name]['qid_seen'] for r in records)} for name,idx in indexes.items()}
 for name in ['generator_train_targets','generator_valid_targets','critic_v3_train_used_candidates','critic_v3_valid_used_candidates','critic_v4_train_used_candidates','critic_v4_valid_used_candidates']:
  assert counts[name]['test_query_overlap']==0 and counts[name]['test_qid_overlap']==0,(name,counts[name])
 summary={'n':len(records),'sources_sha256':sources,'counts':counts,'exact_bundle_retrieved_top5':sum(r['exact_bundle_in_retrieved_top5'] for r in records),'all_components_seen':sum(r['all_target_components_seen_in_task_training'] for r in records),'limitations':['Post-hoc subgroup analysis on frozen test, not a prospectively held-out combination split.','Task-training audit only: does not establish absence from upstream pretrained encoders or foundation model pretraining.','Configuration exposure concerns explicit paired configurations; separately indexed generator prompt bundles and unpaired LLM/tool inventories.','Full PartII dataset catalog is not asserted to equal the deployed retrieval index.','Critic inherits both V3 training and V4 fine-tuning; V3 actual selected pair endpoints reconstructed using original score thresholds and random seeds.']}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'per_query.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
 # Save compact fingerprints for independent checking and prediction novelty analysis.
 for name,idx in indexes.items():
  (OUT/(name+'.json')).write_text(json.dumps({'bundles':sorted(idx['bundle']),'configurations':sorted(idx['config'])},ensure_ascii=False)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k!='sources_sha256'},indent=2),flush=True)
if __name__=='__main__':main()
