from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import hashlib,json,shutil
from pathlib import Path
R=Path(str(AC_ROOT)); W=R/'experiments/domain_generalization/all19'; O=R/'outputs/SkillsBench_TEST19_20260923'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
m=read(R/'datasets/SkillBench_SFT_v1/split_manifest.json');ids=sorted(m['query_ids']['val']+m['query_ids']['test']);train=m['query_ids']['train']
assert len(ids)==len(set(ids))==19 and len(train)==78 and not set(ids)&set(train)
qs=read(R/'datasets/SkillBench/canonical/questions/merge.json');agents=read(R/'datasets/SkillBench/agents/merge.json');labels=read(R/'datasets/SkillBench/canonical/rankings/merge.json')['rankings']
prepared={q['sample_id']:q for q in read(R/'outputs/SkillBench_FROZEN_97/prepared.json')}
stats={}
for split,sids in [('train',train),('test',ids)]:
 stats[split]={'tasks':len({qs[s]['metadata']['task_id'] for s in sids}),'queries':len(sids),'resolved_pairs':sum(bool(agents[a]['M'].get('name')) for s in sids for a in labels[s]),'unresolved_pairs':sum(not bool(agents[a]['M'].get('name')) for s in sids for a in labels[s])}
assert stats['train']=={'tasks':63,'queries':78,'resolved_pairs':506,'unresolved_pairs':0}
assert stats['test']=={'tasks':16,'queries':19,'resolved_pairs':109,'unresolved_pairs':1}
assert not {qs[s]['metadata']['task_id'] for s in train}&{qs[s]['metadata']['task_id'] for s in ids}
assert not {qs[s]['input'].strip() for s in train}&{qs[s]['input'].strip() for s in ids}
ck=R/'checkpoints/skillbench_v1_from_v10/checkpoint-192'
state=read(ck/'trainer_state.json');assert state['global_step']==192 and state['epoch']==3.0
protocol={'dataset':stats,'test_ids':ids,'train_ids':train,'test_manifest_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'checkpoint':str(ck),'checkpoint_sha256':sha(ck/'adapter_model.safetensors'),'checkpoint_selection':'Fixed final epoch 3 (step192); do not use validation-selected best/step64. No retraining or selection on these results.','historical_exposure':'Seven queries were validation queries in the earlier run; this is not an untouched test set.','generation':'Existing generator-only beam(1,2), max_tools10, original search length penalty0.1; final ranking generator average log probability, no final length penalty; ties by node ID. No critic.','baseline_policy':'Reuse exact first-response records with identical prompts, request only missing model/query pairs; no format retries.','metrics':'Query macro average over all19, missing/invalid/duplicate ranks zero; fixed10 discount normalization; literal identifier matching; valid tools may score with invalid LLM; all metrics scaled0-100.'}
write(O/'protocol.json',protocol);write(O/'test_manifest.json',[prepared[s] for s in ids]);write(O/'gold.json',{s:[{'llm':agents[a]['M'].get('name'),'tools':agents[a]['T']['tools']} for a in labels[s]] for s in ids})
write(O/'queries.json',[{'sample_id':s,'task_id':qs[s]['metadata']['task_id'],'query':qs[s]['input'],'original_split':'val' if s in m['query_ids']['val'] else 'test'} for s in ids])
x=read(R/'experiments/domain_generalization/frozen/inputs.json');write(O/'catalog_map.json',{'components':{a['token']:a['id'] for a in x['components']},'llms':{a['token']:a['id'] for a in x['llms']}})
models=['gpt-5.4-2026-03-05','gpt-5.4-mini-2026-03-17','gpt-5.4-nano-2026-03-17','gpt-5.6-luna','gpt-5.6-terra','gpt-5.6-sol','deepseek-ai/DeepSeek-V4-Pro','Qwen/Qwen3.8-27B','Pro/moonshotai/Kimi-K2.6'];sources={};missing=[]
for model in models:
 for sid in ids:
  canonical=read(R/'outputs/SkillBench_FROZEN_97/gpt-5.4-2026-03-05/per_sample'/f'{sid}.json')['messages']
  src=next((base/model/'per_sample'/f'{sid}.json' for base in [R/'outputs/SkillBench_EXTRA_12',R/'outputs/SkillBench_SFT_v1_NO_CRITIC',R/'outputs/SkillBench_FROZEN_97'] if (base/model/'per_sample'/f'{sid}.json').exists()),None)
  if src:
   row=read(src);assert row['sample_id']==sid and row['messages']==canonical and row['api']['returned_model']==model
   dst=O/model/'per_sample'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);sources[model+'/'+sid]={'path':str(src),'sha256':sha(src)}
  else:missing.append({'model':model,'sample_id':sid,'messages':canonical})
write(O/'cached_prediction_provenance.json',sources);write(O/'missing_api_requests.json',missing)
print(json.dumps({'stats':stats,'checkpoint_sha256':protocol['checkpoint_sha256'],'cached_baseline_responses':len(sources),'missing_baseline_requests':len(missing)}),flush=True)
