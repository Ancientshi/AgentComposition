from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json,hashlib,sys
from pathlib import Path
R=Path(str(AC_ROOT));W=R/'experiments/domain_generalization/sft';D=R/'datasets/SkillBench';O=R/'datasets/SkillBench_SFT_v1';sys.path.insert(0,str(R/'training/generator'));from compact_context import encode_supervision
from transformers import AutoTokenizer

def wr(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
x=json.loads((R/'experiments/domain_generalization/frozen/inputs.json').read_text());lm={z['id']:z['token'] for z in x['llms']};tool={z['id']:z['token'] for z in x['components']};prepared={q['sample_id']:q for q in json.loads((R/'outputs/SkillBench_FROZEN_97/prepared.json').read_text())};questions=json.loads((D/'canonical/questions/merge.json').read_text());rank=json.loads((D/'canonical/rankings/merge.json').read_text())['rankings'];agents=json.loads((D/'agents/merge.json').read_text());splits=json.loads((D/'task_splits.json').read_text())
assert all(not set(splits[a])&set(splits[b]) for a,b in [('train','val'),('train','test'),('val','test')])
tok=AutoTokenizer.from_pretrained(R/'checkpoints/generative_v10_compact',local_files_only=True);ids={};querysets={};stats={};O.mkdir(exist_ok=True)
for split,tasks in splits.items():
 selected=sorted(k for k,v in questions.items() if v['metadata']['task_id'] in tasks);ids[split]=selected;querysets[split]={questions[k]['input'].strip() for k in selected};rows=[];skipped=[]
 for sid in selected:
  q=prepared[sid];assert q['query']==questions[sid]['input']
  if split=='test':continue # no test targets are tokenized for optimization
  seen=set()
  for aid in rank[sid]:
   a=agents[aid]
   if not a['M']:skipped.append(aid);continue
   ts=sorted(tool[z] for z in a['T']['tools']);assert 1<=len(ts)<=10
   target=lm[a['M']['name']]+' <TOOL_SEP> '+' '.join(ts)+' <SPECIAL_END>'
   if target in seen:continue
   seen.add(target);enc=encode_supervision(q['prompt'],target,tok,8192)
   rows.append({'sample_id':sid,'agent_id':aid,'task_id':questions[sid]['metadata']['task_id'],'query':q['query'],'target':target,**enc})
 if split!='test':(O/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 stats[split]={'tasks':len(tasks),'queries':len(selected),'training_pairs':len(rows),'skipped_unresolved_llm':len(skipped),'max_sequence':max((len(r['input_ids']) for r in rows),default=0),'min_supervised_tokens':min((sum(y!=-100 for y in r['labels']) for r in rows),default=0)}
assert all(not querysets[a]&querysets[b] for a,b in [('train','val'),('train','test'),('val','test')])
wr(O/'split_manifest.json',{'task_ids':splits,'query_ids':ids,'test_manifest':[prepared[k] for k in ids['test']]})
wr(O/'preparation_report.json',{'stats':stats,'split_sha256':sha(D/'task_splits.json'),'inputs_sha256':sha(R/'experiments/domain_generalization/frozen/inputs.json'),'source_prepared_sha256':sha(R/'outputs/SkillBench_FROZEN_97/prepared.json'),'train_sha256':sha(O/'train.jsonl'),'val_sha256':sha(O/'val.jsonl'),'max_seq_len':8192,'prompt_policy':'exact frozen per-query prompts; no query truncation; full shared inventory','label_policy':'up to10 unique observed successful named-LLM positives per training query, deduplicated; sorted set order','test_policy':'9 held-out tasks/12 queries; original split fixed before this fine-tuning; never used for checkpoint selection'})
print(json.dumps(stats,indent=2))
