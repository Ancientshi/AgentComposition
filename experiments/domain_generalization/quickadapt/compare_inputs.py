from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json,sys,statistics,re
from pathlib import Path
R=Path(str(AC_ROOT));W=R/'experiments/domain_generalization/quickadapt';sys.path.insert(0,str(R/'training/generator'))
from transformers import AutoTokenizer
from compact_context import components,TASK
T=AutoTokenizer.from_pretrained(R/'checkpoints/generative_v10_compact',local_files_only=True)
def stats(a):return {'n':len(a),'min':min(a),'median':statistics.median(a),'mean':round(statistics.mean(a),2),'max':max(a)}
def analyze(rows):
 d={k:[] for k in ['query_tokens','prompt_tokens','llms','tools','bundles']};heads=set();contexts=set()
 for row in rows:
  p=row['prompt'];q=row['query'];ctx=p.split('### Context:\n',1)[1].rsplit('\n\n### User Query:',1)[0];heads.add(p.split('### Context:',1)[0]);contexts.add(ctx)
  lm,ts=components(ctx);d['query_tokens'].append(len(T.encode(q,add_special_tokens=False)));d['prompt_tokens'].append(len(T.encode(p,add_special_tokens=True)));d['llms'].append(len(lm));d['tools'].append(len(ts));bundle=ctx.split('Retrieved tool bundles:\n',1)[1].split('\n\nTool candidates',1)[0];d['bundles'].append(len(re.findall(r'^\d+\.',bundle,re.M)))
 return {'stats':{k:stats(v) for k,v in d.items()},'unique_contexts':len(contexts),'task_prefixes':sorted(heads),'example':{'query':rows[0]['query'],'prompt_start':rows[0]['prompt'][:1800],'target':rows[0].get('target')}}
source=[]
with (R/'datasets/generative_v10_compact/sft_train.jsonl').open() as f:
 for s in f:source.append(json.loads(s))
# Deduplicate query-prompt pairs so multiple positive targets do not overweight inputs.
u={}
for r in source:u.setdefault((r['query'],r['prompt']),r)
a=analyze(list(u.values()));b=analyze(json.loads((R/'outputs/SkillBench_FROZEN_97/prepared.json').read_text()));out={'AgentSelect_v10_train':a,'SkillsBench_frozen97':b,'source_training_rows':len(source)}
(W/'INPUT_COMPARISON.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2))
