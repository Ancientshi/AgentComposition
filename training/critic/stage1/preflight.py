from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json,pathlib,collections,hashlib,sys,statistics
from transformers import AutoTokenizer
from compact_input import serialize
from serve import inventory
ROOT=pathlib.Path(__file__).resolve().parent;D=AC_ROOT/'datasets/critic_stage1';tok=AutoTokenizer.from_pretrained(str(AC_EASYREC_MODEL),local_files_only=True)
report={}
for split in ['train','valid']:
 count=0;lengths=[];budgets=collections.Counter()
 for line in (D/f'cases_{split}.jsonl').read_text().splitlines():
  r=json.loads(line);seen=set()
  for c in r['candidates']:
   _,ids,m=serialize(r['query'],c,r['inventory'],tok);assert tuple(ids) not in seen;seen.add(tuple(ids));count+=1;lengths.append(len(ids));budgets[str(m['query_budget'])+','+str(m['evidence_budget_each'])]+=1
 report[split]={'candidates':count,'max_length':max(lengths),'median_length':statistics.median(lengths),'budgets':budgets}
# Verify long queries no longer erase identities on fixed test, without using targets.
rs=[json.loads(x) for x in pathlib.Path(str(AC_ROOT / 'outputs/table1_ours_v10_compact_fixedtest100/results.jsonl')).read_text().splitlines()];seen_queries=0;count=0
for r in rs:
 inv=inventory(r['context']['text']);seen=set()
 for c in r['results']:
  text,ids,m=serialize(r['query'],{'llm':c['llm_token'],'tools':c['tool_tokens']},inv,tok);assert len(ids)<=512 and tuple(ids) not in seen;seen.add(tuple(ids));count+=1
 seen_queries+=1
report['test_input_only']={'queries':seen_queries,'candidates':count,'collisions':0,'targets_used':False}
labels=[json.loads(x.read_text()) for x in (D/'labels').glob('*.json')];report['pilot']={'records':len(labels),'models':list(set(x['model'] for x in labels)),'score_range_per_record':[(min(x['scores'].values()),max(x['scores'].values())) for x in labels]}
(D/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
