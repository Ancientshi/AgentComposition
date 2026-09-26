from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import pathlib,json,hashlib,random,collections
from compact_input import compact_inventory,tokens
ROOT=pathlib.Path(str(AC_ROOT));OUT=AC_ROOT/'datasets/critic_stage1';OUT.mkdir(exist_ok=True)
sha=lambda x:hashlib.sha256(x.encode()).hexdigest()
norm=lambda x:' '.join(x.split())
tests=[json.loads(x) for x in (ROOT/'datasets/generative_v10_compact/test_manifest.frozen.jsonl').read_text().splitlines()];tq={norm(x['query']) for x in tests};ti={x['qid'] for x in tests}
groups={};excluded=collections.Counter()
for split in ['train','valid']:
 with (ROOT/f'dataset/generative_v10_compact/sft_{split}.jsonl').open() as f:
  for l in f:
   r=json.loads(l);q=norm(r['query'])
   if q in tq or r['qid'] in ti:excluded['test']+=1;continue
   if q in groups:continue
   inv=compact_inventory(r['prompt']);inv['llms']=inv['llms'][:10]
   if len(inv['llms'])!=10:excluded['not10llms']+=1;continue
   # Training-only target tools seed one plausible toolset; no gold label goes to teacher.
   gt=[t for t in tokens(r['target']) if not t.startswith('<LLM_')]
   sets=[]
   def add(ts):
    ts=sorted(set(ts));
    if 1<=len(ts)<=6 and set(ts)<=set(inv['tools']) and ts not in sets:sets.append(ts)
   add(gt)
   for b in inv['bundles']:
    if len(sets)<4:add(b)
   if len(gt)>1:add(gt[:-1])
   other=[t for t in inv['tools'] if t not in gt]
   if other:add((gt+[other[0]]) if len(gt)<6 else (gt[:-1]+[other[0]]))
   if len(sets)<2:excluded['few_toolsets']+=1;continue
   sets=sets[:6];rng=random.Random(sha(q));rng.shuffle(sets)
   llms=list(inv['llms']);rng.shuffle(llms)
   candidates=[{'llm':a,'tools':b} for b in sets for a in llms];rng.shuffle(candidates)
   for i,c in enumerate(candidates):c['id']=f'C{i:03}'
   groups[q]={'qid':r['qid'],'query':r['query'],'query_hash':sha(q),'inventory':inv,'candidates':candidates}
# Group by both query and qid; choose deterministic hash split, never test-driven.
qids=sorted({r['qid'] for r in groups.values()},key=lambda x:sha('critic-v3-split42:'+x));valq=set(qids[:max(1,len(qids)//8)])
rs=sorted(groups.values(),key=lambda r:sha('critic-v3-sample42:'+r['query_hash']))
train=[r for r in rs if r['qid'] not in valq][:1000];val=[r for r in rs if r['qid'] in valq][:150]
assert len(train)==1000 and len(val)==150
assert not {r['qid'] for r in train}&{r['qid'] for r in val}
for split,records in [('train',train),('valid',val)]:
 for r in records:r['split']=split
 (OUT/f'cases_{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
report={'train':len(train),'valid':len(val),'unique_queries':len(groups),'test_query_overlap':0,'test_qid_overlap':0,'test_manifest_sha256':hashlib.sha256((ROOT/'datasets/generative_v10_compact/test_manifest.frozen.jsonl').read_bytes()).hexdigest(),'exclusions':excluded,'candidate_policy':'up to 6 training target/retrieved/perturbed toolsets crossed with 10 retrieved LLMs; randomized IDs; teacher does not receive gold','limitation':'candidate shape matches Cartesian inference; toolsets not yet sampled from generator; source cached retrieval may contain historical gold membership injection'}
(OUT/'preparation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
