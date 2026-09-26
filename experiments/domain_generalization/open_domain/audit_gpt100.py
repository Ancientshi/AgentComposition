"""Independently evaluate FIRST responses, preserving invalid/repeated rank slots."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import collections, hashlib, json, math, re
from pathlib import Path
ROOT=Path(str(AC_ROOT))
OUT=ROOT/'outputs/OPENClaw50_GPT54_FAMILY_OK100_SEED42_RAW'
SOURCE=ROOT/'outputs/OPENClaw50_V10_CRITICV3_FROZEN'
def hashobj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def calc(rows):
 n=len(rows);return {'n':n,'Hit@1':sum(r==1 for r in rows)/n,'Hit@5':sum(0<r<=5 for r in rows)/n,'Hit@10':sum(r>0 for r in rows)/n,'MRR@10':sum(1/r if r else 0 for r in rows)/n}
def main():
 cfg=json.load(open(OUT/'config.json'));inputs=json.load(open(OUT/'inputs.json'));catalog=json.load(open(SOURCE/'catalog.json'))
 tokenmap={c['token']:c['agent_id'] for c in catalog};ids=cfg['sample_ids'];assert len(ids)==len(set(ids))==100
 gold=json.load(open(ROOT/'dataset_opendomain/rankings/merge.json'))['rankings']
 questions=json.load(open(ROOT/'dataset_opendomain/questions/merge.json'))
 result={};audit={};records=[]
 for model in cfg['models']:
  ranks=[];stats=collections.Counter();usage=collections.Counter()
  for qid in ids:
   d=json.load(open(OUT/model/'attempts'/f'{qid}_0.json'))
   assert d['messages_hash']==cfg['input_hashes'][qid]==hashobj(inputs[qid])
   assert d['api']['requested_model']==d['api']['returned_model']==model
   assert questions[qid]['metadata']['judge_verdict']=='ok'
   text=d['raw_output'].strip();text=re.sub(r'^```(?:json)?\s*|\s*```$','',text) if text.startswith('```') else text
   try:
    raw=json.loads(text)['ranking'];assert isinstance(raw,list)
   except (ValueError,KeyError,TypeError,AssertionError):raw=[];stats['unparseable_queries']+=1
   stats['missing_slots']+=max(0,10-len(raw));stats['extra_slots']+=max(0,len(raw)-10)
   seen=set();ranked=[];flags=[]
   for t in raw[:10]:
    if not isinstance(t,str) or t not in tokenmap:
     ranked.append(None);flags.append('out_of_catalog');stats['out_of_catalog_slots']+=1
    elif t in seen:
     ranked.append(None);flags.append('duplicate');stats['duplicate_slots']+=1
    else:ranked.append(tokenmap[t]);flags.append('valid');seen.add(t)
   ranked += [None]*(10-len(ranked));assert len(ranked)==10
   assert len(gold[qid])==1
   rank=ranked.index(gold[qid][0])+1 if gold[qid][0] in ranked else 0;ranks.append(rank)
   for k,v in (d['api'].get('usage') or {}).items():
    if isinstance(v,(int,float)):usage[k]+=v
   records.append({'model':model,'query_id':qid,'ranking':ranked,'raw_ranking':raw,'slot_flags':flags,'gold_rank':rank,'messages_hash':d['messages_hash']})
  result[model]=calc(ranks)
  allcalls=list((OUT/model/'attempts').glob('*.json'));allusage=collections.Counter()
  for p in allcalls:
   d=json.load(open(p))
   for k,v in (d['api'].get('usage') or {}).items():
    if isinstance(v,(int,float)):allusage[k]+=v
  audit[model]={'first_response_usage':dict(usage),'successful_saved_calls_in_run':len(allcalls),'all_saved_call_usage':dict(allusage),'format':dict(stats),'returned_model_verified':True,'identical_inputs_verified':True}
 for method in ['generator','ours_generator10_critic','critic_all50']:
  rr=[]
  for qid in ids:
   d=json.load(open(SOURCE/'per_sample'/f'{qid}.json'));ranked=d['rankings'][method]
   assert len(ranked)==len(set(ranked))==10
   rr.append(ranked.index(gold[qid][0])+1 if gold[qid][0] in ranked else 0)
  result[method]=calc(rr)
 summary={'protocol':'first-response only; first10 original slots; invalid, duplicate and missing slots are misses; no rank compaction or ID correction','sample_ids':ids,'results':result,'audit':audit,'note':'Two extra preliminary GPT-5.4 repair responses remain in the pilot directory and are excluded from main-run call totals. They are not scored.'}
 (OUT/'first_response_metrics.json').write_text(json.dumps(summary,indent=2))
 (OUT/'first_response_evaluated.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
