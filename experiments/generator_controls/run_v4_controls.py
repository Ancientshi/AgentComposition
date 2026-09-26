"""Frozen V4 comparisons, shared evidence and no gold in candidate construction."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse,collections,copy,hashlib,importlib.util,inspect,json,math,os,pathlib,statistics,sys,time
P=pathlib.Path
ROOT=P(str(AC_ROOT)); B=P(str(AC_ROOT))
CK=AC_ROOT/'checkpoints/bundle_critic_reference_v4/best_critic.pt'
EXPECTED='becd7447dcac496c1216d55998cf399c2bb9efcc6a9f49a75dd5189e447bc061'
SOURCE=ROOT/'outputs/table1_ours_v10_compact_fixedtest100/results.jsonl'
HIST=ROOT/'outputs/component_ret_dual_cf_cartesian_seed42_n100_20260921/cached_inputs.json'
METRICS=['ToolR@1','Tool-Hit@10','CompR@1','CR-Hit@1','CR-Hit@10','CR-MRR@10','RDCR@10','RDCP@10','RDCF1@10','CompP@1','ToolP@1','Exact-Config-Hit@10','Exact-Bundle-Hit@10']
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def rows(p):
 with P(p).open() as f:
  for l in f:
   if l.strip():yield json.loads(l)
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def lines(p,rs):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
def key(c):return c['llm'],tuple(sorted(set(c['tools'])-{'<TOOL_EMPTY>'}))
def cand(l,t):return {'llm':l,'tools':sorted(set(t)-{'<TOOL_EMPTY>'})}
def unique(cs):return list({key(c):c for c in cs}.values())
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def setup():
 sys.path.insert(0,str(B/'training/critic/stage1'));import compact_input as ser
 sys.path.insert(0,str(ROOT/'training/generator'));from compact_context import parse_context
 def inventory(text):
  if 'LLM candidates (retrieval order):' in text:return ser.compact_inventory(text)
  llms,tools,bundles,desc,strengths,_=parse_context(text)
  return {'llms':llms,'tools':tools,'bundles':bundles,'descriptions':{**desc,**strengths}}
 source=inspect.getsource(ser.serialize)
 old="assert llm.startswith('<LLM_') and 1<=len(tools)<=6"
 assert source.count(old)==1
 ns=dict(ser.__dict__);exec(source.replace(old,"assert llm.startswith('<LLM_')"),ns)
 evaluator=load(ROOT/'outputs/component_ret_dual_cf_cartesian_seed42_n100_20260921/evaluator.py','control_evaluator')
 core=load(B/'training/critic/stage2/core.py','control_core')
 return ser,ns['serialize'],inventory,evaluator,core

def prepare(inventory,ev):
 sources=list(rows(SOURCE));hist={r['qid']:r for r in json.loads(HIST.read_text())['rows']}
 rag={p.parent.name:list(rows(p)) for p in sorted((ROOT/'EXP').glob('*rag*top10*/results.jsonl'))}
 assert len(rag)==10 and all(len(rs)==100 for rs in rag.values())
 rag={n:{r['dataset_example']['qid']:r for r in rs} for n,rs in rag.items()}
 prepared=[];align=[]
 for r in sources:
  ex=r['dataset_example'];qid=ex['qid'];h=hist[qid];query=r['query'];inv=inventory(r['context']['text'])
  llms=list(dict.fromkeys(inv['llms']))[:10];assert len(llms)==10
  assert h['query_sha256']==hashlib.sha256(query.encode()).hexdigest()
  assert h['dataset_example']['target']==ex['target']
  assert llms==[x['token'] for x in h['llms'][:10]]
  # Cross-check intact bundles against the SAME raw retrieval response.
  raw=r['retrieval']['cf_tool_response']['results'][:5]
  rawsets=[]
  for b in raw:
   ts=b.get('tool_ids',[])
   ts=[t if t.startswith('<<') or (t.startswith('<') and t.endswith('>')) else '<'+t+'>' for t in ts]
   rawsets.append(tuple(sorted(set(ts))))
  hsets=[tuple(sorted(set(b['tool_tokens']))) for b in h['bundles'][:5]]
  normalize=lambda xs:[tuple(sorted(' '.join(t.replace('<Tool_','<TOOL_').split()) for t in ts)) for ts in xs]
  assert normalize(rawsets)==normalize(hsets),(qid,rawsets,hsets)
  trace=r['generation']['search_trace'];assert trace['search_critic_enabled'] is False
  completed=unique([cand(c['llm'],c['tools']) for c in trace['final_rerank_pool_before_cap']])
  toolsets=sorted({key(c)[1] for c in completed})
  fullcart=[cand(l,t) for t in toolsets for l in llms]
  products=[]
  for i,l in enumerate(llms,1):
   for j,b in enumerate(h['bundles'][:5],1):products.append({**cand(l,b['tool_tokens']),'rank_sum':i+j,'llm_rank':i,'bundle_rank':j})
  products.sort(key=lambda c:(c['rank_sum'],c['llm_rank'],c['bundle_rank']))
  seen=set();history=[]
  for c in products:
   if key(c) not in seen:seen.add(key(c));history.append(c)
  groups={'history_full':history,'history_legal':[c for c in history if 1<=len(c['tools'])<=6], 'ours_completed':completed,'ours_fullcart':fullcart}
  for n,rs in rag.items():
   rr=rs[qid];assert rr['dataset_example']==ex and rr['query']==query
   cs=[cand(l,t) for l,t in ev.get_candidates(rr)]
   assert len(cs)==10 and len({key(c) for c in cs})==10
   groups[n]=cs
  legacy_tools={key(cand(c['llm_token'],c['tool_tokens']))[1] for c in r['results'][:10]}
  prepared.append({'qid':qid,'query':query,'inventory':inv,'dataset_example':ex,'groups':groups,'legacy_tools':legacy_tools})
  align.append({'qid':qid,'llms_exact':True,'bundles_exact':rawsets==hsets,'bundles_formatting_equivalent':True,'rag_query_reference_exact':True})
 return prepared,rag,align

def evaluate(cs,example,ev,core):
 preds=[{'rank':i+1,'llm_token':c['llm'],'tool_tokens':c['tools']} for i,c in enumerate(cs[:10])]
 off=ev.evaluate_one({'dataset_example':example,'results':preds})
 ref=core.parse_target(example['target']);ms=[core.metrics(c,[ref]) for c in cs[:10]]
 weights=[1/math.log2(i+2) for i in range(10)];den=sum(weights)
 m={'ToolR@1':ms[0]['tr'],'Tool-Hit@10':float(any(x['tool_hit'] for x in ms)), 'CompR@1':ms[0]['cr'],'CR-Hit@1':float(ms[0]['agent_hit']),'CR-Hit@10':float(any(x['agent_hit'] for x in ms)), 'CR-MRR@10':next((1/(i+1) for i,x in enumerate(ms) if x['agent_hit']),0.), 'RDCR@10':sum(w*x['cr'] for w,x in zip(weights,ms))/den,'RDCP@10':sum(w*x['cp'] for w,x in zip(weights,ms))/den,'RDCF1@10':sum(w*x['cf1'] for w,x in zip(weights,ms))/den,'CompP@1':ms[0]['cp'],'ToolP@1':ms[0]['tp']}
 m['Exact-Config-Hit@10']=float(any(key(c)==key(ref) for c in cs[:10]))
 m['Exact-Bundle-Hit@10']=float(any(key(c)[1]==key(ref)[1] for c in cs[:10]))
 fields=['top1_tool_recall','tool_hit@10','top1_component_recall','top1_complete_recall','cr_hit@10','cr_mrr@10','rdcr@10']
 assert max(abs(m[k]-off[f]) for k,f in zip(METRICS,fields))<1e-12
 return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=P,required=True);ap.add_argument('--check-only',action='store_true');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 ser,serialize,inventory,ev,core=setup();prepared,rag,alignment=prepare(inventory,ev)
 assert sha(CK)==EXPECTED
 cfg=json.loads((CK.parent/'ranking_config.json').read_text());assert cfg['alpha']==1 and cfg['epoch']==2
 import torch
 from transformers import AutoConfig,AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(AC_EASYREC_MODEL),local_files_only=True)
 preflight={'n':len(prepared),'rag_models':list(rag),'source_sha256':sha(SOURCE),'history_sha256':sha(HIST),'checkpoint_sha256':sha(CK),'alignment':alignment,'serialization_extended_only_cardinality_assertion':True,'out_of_range':[],'unscorable':[],'in_range_equal':0,'candidates':0}
 for r in prepared:
  union=unique([c for cs in r['groups'].values() for c in cs]);r['union']=sorted(union,key=key);r['ids']=[];r['encoding_errors']={}
  tokenkeys={}
  for c in r['union']:
   k=key(c);preflight['candidates']+=1
   if not 1<=len(c['tools'])<=6:preflight['out_of_range'].append({'qid':r['qid'],**c})
   try:
    text,ids,meta=serialize(r['query'],c,r['inventory'],tok,512)
    if 1<=len(c['tools'])<=6:
     original=ser.serialize(r['query'],c,r['inventory'],tok,512);assert original==(text,ids,meta);preflight['in_range_equal']+=1
    if tuple(ids) in tokenkeys:assert tokenkeys[tuple(ids)]==k
    tokenkeys[tuple(ids)]=k;r['ids'].append(ids)
   except ValueError as e:
    r['encoding_errors'][k]=str(e);r['ids'].append(None);preflight['unscorable'].append({'qid':r['qid'],**c,'error':str(e)})
  assert not any(key(c) in r['encoding_errors'] for c in r['groups']['history_legal'])
  assert not any(key(c) in r['encoding_errors'] for c in r['groups']['ours_completed'])
 print(json.dumps({k:v for k,v in preflight.items() if k not in ['alignment','out_of_range','unscorable']}),flush=True)
 print('BOUNDARIES',len(preflight['out_of_range']),len(preflight['unscorable']),flush=True)
 if a.check_only:
  write(a.output/'preflight.json',preflight);return
 if a.output.exists() and not a.resume:raise FileExistsError('Use a fresh output or explicit resume')
 a.output.mkdir(parents=True,exist_ok=True)
 import fcntl
 lock=(a.output/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 config={'checkpoint_sha256':EXPECTED,'epoch':2,'alpha':1,'source_sha256':sha(SOURCE),'history_sha256':sha(HIST),'rag_sha256':{n:sha(ROOT/'EXP'/n/'results.jsonl') for n in rag},'code_sha256':sha(__file__),'serialization_sha256':sha(B/'training/critic/stage1/compact_input.py'),'dtype':'float16 autocast','batch_size':48,'ranking':'raw V4 descending, canonical LLM/unordered tool tuple ties','reference_used_for_ranking':False}
 if (a.output/'config.json').exists():assert json.loads((a.output/'config.json').read_text())==config
 write(a.output/'config.json',config);write(a.output/'preflight.json',preflight)
 sys.path.insert(0,str(AC_ROOT/'models'));from train_bundle_critic_improved import EasyRecBundleCritic
 sys.path.insert(0,str(AC_ROOT / 'models/easyrec'));from model import Easyrec
 ck=torch.load(CK,map_location='cpu',weights_only=False);assert ck['epoch']==2;ca=ck['args']
 modelcfg=AutoConfig.from_pretrained(ca['model_dir'],local_files_only=True)
 enc=Easyrec.from_pretrained(ca['model_dir'],config=modelcfg,local_files_only=True)
 model=EasyRecBundleCritic(enc,modelcfg.hidden_size,ca['head_hidden'],ca['dropout'],ca['normalize_embedding']).cuda()
 model.load_state_dict(ck['model_state_dict'],strict=True);model.eval();del ck
 predictions=[];pools=[];start=time.time()
 for qi,r in enumerate(prepared,1):
  qid=r['qid'];cache=a.output/'scores'/(qid+'.json');un=r['union'];valid=[i for i,x in enumerate(r['ids']) if x is not None]
  if cache.exists():
   saved=json.loads(cache.read_text());assert saved['config_sha256']==digest(config);assert [key(c) for c in saved['candidates']]==[key(c) for c in un];scores=saved['scores']
  else:
   scores=[None]*len(un)
   with torch.inference_mode():
    for j in range(0,len(valid),48):
     ix=valid[j:j+48];batch=tok.pad({'input_ids':[r['ids'][i] for i in ix]},padding=True,return_tensors='pt').to('cuda')
     with torch.autocast('cuda',dtype=torch.float16):ss=model(batch)
     for i,s in zip(ix,ss.float().cpu().tolist()):assert math.isfinite(s);scores[i]=s
   write(cache,{'qid':qid,'config_sha256':digest(config),'candidates':un,'scores':scores})
  scoremap={key(c):s for c,s in zip(un,scores)}
  def rank(cs):
   assert all(scoremap[key(c)] is not None for c in cs)
   return sorted(cs,key=lambda c:(-scoremap[key(c)],key(c)))
  g=r['groups'];views={};viewpools={}
  for name in ['history_legal','history_full']:
   if any(key(c) in r['encoding_errors'] for c in g[name]):continue
   views[name+'_retrieval']=g[name][:10];views[name+'_v4']=rank(g[name])[:10]
   viewpools[name+'_retrieval']=viewpools[name+'_v4']=g[name]
  original=rank(g['ours_completed']);selected_tools={key(c)[1] for c in original[:10]}
  cart=[c for c in g['ours_fullcart'] if key(c)[1] in selected_tools]
  legacy=[c for c in g['ours_fullcart'] if key(c)[1] in r['legacy_tools']]
  for name,cs in [('ours_no_expansion_v4',g['ours_completed']),('ours_cartesian_v4',cart),('ours_legacy_seeds_v4',legacy)]:
   views[name]=rank(cs)[:10];viewpools[name]=cs
  for name in rag:
   cs=g[name]
   if any(key(c) in r['encoding_errors'] for c in cs):continue
   views[name+'_original']=cs;views[name+'_v4']=rank(cs);viewpools[name+'_original']=viewpools[name+'_v4']=cs
  ref=core.parse_target(r['dataset_example']['target']);rk=key(ref)
  for name,selected in views.items():
   met=evaluate(selected,r['dataset_example'],ev,core);pool=viewpools[name]
   predictions.append({'qid':qid,'method':name,'dataset_example':r['dataset_example'],'metrics':met,'results':[{'score':scoremap[key(c)],**c} for c in selected],'pool_count':len(pool),'mean_tools':statistics.fmean(len(c['tools']) for c in selected),'out_of_range_top10':sum(not 1<=len(c['tools'])<=6 for c in selected)})
   pools.append({'qid':qid,'method':name,'pool_count':len(pool),'exact_configuration_available':any(key(c)==rk for c in pool),'complete_recall_available':any(c['llm']==ref['llm'] and set(ref['tools'])<=set(c['tools']) for c in pool),'exact_bundle_available':any(key(c)[1]==rk[1] for c in pool),'candidate_signatures_sha256':digest(sorted(key(c) for c in pool))})
  for name in rag:
   if name+'_v4' in views:
    before=evaluate(views[name+'_original'],r['dataset_example'],ev,core);after=evaluate(views[name+'_v4'],r['dataset_example'],ev,core)
    assert all(before[k]==after[k] for k in ['CR-Hit@10','Tool-Hit@10'])
    assert {key(c) for c in views[name+'_original']}=={key(c) for c in views[name+'_v4']}
  print(json.dumps({'queries':qi,'total':len(prepared),'scored_candidates':len(un),'seconds':round(time.time()-start,1)}),flush=True)
  write(a.output/'status.json',{'stage':'scoring','queries':qi,'total':len(prepared),'seconds':time.time()-start})
 lines(a.output/'predictions.jsonl',predictions);lines(a.output/'pool_diagnostics.jsonl',pools)
 methods=sorted({p['method'] for p in predictions});summary={}
 for name in methods:
  rs=[p for p in predictions if p['method']==name];ps=[p for p in pools if p['method']==name]
  summary[name]={'n':len(rs),'metrics':{k:statistics.fmean(p['metrics'][k] for p in rs) for k in METRICS},'mean_pool_count':statistics.fmean(p['pool_count'] for p in rs),'mean_tool_count':statistics.fmean(p['mean_tools'] for p in rs),'pool_complete_recall':statistics.fmean(p['complete_recall_available'] for p in ps),'pool_exact_config':statistics.fmean(p['exact_configuration_available'] for p in ps),'out_of_range_top10':sum(p['out_of_range_top10'] for p in rs)}
 write(a.output/'summary.json',summary);write(a.output/'status.json',{'stage':'complete','n':len(prepared),'seconds':time.time()-start,'checkpoint':str(CK),'checkpoint_sha256':EXPECTED})
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
