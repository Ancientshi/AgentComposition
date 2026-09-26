"""Use the actual frozen Ours implementation with a shared deployable inventory."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(str(AC_ROOT))
OUT = ROOT/'experiments/end_to_end_initial'
sys.path[:0] = [str(ROOT), str(ROOT/'training/generator')]
import baseline5_run_infer_rag_gpt as api
api.ensure_env_cuda_library()
import run_infer_table1_compact as wrapper


def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)


p=argparse.ArgumentParser()
p.add_argument('--dataset',choices=['AgentSelect','SkillsBench'],required=True)
p.add_argument('--shard',type=int,default=0)
p.add_argument('--shards',type=int,default=1)
p.add_argument('--limit',type=int,default=0)
a=p.parse_args()
protocol=json.loads((OUT/'protocol.json').read_text())
context=(OUT/'context.txt').read_text()
assert hashlib.sha256(context.encode()).hexdigest()==protocol['global_context_sha256']
source=ROOT/('experiments/domain_generalization/sft/sota_no_critic.py' if a.dataset=='SkillsBench' else 'inference/run_infer_v13_batch_eval_sota.py')
sota=api.load_module(source,'e2e_ours_'+a.dataset)
model=protocol['skillsbench_model' if a.dataset=='SkillsBench' else 'agentselect_model']
opts=SimpleNamespace(model_dir=model,base_model=str(AC_BASE_MODEL),critic_url='http://127.0.0.1:8016')
args=wrapper.configure(sota,opts)
args.num_results=30
args.complete_pool_size=50
args.search_checkpoint_interval=1
args.max_tools=10 if a.dataset=='SkillsBench' else 6
args.max_source_length=8192
if a.dataset=='SkillsBench':
    args.critic_required=0;args.critic_score_weight=0
    class NoCritic:
        def __init__(self,*args,**kwargs):self.events=[]
        def score_nodes(self,*args,**kwargs):raise AssertionError('No SkillsBench critic')
        def analyze_node(self,*args,**kwargs):return {'skipped':True}
    sota.BundleCriticClient=NoCritic
else:
    h=wrapper.critic_health(opts.critic_url)
    assert h['model']['weights_sha256']=='becd7447dcac496c1216d55998cf399c2bb9efcc6a9f49a75dd5189e447bc061'

def prompt(*,context,query,**kwargs):
    return '### Task:\n'+wrapper.TASK+'\n\n### Context:\n'+context+'\n\n### User Query:\n'+query.strip()+'\n\n### Answer:\n'
sota.build_prompt_v12=prompt
tok=sota.load_tokenizer_peft_aware(model,opts.base_model)
rows=[q for q in json.loads((OUT/'queries.json').read_text()) if q['dataset']==a.dataset]
rows=[q for i,q in enumerate(rows) if i%a.shards==a.shard]
if a.limit: rows=rows[:a.limit]
for q in rows:
    sid=q['sample_id'];dest=OUT/'recommendations'/f'{sid}.json'
    if dest.exists():
        old=json.loads(dest.read_text())
        assert old['context_sha256']==protocol['global_context_sha256'] and old['query_sha256']==q['query_sha256']
        continue
    prompt_text=prompt(context=context,query=q['query'])
    n=len(tok.encode(prompt_text))
    assert n<7800,(sid,n)
    args.query=q['query'];args.context=context;args.output_json=str(OUT/'generation'/f'{sid}.json')
    log=OUT/'generation'/f'{sid}.log';log.parent.mkdir(parents=True,exist_ok=True)
    start=time.time()
    with log.open('w') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
        result=sota.run_pipeline(args)
    trace=result['generation']['search_trace']
    assert not trace['search_critic_enabled']
    pool=trace['final_rerank_pool_before_cap']
    if a.dataset=='SkillsBench':
        assert not trace['final_critic_reranking'] and not result['generation']['critic_api_events']
        pool=sorted(pool,key=lambda x:(-x['generator_avg_logprob'],int(x['node_id'][1:])))
    else:
        events=result['generation']['critic_api_events']
        assert trace['final_critic_reranking'] and events
        assert all(e['stage']=='final_rerank' and not e.get('error') for e in events)
    seen=set()
    for rank,node in enumerate(pool,1):
        key=(node['llm'],tuple(sorted(node['tools'])))
        assert key not in seen
        seen.add(key);node['recommendation_rank']=rank
    assert len(pool)>=max(protocol['ranks']),(sid,len(pool))
    save(dest,{'sample_id':sid,'dataset':a.dataset,'query_sha256':q['query_sha256'],
               'context_sha256':protocol['global_context_sha256'],'prompt_sha256':hashlib.sha256(prompt_text.encode()).hexdigest(),
               'prompt_tokens':n,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
               'model':model,'elapsed_sec':time.time()-start,'pool_size':len(pool),'ranked_pool':pool,
               'selected':[pool[i-1] for i in protocol['ranks']]})
    print(json.dumps({'sample_id':sid,'pool_size':len(pool),'elapsed':round(time.time()-start,1)}),flush=True)
