from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse,contextlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace
R=Path(str(AC_ROOT));W=R/'experiments/domain_generalization/sft';O=R/'outputs/SkillBench_SFT_v1_NO_CRITIC';sys.path.insert(0,str(R));sys.path.insert(0,str(R/'training/generator'))
import baseline5_run_infer_rag_gpt as api
api.ensure_env_cuda_library()
import run_infer_table1_compact as wrapper
sota=api.load_module(W/'sota_no_critic.py','skillbench_sft_sota_no_critic')
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4);a=p.parse_args()
ck=R/'checkpoints/skillbench_v1_from_v10';assert (ck/'COMPLETE.json').exists()
opts=SimpleNamespace(model_dir=ck/'best',base_model=str(AC_BASE_MODEL),critic_url='disabled')
args=wrapper.configure(sota,opts);args.max_tools=10;args.critic_required=0;args.critic_score_weight=0
class NoCritic:
 def __init__(self,*args,**kwargs):self.events=[]
 def score_nodes(self,*args,**kwargs):raise AssertionError('Critic is forbidden for this experiment')
 def analyze_node(self,*args,**kwargs):return {'skipped':True,'reason':'critic disabled'}
sota.BundleCriticClient=NoCritic
x=json.loads((R/'experiments/domain_generalization/frozen/inputs.json').read_text());manifest=json.loads((R/'datasets/SkillBench_SFT_v1/split_manifest.json').read_text());rows=manifest['test_manifest']
for i,q in enumerate(rows):
 if i%a.shards!=a.shard:continue
 path=O/'finetuned/per_sample'/f"{q['sample_id']}.json"
 if path.exists():continue
 genpath=O/'finetuned/generation'/path.name;genpath.parent.mkdir(parents=True,exist_ok=True);start=time.time()
 args.query=q['query'];args.context=x['context'];args.output_json=str(genpath);sota.build_prompt_v12=lambda **kwargs:q['prompt']
 with genpath.with_suffix('.log').open('w') as log,contextlib.redirect_stdout(log):r=sota.run_pipeline(args)
 trace=r['generation']['search_trace'];assert not trace['search_critic_enabled'] and not trace['final_critic_reranking'] and not r['generation']['critic_api_events']
 pool=trace['final_rerank_pool_before_cap'];expected=sorted(pool,key=lambda n:(-(n['generator_avg_logprob']-.1*len(n['tools'])),-n['generator_avg_logprob'],int(n['node_id'][1:])))[:10]
 result=[{'llm':z['llm_token'],'tools':z['tool_tokens']} for z in r['results']];assert [(z['llm'],z['tools']) for z in expected]==[(z['llm'],z['tools']) for z in result]
 write(path,{'sample_id':q['sample_id'],'results':result,'method':'finetuned','elapsed_sec':time.time()-start})
 raw=sorted(pool,key=lambda n:(-n['generator_avg_logprob'],int(n['node_id'][1:])))[:10]
 write(O/'finetuned_no_length_penalty/per_sample'/path.name,{'sample_id':q['sample_id'],'results':[{'llm':n['llm'],'tools':n['tools']} for n in raw],'method':'finetuned_no_length_penalty'})
 print('FINETUNED TEST',i+1,'/',len(rows),'seconds',round(time.time()-start,1),flush=True)
