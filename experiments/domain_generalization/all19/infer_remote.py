from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse,contextlib,json,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
R=Path(str(AC_ROOT));W=R/'experiments/domain_generalization/all19';O=R/'outputs/SkillsBench_TEST19_20260923'
sys.path[:0]=[str(R),str(R/'training/generator')]
import baseline5_run_infer_rag_gpt as api
api.ensure_env_cuda_library()
import run_infer_table1_compact as wrapper
sota=api.load_module(R/'experiments/domain_generalization/sft/sota_no_critic.py','skillbench_test19_sota')
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
p=argparse.ArgumentParser();p.add_argument('--shard',type=int);p.add_argument('--shards',type=int,default=8);a=p.parse_args()
protocol=json.loads((O/'protocol.json').read_text());ck=Path(protocol['checkpoint'])
opts=SimpleNamespace(model_dir=ck,base_model=str(AC_BASE_MODEL),critic_url='disabled')
args=wrapper.configure(sota,opts);args.max_tools=10;args.critic_required=0;args.critic_score_weight=0
class NoCritic:
 def __init__(self,*args,**kwargs):self.events=[]
 def score_nodes(self,*args,**kwargs):raise AssertionError('Critic is forbidden')
 def analyze_node(self,*args,**kwargs):return {'skipped':True,'reason':'critic disabled'}
sota.BundleCriticClient=NoCritic
x=json.loads((R/'experiments/domain_generalization/frozen/inputs.json').read_text());rows=json.loads((O/'test_manifest.json').read_text())
for i,q in enumerate(rows):
 if i%a.shards!=a.shard:continue
 path=O/'ours/per_sample'/f"{q['sample_id']}.json"
 if path.exists():continue
 genpath=O/'ours/generation'/path.name;genpath.parent.mkdir(parents=True,exist_ok=True);start=time.time()
 args.query=q['query'];args.context=x['context'];args.output_json=str(genpath);sota.build_prompt_v12=lambda **kwargs:q['prompt']
 try:
  with genpath.with_suffix('.log').open('w') as log,contextlib.redirect_stdout(log):r=sota.run_pipeline(args)
  trace=r['generation']['search_trace'];assert not trace['search_critic_enabled'] and not trace['final_critic_reranking'] and not r['generation']['critic_api_events']
  pool=trace['final_rerank_pool_before_cap'];ranked=sorted(pool,key=lambda n:(-n['generator_avg_logprob'],int(n['node_id'][1:])))[:10]
  result=[{'llm':n['llm'],'tools':n['tools']} for n in ranked]
  write(path,{'sample_id':q['sample_id'],'results':result,'method':'ours_final_epoch3_no_final_length_penalty','elapsed_sec':time.time()-start,'checkpoint_sha256':protocol['checkpoint_sha256'],'completed_pool_size':len(pool)})
 except Exception as exc:
  write(path,{'sample_id':q['sample_id'],'results':[],'generation_error':type(exc).__name__+': '+str(exc),'traceback':traceback.format_exc(),'elapsed_sec':time.time()-start,'checkpoint_sha256':protocol['checkpoint_sha256']});traceback.print_exc()
 print('COMPLETE',q['sample_id'],'seconds',round(time.time()-start,1),flush=True)
