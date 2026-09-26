"""Reuse frozen no-critic beam inference, with an optional explicit task framing."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse,contextlib,json,sys,time,shutil,hashlib
from pathlib import Path
from types import SimpleNamespace
R=Path(str(AC_ROOT));W=R/'experiments/domain_generalization/quickadapt';sys.path.insert(0,str(R));sys.path.insert(0,str(R/'training/generator'))
import baseline5_run_infer_rag_gpt as api
api.ensure_env_cuda_library()
import run_infer_table1_compact as wrapper
sota=api.load_module(R/'experiments/domain_generalization/sft/sota_no_critic.py','quickadapt_sota')
CLARIFY='The User Query below describes a task to assign to an agent. Select one LLM and a set of tools or skills from the Context to perform that task. Do not perform the task yourself. Return only their exact candidate identifiers in this format: LLM_identifier <TOOL_SEP> tool_identifier(s) <SPECIAL_END>.'
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--split',choices=['val','test'],default='test');a=p.parse_args();manifest=json.loads((W/'data/manifest.json').read_text());rows=manifest[a.split+'_manifest'];O=W/'predictions'/a.job/a.split
 import fcntl
 (W/'locks').mkdir(exist_ok=True);lock=(W/'locks'/f'{a.job}.{a.split}').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 if (O/'COMPLETE.json').exists():return
 if a.job=='E_base_frozen':ck=Path(str(AC_BASE_MODEL))
 elif a.job in ['A_frozen','B_explicit_instruction']:ck=R/'checkpoints/generative_v10_compact'
 else:
  assert (W/'runs'/a.job/'COMPLETE.json').exists();ck=W/'runs'/a.job/'best'
 opts=SimpleNamespace(model_dir=ck,base_model=str(AC_BASE_MODEL),critic_url='disabled');args=wrapper.configure(sota,opts);args.max_tools=10;args.critic_required=0;args.critic_score_weight=0
 class NoCritic:
  def __init__(self,*args,**kwargs):self.events=[]
  def score_nodes(self,*args,**kwargs):raise AssertionError('Critic forbidden')
  def analyze_node(self,*args,**kwargs):return {'skipped':True}
 sota.BundleCriticClient=NoCritic;x=json.loads((R/'experiments/domain_generalization/frozen/inputs.json').read_text())
 for i,q in enumerate(rows):
  path=O/'per_sample'/f"{q['sample_id']}.json"
  if path.exists():continue
  prompt=q['prompt']
  if a.job=='B_explicit_instruction':prompt='### Task:\n'+CLARIFY+'\n\n### Context:\n'+prompt.split('### Context:\n',1)[1]
  genpath=O/'generation'/path.name;genpath.parent.mkdir(parents=True,exist_ok=True);args.query=q['query'];args.context=x['context'];args.output_json=str(genpath);sota.build_prompt_v12=lambda **kwargs:prompt;start=time.time()
  with genpath.with_suffix('.log').open('w') as log,contextlib.redirect_stdout(log):r=sota.run_pipeline(args)
  trace=r['generation']['search_trace'];assert not trace['search_critic_enabled'] and not trace['final_critic_reranking'] and not r['generation']['critic_api_events'];assert r['prompt']['text']==prompt
  result=[{'llm':z['llm_token'],'tools':z['tool_tokens']} for z in r['results']]
  write(path,{'sample_id':q['sample_id'],'results':result,'method':a.job,'elapsed_sec':time.time()-start,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'checkpoint':str(ck)})
  print(a.job,a.split,i+1,'/',len(rows),round(time.time()-start,2),flush=True)
 write(O/'COMPLETE.json',{'job':a.job,'split':a.split,'queries':len(rows),'checkpoint':str(ck)})
if __name__=='__main__':main()
