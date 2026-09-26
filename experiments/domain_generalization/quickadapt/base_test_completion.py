"""Complete the omitted original-base held-out test inference without retraining."""
import os,sys,json,subprocess,fcntl
from pathlib import Path
W=Path(__file__).resolve().parent

def status(stage,**kw):
 p=W/'BASE_CONTROL_STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'stage':stage,'validation_complete':True,'test_extension':'70 queries,54 tasks; same selected checkpoints and decoder',**kw},indent=2));t.replace(p)
def main():
 lock=(W/'base_test_completion.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 status('test_inference')
 for job in ['E_base_n3_s42','E_base_frozen']:
  if not (W/'predictions'/job/'test/COMPLETE.json').exists():
   with (W/'logs'/f'base_test.{job}.log').open('w') as log:subprocess.run([sys.executable,'-u',str(W/'infer.py'),'--job',job,'--split','test'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  ev=dict(env,SKILLBENCH_EVAL_ROOT=str(W.parent))
  subprocess.run([sys.executable,str(W/'calculate_seven_metrics.py')],env=ev,check=True)
 status('complete',test_complete=True)
if __name__=='__main__':
 try:main()
 except Exception as e:status('failed',error=str(e));raise
