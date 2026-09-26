"""Run clean Top-10 inference once the current retraining process succeeds."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = AC_ROOT
ap=argparse.ArgumentParser()
ap.add_argument('--training_pid',type=int,required=True)
args=ap.parse_args()
training=ROOT/'outputs/text2bundle_top10_disjoint_train_seed42'
output=ROOT/'outputs/baseline3_text2bundle_top10_seed42_n100'
status=ROOT/'outputs/top10_jobs/text2bundle_pipeline_status.json'
def write(state,**extra):
    status.write_text(json.dumps({'state':state,'training_pid':args.training_pid,
        'output':str(output),'updated_at':time.time(),**extra},indent=2))
proc=Path('/proc')/str(args.training_pid)
start=(proc/'stat').read_text().split()[21] if proc.exists() else None
write('waiting_for_training')
while not (training/'training_summary.json').exists():
    try: alive=(proc/'stat').read_text().split()[21]==start
    except FileNotFoundError: alive=False
    if not alive:
        write('failed',reason='Training exited without training_summary.json')
        raise SystemExit(1)
    time.sleep(30)
write('running_inference')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
command=[sys.executable,'-u',str(ROOT/'baselines/baseline3_infer_top10.py'),'--experiment_dir',str(output)]
result=subprocess.run(command,cwd=ROOT,env=env)
write('complete' if result.returncode==0 else 'failed',returncode=result.returncode)
raise SystemExit(result.returncode)
