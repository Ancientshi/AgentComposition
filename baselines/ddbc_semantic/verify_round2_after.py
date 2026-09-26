#!/usr/bin/env python3
"""Finish independent audits if the desktop connection is interrupted."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json,os,subprocess,time
from pathlib import Path
root=Path(str(AC_ROOT));run=root/'outputs/ddbc_semantic_round2_seed42_v1';python='/root/miniconda3/envs/agentrec/bin/python';deadline=time.time()+2700
while not (run/'completed.json').exists():
 if time.time()>deadline:raise TimeoutError('Round-two experiment did not complete within 45 minutes')
 time.sleep(10)
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='4';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for args in [
 ['baselines/ddbc_semantic/replay_round2.py','--run',str(run)],
 ['baselines/ddbc_semantic/summarize_round2.py','--run',str(run),'--original',str(root/'outputs/ddbc_semantic_optimized_seed42_v1/guided_selected')]]:
 subprocess.run([python]+args,cwd=root,env=env,check=True)
(run/'verification_completed.json').write_text(json.dumps({'independent_metrics_and_paired_diagnostics':True,'exact_replay_all_100_queries':True},indent=2)+'\n')
