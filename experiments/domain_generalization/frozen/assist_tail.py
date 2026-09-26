from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import os,subprocess,sys,time
from pathlib import Path
# After the original GPU7 shard exits, use that GPU for two not-yet-started
# tail queries from the slowest shard. Existing or active generations are skipped.
p=Path('/proc/3762000/cmdline')
while p.exists() and b'skillbench_eval_work/run.py' in p.read_bytes():time.sleep(10)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
subprocess.run([sys.executable,'-u',str(AC_ROOT / 'experiments/domain_generalization/frozen/run.py'),'ours','--only-indices','92,86'],env=env,check=True)
