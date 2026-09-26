from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json,shutil,subprocess,sys,time,hashlib
from pathlib import Path
R=Path(str(AC_ROOT));W=R/'experiments/domain_generalization/frozen';O=R/'outputs/SkillBench_FROZEN_97'
expected={q['sample_id'] for q in json.loads((W/'inputs.json').read_text())['queries']}
while {p.stem for p in (O/'ours/per_sample').glob('*.json')}!=expected:time.sleep(10)
for command in [('check_scoring.py',),('evaluate.py',),('evaluate.py','--extract-identifiers'),('audit.py',),('export.py',),('report.py',)]:
 subprocess.run([sys.executable,str(W/command[0]),*command[1:]],check=True)
shutil.copy2(W/'EVALUATION_PROTOCOL.md',O/'EVALUATION_PROTOCOL.md')
code=O/'reproduction';code.mkdir(exist_ok=True)
for filename in ['run.py','evaluate.py','audit.py','export.py','report.py','check_scoring.py','serve_extended.py','inputs.json']:
 shutil.copy2(W/filename,code/filename)
(O/'RUN_STATUS.json').write_text(json.dumps({'status':'complete','dataset':'SkillBench','queries_per_method':97,'methods':4,'training':False,'audit':'PASS'},indent=2))
checks={str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in O.rglob('*') if p.is_file() and p.name!='checksums.json'}
(O/'checksums.json').write_text(json.dumps(checks,indent=2))
archive=shutil.make_archive(str(O),'zip',root_dir=O.parent,base_dir=O.name)
print('COMPLETE',archive,flush=True)
