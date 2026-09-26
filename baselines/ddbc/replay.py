#!/usr/bin/env python3
"""Guard all dataset fingerprints before replaying a saved experiment."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();out=a.output.resolve()
c=json.loads((out/'config.json').read_text());audit=json.loads((out/'split_audit.json').read_text());root=Path(c['root'])
td=root/'outputs/text2bundle_top10_disjoint_train_seed42'
for key,path in [('source_sha256',td/'training_excluding_table1.jsonl'),('train_qids_sha256',td/'train_qids.txt'),('val_qids_sha256',td/'val_qids.txt'),('test_manifest_sha256',root/'outputs/baseline4_rag_llama_instruct_seed42_n100/sample_manifest.jsonl')]:
    if sha(path)!=audit[key]:raise ValueError(f'Changed input: {path}')
runner=Path(__file__).parent/'run_ddbc_adapt.py'
if sha(runner)!=c['code_sha256']:raise ValueError('Runner changed')
if sha(runner.parent/'upstream/noise_schedule.py')!=c['official_noise_sha256']:raise ValueError('Official noise code changed')
complete=json.loads((out/'completed.json').read_text())
if sha(out/'model.pt')!=complete['checkpoint_sha256']:raise ValueError('Checkpoint changed')
before={name:sha(out/name) for name in ['predictions_blind.jsonl','results.jsonl','catalog.json']}
cmd=[sys.executable,str(runner),'--root',str(root),'--output',str(out),'--infer-only']
for key in ['seed','epochs','steps','draws','device']:cmd.extend(['--'+key,str(c[key])])
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
subprocess.run(cmd,check=True,cwd=root)
after={name:sha(out/name) for name in before}
report={'input_fingerprints_verified':True,'byte_identical':before==after,'before':before,'after':after}
(out/'replay_audit.json').write_text(json.dumps(report,indent=2)+'\n')
assert before==after,'Replay differs; examine replay_audit.json'
print('All inputs verified; predictions, results, and catalog reproduced byte-for-byte.')
