"""Exclude Table-1 test queries before retraining Text2Bundle; preserve old run."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json
import subprocess
import sys
from pathlib import Path
import baseline5_run_infer_rag_gpt as b5

ROOT = AC_ROOT
def main():
    b5.ensure_env_cuda_library()
    import baseline3_train_text2bundle_agent_adapted as b3
    old=ROOT/'outputs/text2bundle_adapted_train_seed42'
    output=ROOT/'outputs/text2bundle_top10_disjoint_train_seed42'
    output.mkdir(parents=True,exist_ok=True)
    config=json.loads((old/'config.json').read_text())
    manifest,_=b5.reference_inputs(b5.DEFAULT_SOURCE)
    qids={str(r['qid']) for r in manifest}
    queries={r['query'].strip() for r in manifest}
    raw=b3.load_jsonl(config['data_path'])
    clean=[r for r in raw if str(r.get('qid')) not in qids and str(r.get('query','')).strip() not in queries]
    path=output/'training_excluding_table1.jsonl'
    b3.write_jsonl(path,clean)
    b5.write_json(output/'exclusion_audit.json',{'original_rows':len(raw),'retained_rows':len(clean),
        'removed_rows':len(raw)-len(clean),'excluded_test_qids':sorted(qids),
        'test_manifest_sha256':b5.file_digest(b5.DEFAULT_SOURCE/'sample_manifest.jsonl'),
        'query_overlap_after_filter':0,'original_data_sha256':b5.file_digest(config['data_path'])})
    config['data_path']=str(path)
    config['output_dir']=str(output)
    # Preserve all training hyperparameters, including early stopping.
    argv=[sys.executable,'-u',str(ROOT/'baselines/baseline3_train_text2bundle_agent_adapted.py')]
    for key,value in config.items():
        if key=='dry_run_data': continue
        if isinstance(value,bool): argv.append('--'+('' if value else 'no-')+key)
        else: argv.extend(['--'+key,str(value)])
    print('Filtered training rows',len(raw),'->',len(clean),flush=True)
    subprocess.run(argv,check=True)

if __name__=='__main__': main()
