#!/usr/bin/env python3
"""Apply the project's six-tool legality constraint to a saved DDBC checkpoint."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse,json,os,sys,shutil,hashlib
from pathlib import Path
from types import SimpleNamespace
import run_ddbc_adapt as base

p=argparse.ArgumentParser();p.add_argument('--train-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--replay',action='store_true');a=p.parse_args()
src=a.train_output.resolve();out=a.output.resolve();cfg=json.loads((src/'config.json').read_text());root=Path(cfg['root']);sys.path.insert(0,str(root))
import baseline5_run_infer_rag_gpt as b5
b5.ensure_env_cuda_library()
import torch,joblib,numpy as np
torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
assert base.digest(Path(base.__file__))==cfg['code_sha256']
old_audit=json.loads((src/'split_audit.json').read_text());td=root/'outputs/text2bundle_top10_disjoint_train_seed42'
for k,path in [('source_sha256',td/'training_excluding_table1.jsonl'),('train_qids_sha256',td/'train_qids.txt'),('val_qids_sha256',td/'val_qids.txt'),('test_manifest_sha256',b5.DEFAULT_SOURCE/'sample_manifest.jsonl')]:
    assert base.digest(path)==old_audit[k],f'Changed input {path}'
assert base.digest(src/'model.pt')==json.loads((src/'completed.json').read_text())['checkpoint_sha256']
assert base.digest(Path(base.__file__).parent/'upstream/noise_schedule.py')==cfg['official_noise_sha256']
config={**cfg,'output':str(out),'training_output':str(src),'max_tools':6,'constraint_code_sha256':base.digest(__file__),
    'checkpoint_sha256':base.digest(src/'model.pt'),'text_encoder_sha256':base.digest(src/'text_encoder.joblib'),
    'agent_features_sha256':base.digest(src/'agent_features.npy'),'protocol':'DDBC-Adapt-ID fixed trained checkpoint, max6 legality mask, no retraining'}
if a.replay:
    assert json.loads((out/'config.json').read_text())==config,'Replay configuration changed'
    before={name:base.digest(out/name) for name in ['predictions_blind.jsonl','results.jsonl','catalog.json']}
else:
    assert not out.exists(),'Use a new output path';out.mkdir(parents=True);base.write(out/'config.json',config)
queries,bundles,catalog,manifest,ev=base.prepare(root,out)
legal=[len(c['tool_tokens'])<=6 for c in catalog]
base.write(out/'candidate_pool.json',[c for c,ok in zip(catalog,legal) if ok])
base.write(out/'legality_audit.json',{'model_vocabulary_size':len(catalog),'legal_candidate_pool_size':sum(legal),
    'excluded_over_six_tools':len(legal)-sum(legal),'training_labels_unchanged':True,'constraint_applied_before_sampling':True})
vectorizer,svd=joblib.load(src/'text_encoder.joblib');model=base.build_model(torch,np.load(src/'agent_features.npy')).to(cfg['device'])
model.load_state_dict(torch.load(src/'model.pt',map_location=cfg['device'],weights_only=False)['state_dict']);model.eval()
class LegalModel:
    n=model.n
    def eval(self):model.eval()
    def __call__(self,*args):
        return model(*args).masked_fill(~torch.tensor(legal,device=cfg['device'])[None,None,:],-torch.inf)
base.infer(SimpleNamespace(**cfg),out,LegalModel(),vectorizer,svd,bundles,catalog,manifest,ev)
coverage=[]
for row in manifest:
    llm,tools=ev.parse_agent_text(row['target']);gt=set(tools);gc=gt|{llm};pool=[c for c,ok in zip(catalog,legal) if ok]
    coverage.append({'sample_id':row['sample_id'],'exact_agent_in_catalog':any(c['llm_token']==llm and set(c['tool_tokens'])==gt for c in pool),
        'complete_recall_possible':any(c['llm_token']==llm and gt<=set(c['tool_tokens']) for c in pool),
        'max_component_recall':max(len(gc&(set(c['tool_tokens'])|{c['llm_token']}))/len(gc) for c in pool)})
base.write(out/'evaluation/catalog_coverage.json',{'scope':'legal training candidate pool, at most six tools',
    'mean':{k:sum(r[k] for r in coverage)/len(coverage) for k in ['exact_agent_in_catalog','complete_recall_possible','max_component_recall']},'per_sample':coverage})
for name in ['history.json'] :shutil.copyfile(src/name,out/name)
base.write(out/'completed.json',{'checkpoint_sha256':base.digest(src/'model.pt'),'predictions_sha256':base.digest(out/'predictions_blind.jsonl'),'results_sha256':base.digest(out/'results.jsonl')})
if a.replay:
    after={name:base.digest(out/name) for name in before}
    base.write(out/'replay_audit.json',{'input_fingerprints_verified':True,'byte_identical':before==after,'before':before,'after':after})
    assert before==after
print('Completed max-six-tool protocol.',flush=True)
