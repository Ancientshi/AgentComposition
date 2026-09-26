from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import sys,pathlib,json,torch
from transformers import AutoTokenizer,AutoConfig
ROOT=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent));sys.path.insert(0,str(AC_ROOT / 'models/easyrec'))
from train_bundle_critic_improved import EasyRecBundleCritic
from model import Easyrec
from compact_input import serialize
BASE=str(AC_EASYREC_MODEL);tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);cfg=AutoConfig.from_pretrained(BASE,local_files_only=True)
r=json.loads((ROOT/'data/cases_train.jsonl').read_text().splitlines()[0]);cs=r['candidates'][:2];ids=[serialize(r['query'],c,r['inventory'],tok)[1] for c in cs];batch=tok.pad({'input_ids':ids},padding=True,return_tensors='pt').to('cuda')
model=EasyRecBundleCritic(Easyrec.from_pretrained(BASE,config=cfg,local_files_only=True),cfg.hidden_size,512,.1,True).cuda();model.train()
with torch.autocast('cuda',dtype=torch.float16):
 ss=model(batch);loss=-torch.nn.functional.logsigmoid(ss[0]-ss[1])
assert torch.isfinite(loss);loss.backward();grad=float(model.scorer[-1].weight.grad.norm());assert grad>0 and grad<1e10
report={'passed':True,'loss':float(loss),'head_gradient_norm':grad,'input_lengths':list(map(len,ids))};(ROOT/'data/smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
