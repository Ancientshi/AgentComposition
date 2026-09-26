"""Task-owned standard-library HTTP adapter for frozen v4 scoring."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import sys,pathlib,hashlib,threading,json,os,math,traceback
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import torch
from transformers import AutoConfig,AutoTokenizer
P=pathlib.Path;B=P(str(AC_ROOT));sys.path.insert(0,str(B/'training/critic/stage1'))
from compact_input import serialize,compact_inventory,VERSION
sys.path.insert(0,str(AC_ROOT / 'training/generator'))
from compact_context import parse_context
sys.path.insert(0,str(AC_ROOT/'models'));from train_bundle_critic_improved import EasyRecBundleCritic
sys.path.insert(0,str(AC_ROOT / 'models/easyrec'));from model import Easyrec
ck=AC_ROOT/'checkpoints/bundle_critic_reference_v4/best_critic.pt'
expected='becd7447dcac496c1216d55998cf399c2bb9efcc6a9f49a75dd5189e447bc061'
assert hashlib.sha256(ck.read_bytes()).hexdigest()==expected
checkpoint=torch.load(ck,map_location='cpu',weights_only=False);ca=checkpoint['args'];assert checkpoint['epoch']==2
cfg=AutoConfig.from_pretrained(ca['model_dir'],local_files_only=True)
tok=AutoTokenizer.from_pretrained(ca['model_dir'],local_files_only=True)
enc=Easyrec.from_pretrained(ca['model_dir'],config=cfg,local_files_only=True)
model=EasyRecBundleCritic(enc,cfg.hidden_size,ca['head_hidden'],ca['dropout'],ca['normalize_embedding']).cuda()
model.load_state_dict(checkpoint['model_state_dict'],strict=True);model.eval();del checkpoint
metadata={'checkpoint_dir':str(ck.parent),'checkpoint_epoch':2,'weights_sha256':expected,'serialization_version':VERSION,'scoring_dtype':'float16 autocast','batch_size':48,'max_length':512,'critic_version':'v4_epoch2_raw','task':'experiments/end_to_end_initial'}
lock=threading.Lock()
def inventory(text):
    if 'LLM candidates (retrieval order):' in text:return compact_inventory(text)
    ls,ts,bs,ds,ss,_=parse_context(text)
    return {'llms':ls,'tools':ts,'bundles':bs,'descriptions':{**ds,**ss}}
def score(query,candidates,context):
    inv=inventory(context);ids=[serialize(query,c,inv,tok,512)[1] for c in candidates];scores=[]
    with lock,torch.inference_mode():
        for start in range(0,len(ids),48):
            batch=tok.pad({'input_ids':ids[start:start+48]},padding=True,return_tensors='pt').to('cuda')
            with torch.autocast('cuda',dtype=torch.float16):ss=model(batch)
            scores.extend(ss.float().cpu().tolist())
    assert all(math.isfinite(x) for x in scores)
    return scores
class Handler(BaseHTTPRequestHandler):
    def respond(self,value,status=200):
        data=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        self.respond({'status':'ok','model':metadata} if self.path=='/health' else {'error':'not_found'},200 if self.path=='/health' else 404)
    def do_POST(self):
        try:
            assert self.path=='/v1/rerank'
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])));query=payload['query'].strip();cs=payload['candidates'];assert query and cs
            scores=score(query,cs,payload.get('evidence_context',''))
            ranked=[dict(c,raw_score=s,sigmoid_score=1/(1+math.exp(-s))) for c,s in zip(cs,scores)]
            ranked.sort(key=lambda c:c['raw_score'],reverse=True)
            for i,c in enumerate(ranked,1):c['rank']=i
            self.respond({'query':query,'count':len(ranked),'ranking':ranked[:int(payload.get('top_k') or len(ranked))]})
        except Exception as e:
            traceback.print_exc();self.respond({'error':str(e)},500)
P(__file__).with_name('service_pid.json').write_text(json.dumps({'pid':os.getpid(),'checkpoint_sha256':expected,'port':8016})+'\n')
print(json.dumps(metadata),flush=True)
ThreadingHTTPServer(('127.0.0.1',8016),Handler).serve_forever()
