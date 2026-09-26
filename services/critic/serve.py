"""Serve either stage of the compact bundle critic without background controllers."""
import argparse, hashlib, json, os, sys
from pathlib import Path
from agentcomposition.paths import ROOT, EASYREC_MODEL
sys.path[:0] = [str(ROOT/'services/critic'), str(ROOT/'training/critic/stage1'), str(ROOT/'training/generator'), str(ROOT/'models')]
from compact_input import serialize, compact_inventory, VERSION
from compact_context import parse_context
from serve_bundle_critic_flask import BundleCriticService, create_app
class CompactService(BundleCriticService):
    def score_candidates(self, query, candidates, evidence_context):
        if 'LLM candidates (retrieval order):' in evidence_context:
            inventory = compact_inventory(evidence_context)
        else:
            llms, tools, bundles, descriptions, strengths, _ = parse_context(evidence_context)
            inventory = {'llms':llms, 'tools':tools, 'bundles':bundles, 'descriptions':{**descriptions, **strengths}}
        texts = [serialize(query, candidate, inventory, self.tokenizer, self.max_length)[0] for candidate in candidates]
        return self.score_texts(texts), texts

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8015)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    service=CompactService(checkpoint_dir=str(args.checkpoint_dir),easyrec_code_dir=str(ROOT/'models/easyrec'),device_name=args.device,batch_size=48)
    service.metadata['serialization_version']=VERSION
    create_app(service).run(host='127.0.0.1',port=args.port,threaded=True,use_reloader=False)
if __name__=='__main__':main()
