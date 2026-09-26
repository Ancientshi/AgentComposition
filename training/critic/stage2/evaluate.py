"""Offline, frozen-checkpoint comparison on saved Cartesian candidate pools."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse,importlib.util,json,os,sys,statistics
from pathlib import Path
from core import blend_scores,parse_target,ranked_metrics,signature
from prepare import jsonl,sha

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--v3-root',type=Path,required=True)
    ap.add_argument('--v3-checkpoint',type=Path,required=True)
    ap.add_argument('--v4-output',type=Path,required=True)
    ap.add_argument('--critic-code',type=Path,required=True)
    ap.add_argument('--easyrec-code',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Refusing to overwrite evaluation')
    cfg=json.loads((args.v4_output/'ranking_config.json').read_text())
    if cfg['baseline_sha256']!=sha(args.v3_checkpoint):raise ValueError('V3 checkpoint identity changed')
    import torch
    from transformers import AutoConfig,AutoTokenizer
    sys.path.insert(0,str(args.v3_root));from compact_input import serialize,compact_inventory
    # Same inventory parsing as V3 serve.py, without importing a Flask service.
    sys.path.insert(0,str(AC_ROOT / 'training/generator'))
    from compact_context import parse_context
    def inventory(text):
        if 'LLM candidates (retrieval order):' in text:return compact_inventory(text)
        llms,tools,bundles,desc,strengths,_=parse_context(text)
        return {'llms':llms,'tools':tools,'bundles':bundles,'descriptions':{**desc,**strengths}}
    sys.path.insert(0,str(args.critic_code));from train_bundle_critic_improved import EasyRecBundleCritic
    sys.path.insert(0,str(args.easyrec_code));from model import Easyrec
    sources=list(jsonl(args.source/'results.jsonl'));prepared=[]
    for row in sources:
        inv=inventory(row['context']['text']);llms=list(dict.fromkeys(inv['llms']))[:10]
        if len(llms)!=10:raise ValueError('Expected the saved retrieved LLM10')
        seed={tuple(sorted(set(c['tool_tokens']))) for c in row['results'][:10]}
        pool=row['generation']['search_trace']['final_rerank_pool_before_cap']
        toolsets=list(dict.fromkeys(tuple(sorted(set(n['tools']))) for n in pool))
        if not seed<=set(toolsets):raise ValueError('Seed outside saved completed pool')
        cs=[{'llm':m,'tools':list(t)} for t in toolsets for m in llms]
        prepared.append({'qid':row['dataset_example']['qid'],'query':row['query'],'inventory':inv,
                         'references':[parse_target(row['dataset_example']['target'])],
                         'candidates':cs,'seed_indices':[i for i,c in enumerate(cs) if tuple(c['tools']) in seed]})
    args.output.mkdir(parents=True)
    all_scores={}
    for label,path in [('v3',args.v3_checkpoint),('v4',args.v4_output/'best_critic.pt')]:
        ck=torch.load(path,map_location='cpu',weights_only=False);ca=ck['args'];base=ca['model_dir']
        config=AutoConfig.from_pretrained(base,local_files_only=True);tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
        enc=Easyrec.from_pretrained(base,config=config,local_files_only=True)
        model=EasyRecBundleCritic(enc,config.hidden_size,ca['head_hidden'],ca['dropout'],ca['normalize_embedding']).cuda()
        model.load_state_dict(ck['model_state_dict'],strict=True);model.eval();del ck
        result={}
        with torch.inference_mode():
            for qi,r in enumerate(prepared):
                ids=[serialize(r['query'],c,r['inventory'],tok,512)[1] for c in r['candidates']];scores=[]
                for j in range(0,len(ids),48):
                    batch=tok.pad({'input_ids':ids[j:j+48]},padding=True,return_tensors='pt').to('cuda')
                    with torch.autocast('cuda',dtype=torch.float16):ss=model(batch)
                    scores.extend(ss.float().cpu().tolist())
                result[r['qid']]=scores
                if qi%10==0:print(json.dumps({'model':label,'queries':qi+1,'total':len(prepared)}),flush=True)
        all_scores[label]=result;del model;torch.cuda.empty_cache()
    results={};predictions=[]
    for view in ['upstream_top10_toolsets','all_completed_toolsets']:
        bymethod={m:[] for m in ['v3','v4_raw','v4_selected_blend']}
        lengths={m:[] for m in bymethod}
        for r in prepared:
            idx=r['seed_indices'] if view=='upstream_top10_toolsets' else list(range(len(r['candidates'])))
            cs=[r['candidates'][i] for i in idx]
            b=[all_scores['v3'][r['qid']][i] for i in idx];v=[all_scores['v4'][r['qid']][i] for i in idx]
            scores={'v3':b,'v4_raw':v,'v4_selected_blend':blend_scores(b,v,cfg['alpha'])}
            for method,ss in scores.items():
                met=ranked_metrics(cs,ss,r['references']);bymethod[method].append(met)
                order=sorted(range(len(cs)),key=lambda i:(-ss[i],signature(cs[i])))[:10]
                lengths[method].extend(len(cs[i]['tools']) for i in order)
                predictions.append({'qid':r['qid'],'view':view,'method':method,'metrics':met,
                                    'results':[{'score':ss[i],**cs[i]} for i in order]})
        results[view]={m:{'n':len(vals),'metrics':{k:statistics.fmean(v[k] for v in vals) for k in vals[0]},
                         'mean_tool_count_top10':statistics.fmean(lengths[m])} for m,vals in bymethod.items()}
    out={'results':results,'ranking_config':cfg,'test_used_for_selection':False,'scoring_dtype':'autocast float16, same as checkpoint validation',
         'source_sha256':sha(args.source/'results.jsonl'),'v3_sha256':sha(args.v3_checkpoint),'v4_sha256':sha(args.v4_output/'best_critic.pt'),
         'comparison':'Frozen V3 and validation-selected V4 on identical saved candidate pools; historical V2 is not the clean baseline.'}
    (args.output/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    with (args.output/'predictions.jsonl').open('w') as f:
        for p in predictions:f.write(json.dumps(p,ensure_ascii=False)+'\n')
    print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
