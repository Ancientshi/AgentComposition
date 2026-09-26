#!/usr/bin/env python3
"""Post-evaluation paired diagnostics; never changes experiment selection."""
import argparse,json,math,random,re,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--original',type=Path,required=True);a=p.parse_args()
def read(p):return [json.loads(x) for x in p.open()]
def per_sample(row):
 text=row['dataset_example']['target'].split('<SPECIAL_END>')[0].split('Explanation:')[0];llm=re.findall(r'<LLM_[^<>\n\r]+>',text)[0];gt=set(re.findall(r'<<[^<>\n\r]+>>|<TOOL_[^<>\n\r]+>',text))-{'<TOOL_SEP>','<TOOL_EMPTY>'};gc=gt|{llm};cr=[];ths=[];chs=[]
 for z in row['results']:
  pt=set(z['tool_tokens']);pc=pt|{z['llm_token']};cr.append(len(gc&pc)/len(gc));ths.append(gt<=pt);chs.append(gc<=pc)
 z=row['results'][0];pt=set(z['tool_tokens']);pc=pt|{z['llm_token']};rec=len(pt&gt)/len(gt) if gt else 1;prec=len(pt&gt)/len(pt) if pt else float(not gt)
 weights=[1/math.log2(i+2) for i in range(10)]
 return {'top1_tool_recall':rec,'top1_component_recall':cr[0],'tool_hit@10':float(any(ths)),'cr_hit@1':float(chs[0]),'cr_hit@10':float(any(chs)),'cr_mrr@10':next((1/(i+1) for i,v in enumerate(chs) if v),0.),'rdcr@10':sum(w*c for w,c in zip(weights,cr))/sum(weights),'top1_tool_precision':prec,'top1_tool_f1':2*prec*rec/(prec+rec) if prec+rec else 0.,'tool_jaccard':len(pt&gt)/len(pt|gt) if pt|gt else 1.,'component_jaccard':len(pc&gc)/len(pc|gc),'top1_num_tools':len(pt),'top1_llm_accuracy':float(z['llm_token']==llm)}
original=read(a.original/'results.jsonl');base=[per_sample(r) for r in original];rng=random.Random(4242);indices=[[rng.randrange(len(base)) for _ in base] for _ in range(10000)];summary={}
for name in ['rerank_only','guided_selected']:
 rows=read(a.run/name/'results.jsonl');assert [r['dataset_example']['sample_id'] for r in rows]==[r['dataset_example']['sample_id'] for r in original]
 assert [r['dataset_example']['target'] for r in rows]==[r['dataset_example']['target'] for r in original]
 metrics=[per_sample(r) for r in rows];means={k:statistics.mean(r[k] for r in metrics) for k in metrics[0]};official=json.loads((a.run/name/'evaluation/top10_metrics.json').read_text())['mean'];assert all(abs(means[k]-v)<1e-12 for k,v in official.items())
 paired={}
 for k in ['rdcr@10','top1_tool_recall','top1_component_recall']:
  delta=[x[k]-y[k] for x,y in zip(metrics,base)];boots=sorted(sum(delta[i] for i in ix)/len(delta) for ix in indices)
  paired[k]={'difference':statistics.mean(delta),'paired_bootstrap_95_percentile_interval':[boots[249],boots[9749]],'improved_queries':sum(x>1e-12 for x in delta),'worsened_queries':sum(x < -1e-12 for x in delta),'tied_queries':sum(abs(x)<=1e-12 for x in delta)}
 summary[name]={'mean':means,'paired_vs_original':paired}
# Reranking must never add, delete or change any generated component set.
before=read(a.original/'predictions_blind.jsonl');after=read(a.run/'rerank_only/predictions_blind.jsonl')
assert all({tuple(x['components']) for x in b['all_ranked_proposals']}=={tuple(x['components']) for x in c['all_ranked_proposals']} for b,c in zip(before,after))
report={'n':len(base),'original_mean':{k:statistics.mean(r[k] for r in base) for k in base[0]},'variants':summary,'rerank_proposal_sets_unchanged':True,'bootstrap':'10000 paired query resamples, seed 4242; uncertainty conditional on this single training seed and fixed dataset'}
(a.run/'paired_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
