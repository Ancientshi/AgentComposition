from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import pathlib,json,hashlib,re,math,statistics,csv
base=pathlib.Path(__file__).resolve().parent
p=base/'final_results/outputs/table1_ours_v10_compact_fixedtest100'
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=p/'sample_manifest.jsonl'
assert sha(manifest.read_bytes())=='4dc31d07fea9854d7e826f943f187d960e4f6a02dda2f4f194d5e7c2dd49bd5a'
samples=[json.loads(x) for x in manifest.read_text().splitlines()]
assert len(samples)==100 and len(list((p/'per_sample').glob('*.json')))==100
cfg=json.loads((p/'ours_config.json').read_text()); fingerprint=sha(json.dumps(cfg,sort_keys=True,ensure_ascii=False).encode())
rows=[]
for s in samples:
 r=json.loads((p/'per_sample'/f"{s['sample_id']}.json").read_text())
 assert r['ok'] and r['dataset_example']==s and r['config_sha256']==fingerprint
 preds=r['results']; assert len(preds)==10 and [x['rank'] for x in preds]==list(range(1,11))
 assert len({(x['llm_token'],tuple(sorted(x['tool_tokens']))) for x in preds})==10
 g=r['generation']; assert g['search_critic_enabled'] is False and g['final_critic_reranking'] is True
 assert len(g['critic_api_events'])==1 and not g['critic_api_events'][0].get('error')
 target=s['target'].split('<SPECIAL_END>')[0]
 llm=re.findall(r'<LLM_[^<>\n\r]+>',target); assert len(llm)==1
 gt=set(re.findall(r'<<[^<>\n\r]+>>',target))|set(re.findall(r'<TOOL_[^<>\n\r]+>',target))
 gt-= {'<TOOL_SEP>','<TOOL_EMPTY>'}; comps=gt|set(llm)
 tr=[];cr=[];th=[];ch=[]
 for x in preds:
  pt=set(x['tool_tokens']);assert 1<=len(pt)<=6
  tr.append(len(gt&pt)/len(gt) if gt else 1.)
  cr.append(len(comps&(pt|{x['llm_token']}))/len(comps))
  th.append(gt<=pt);ch.append(gt<=pt and x['llm_token']==llm[0])
 rows.append([tr[0],float(any(th)),cr[0],float(ch[0]),float(any(ch)),next((1/(i+1) for i,v in enumerate(ch) if v),0),sum(v/math.log2(i+2) for i,v in enumerate(cr))/sum(1/math.log2(i+2) for i in range(10))])
labels=['ToolR@1','Tool-Hit@10','CompR@1','CR-Hit@1','CR-Hit@10','CR-MRR@10','RDCR@10']
raw=dict(zip(labels,map(statistics.fmean,zip(*rows))))
reported=json.loads((p/'evaluation/ours_ranked_recall_metrics.json').read_text())
assert all(abs(raw[k]-reported['raw'][k])<1e-12 for k in labels)
table={k:v*(1 if k=='CR-MRR@10' else 100) for k,v in raw.items()}
out=base/'ours_final';out.mkdir(exist_ok=True)
audit={'passed':True,'n':100,'unique_candidates_per_sample':10,'test_manifest_sha256':sha(manifest.read_bytes()),'config_sha256':fingerprint,'adapter_weights_sha256':cfg['adapter_weights_sha256'],'independent_raw':raw,'table1':table,'max_metric_difference':max(abs(raw[k]-reported['raw'][k]) for k in labels)}
(out/'ours_metrics.json').write_text(json.dumps(audit,indent=2))
with (out/'ours_table1.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['Method']+labels);w.writerow(['Ours']+[f'{table[k]:.4f}' if k=='CR-MRR@10' else f'{table[k]:.2f}' for k in labels])
(out/'ours_table1.tex').write_text('Ours & '+' & '.join(f'{table[k]:.4f}' if k=='CR-MRR@10' else f'{table[k]:.2f}' for k in labels)+r' \\'+'\n')
print(json.dumps(audit,indent=2))
