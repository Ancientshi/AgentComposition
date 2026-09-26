"""Recalculate original seven recommendation metrics from saved ranked predictions."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json,math,statistics,hashlib,os
from pathlib import Path
from collections import defaultdict
W=Path(__file__).resolve().parent;R=Path(os.environ.get('SKILLBENCH_EVAL_ROOT',str(W/'seven_metrics_source')));O=W/'seven_metrics';O.mkdir(exist_ok=True)
FIELDS=['ToolR@1','Tool-Hit@10','CompR@1','CR-Hit@1','CR-Hit@10','CR-MRR@10','RDCR@10']
def read(p):return json.loads(p.read_text())
def calc(pred,gold):
 known=[(m,g) for m,g in gold if m];assert known and all(g for _,g in gold)
 tr=[max(len(p[1]&g)/len(g) for _,g in gold) if p else 0. for p in pred]
 cr=[max((len(p[1]&g)+int(p[0]==m))/(len(g)+1) for m,g in known) if p else 0. for p in pred]
 hits=[bool(p) and any(p[0]==m and g<=p[1] for m,g in known) for p in pred]
 rank=next((i+1 for i,v in enumerate(hits) if v),None)
 return dict(zip(FIELDS,[tr[0],float(any(p and any(g<=p[1] for _,g in gold) for p in pred)),cr[0],float(hits[0]),float(rank is not None),1/rank if rank else 0.,sum(v/math.log2(i+2) for i,v in enumerate(cr))/sum(1/math.log2(i+2) for i in range(10))]))
# Hand-computable checks: extra tools allowed, wrong LLM, rank discount, missing ranks.
p=[('m',{'a','b','extra'})]+[None]*9;v=calc(p,[('m',{'a','b'})]);assert all(v[k]==1 for k in FIELDS[:-1]);assert abs(v['RDCR@10']-1/sum(1/math.log2(i+2) for i in range(10)))<1e-12
v=calc([('wrong',{'a','b'}),('m',{'a','b'})]+[None]*8,[('m',{'a','b'})]);assert v['CompR@1']==2/3 and v['CR-Hit@1']==0 and v['CR-MRR@10']==.5
v=calc([None]*10,[('m',{'a'})]);assert all(x==0 for x in v.values())
x=read(R/'experiments/domain_generalization/frozen/inputs.json');cs={c['token']:c['id'] for c in x['components']};ls={c['token']:c['id'] for c in x['llms']};D=R/'datasets/SkillBench';agents=read(D/'agents/merge.json');rankings=read(D/'canonical/rankings/merge.json')['rankings'];qs=read(D/'canonical/questions/merge.json');manifest=read(R/'experiments/domain_generalization/quickadapt/data/manifest.json');result={};incomplete=[];hashes={}
for method in sorted((R/'experiments/domain_generalization/quickadapt/predictions').iterdir()):
 if not method.is_dir():continue
 for split in ['val','test']:
  p=method/split;ids=manifest['query_ids'][split];found={f.stem for f in (p/'per_sample').glob('*.json')}
  if not (p/'COMPLETE.json').exists() or not set(ids)<=found:
   incomplete.append({'method':method.name,'split':split,'available':len(found),'required':len(ids)});continue
  rows=[];bad=0;missing=0
  for sid in ids:
   file=p/'per_sample'/f'{sid}.json';raw=read(file);assert raw['sample_id']==sid;hashes[str(file.relative_to(R))]=hashlib.sha256(file.read_bytes()).hexdigest();pred=[];seen=set();items=(raw.get('results') or [])[:10];missing+=10-len(items)
   for c in items:
    try:
     ts=c['tools'];assert isinstance(ts,list) and 1<=len(ts)<=10 and all(isinstance(t,str) and t in cs for t in ts) and len(set(ts))==len(ts)
     lm=c.get('llm');assert isinstance(lm,str);key=(lm,tuple(sorted(ts)));assert key not in seen;seen.add(key)
     pred.append((ls.get(lm),{cs[t] for t in ts}))
    except (AssertionError,KeyError,TypeError):pred.append(None);bad+=1
   pred+=[None]*(10-len(pred));gold=[(agents[a]['M'].get('name'),set(agents[a]['T']['tools'])) for a in rankings[sid]];rows.append({'sample_id':sid,'task_id':qs[sid]['metadata']['task_id'],**calc(pred,gold)})
  bytask=defaultdict(list)
  for r in rows:bytask[r['task_id']].append(r)
  query={k:statistics.mean(r[k] for r in rows) for k in FIELDS};task={k:statistics.mean(statistics.mean(r[k] for r in rs) for rs in bytask.values()) for k in FIELDS}
  result.setdefault(split,{})[method.name]={'queries':len(rows),'tasks':len(bytask),'query_macro':query,'task_macro':task,'invalid_slots':bad,'missing_slots':missing,'per_sample':rows}
metadata={'definition':'ToolR: top1 gold-tool recall; Tool-Hit: any top10 covers an observed gold tool set; CompR: top1 recall of tools plus named LLM; CR-Hit: named LLM match and tool coverage; CR-MRR: reciprocal first such rank; RDCR: component recall discounted by 1/log2(rank+1), normalized by all10 rank discounts. Multiple observed gold configurations: maximum/any across gold as in existing SkillsBench evaluator. Invalid/missing slots zero. Additional tools permitted for coverage.','primary_aggregation':'query mean, matching original evaluator; task mean also provided','validation_caveat':'Validation used for stopping and checkpoint selection; not independent test.','incomplete_not_scored':incomplete,'prediction_sha256':hashes}
(O/'metrics.json').write_text(json.dumps({'metadata':metadata,'results':result},ensure_ascii=False,indent=2))
labels={'A_frozen':'v10 不微调','B_explicit_instruction':'v10 仅改instruction','C_source_n15_s42':'v10 原域继续训练 s42','E_base_frozen':'原始Llama 不微调','E_base_n3_s42':'原始Llama＋3任务 s42'}
for agg in ['query_macro','task_macro']:
 lines=['# SkillsBench 七项推荐指标','','百分数列均乘100；CR-MRR@10保留0–1。'+('按query平均，沿用原评估口径。' if agg=='query_macro' else '按task平均，每个任务等权。'),'','验证集参与过早停和checkpoint选择，不能当独立测试。仅对完整split计分，不对部分预测外推。']
 for split,methods in result.items():
  first=next(iter(methods.values()));lines+=['',f'## {"验证集" if split=="val" else "测试集"}：{first["tasks"]}任务 / {first["queries"]} queries','', '| 方法 | '+' | '.join(FIELDS)+' |','|---|'+'---:|'*7]
  for method,r in methods.items():
   vs=r[agg];lines.append('| '+labels.get(method,method)+' | '+' | '.join(f'{vs[k]:.4f}' if k=='CR-MRR@10' else f'{100*vs[k]:.2f}' for k in FIELDS)+' |')
 (O/f'{agg}.md').write_text('\n'.join(lines)+'\n')
print((O/'query_macro.md').read_text())
