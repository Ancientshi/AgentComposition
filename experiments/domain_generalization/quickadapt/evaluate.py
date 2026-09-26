"""Strict identifier metrics; query and task macro; no checkpoint selection here."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json,statistics,random
from pathlib import Path
from collections import defaultdict
R=Path(str(AC_ROOT));W=R/'experiments/domain_generalization/quickadapt'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 m=read(W/'data/manifest.json');x=read(R/'experiments/domain_generalization/frozen/inputs.json');cs={c['token']:c['id'] for c in x['components']};ls={c['token']:c['id'] for c in x['llms']};skills={c['id'] for c in x['components'] if c['type']=='skill'}
 agents=read(R/'datasets/SkillBench/agents/merge.json');rank=read(R/'datasets/SkillBench/canonical/rankings/merge.json')['rankings'];questions=read(R/'datasets/SkillBench/canonical/questions/merge.json')
 methods=['A_frozen','B_explicit_instruction']+[j['name'] for j in m['jobs']];summary={}
 for method in methods:
  seen=set()
  if method.startswith('D_'):
   for s in (W/'data'/f'{method}.jsonl').read_text().splitlines():seen.update(agents[json.loads(s)['agent_id']]['T']['tools'])
  summary[method]={}
  for split in ['val','test']:
   O=W/'predictions'/method/split;assert (O/'COMPLETE.json').exists();rows=[]
   for sid in m['query_ids'][split]:
    raw=read(O/'per_sample'/f'{sid}.json');gold=[(agents[a]['M'].get('name'),set(agents[a]['T']['tools'])) for a in rank[sid]];pred=[];keys=set()
    for p in raw['results'][:10]:
     try:
      ts=p['tools'];assert isinstance(ts,list) and 1<=len(ts)<=10 and len(set(ts))==len(ts) and all(t in cs for t in ts);key=(p['llm'],tuple(sorted(ts)));assert key not in keys;keys.add(key);pred.append((ls.get(p['llm']),{cs[t] for t in ts}))
     except (AssertionError,KeyError,TypeError):pred.append(None)
    pred += [None]*(10-len(pred));top=pred[0];e={'sample_id':sid,'task_id':questions[sid]['metadata']['task_id']}
    e['Set-F1@1']=max(2*len(top[1]&g)/(len(top[1])+len(g)) for _,g in gold) if top else 0
    sg=[g&skills for _,g in gold if g&skills];ps=top[1]&skills if top else set();e['Skill-F1@1']=max(2*len(ps&g)/(len(ps)+len(g)) for g in sg) if sg else None
    e['Tool-Hit@10']=float(any(p and g<=p[1] for p in pred for _,g in gold));e['CR-Hit@10']=float(any(p and lm and p[0]==lm and g<=p[1] for p in pred for lm,g in gold));e['Legal@1']=float(bool(top) and top[0] is not None)
    for kind,allowed in [('seen',seen),('unseen',set(cs.values())-seen)]:
     subset=[g&allowed for _,g in gold if g&allowed];e[kind+'-recall@1']=max(len(top[1]&g)/len(g) for g in subset) if top and subset else (0. if subset else None)
    rows.append(e)
   fields=[k for k in rows[0] if k not in ['sample_id','task_id']];bytask=defaultdict(list)
   for r in rows:bytask[r['task_id']].append(r)
   macro={}
   for k in fields:
    vals=[statistics.mean(r[k] for r in rs if r[k] is not None) for rs in bytask.values() if any(r[k] is not None for r in rs)];macro[k]=statistics.mean(vals) if vals else None
   vals=[statistics.mean(r['Set-F1@1'] for r in rs) for rs in bytask.values()];rng=random.Random(20260919);bs=sorted(statistics.mean(rng.choices(vals,k=len(vals))) for _ in range(2000))
   result={'tasks':len(bytask),'queries':len(rows),'task_macro':macro,'task_bootstrap_F1_95_interval':[bs[49],bs[1949]],'query_macro':{k:statistics.mean(r[k] for r in rows if r[k] is not None) if any(r[k] is not None for r in rows) else None for k in fields},'seen_definition':'actual target adaptation positives only; no target positives for A/B/C','per_sample':rows};write(O/'evaluation.json',result);summary[method][split]={k:v for k,v in result.items() if k!='per_sample'}
  if method.startswith(('C_','D_')):summary[method]['training']=read(W/'runs'/method/'COMPLETE.json')
 groups={}
 for group in ['D_target_n3','D_target_n7','D_target_n15','C_source_n15']:
  members=[name for name in methods if name.startswith(group+'_s')]
  vals=[summary[name]['test']['task_macro']['Set-F1@1'] for name in members]
  groups[group]={'seeds':len(vals),'test_task_macro_F1_mean':statistics.mean(vals),'test_task_macro_F1_seed_sd':statistics.stdev(vals),'mean_gain_over_A':statistics.mean(vals)-summary['A_frozen']['test']['task_macro']['Set-F1@1']}
 write(W/'GROUP_SUMMARY.json',groups)
 write(W/'RESULTS.json',summary)
 lines=['# SkillsBench 少样本快速适配实验','','固定任务划分：15训练池 / 10验证 / 54测试。所有训练从v10重新开始。测试是已有数据重划分，属于探索性证据。','','| 方法 | 测试任务平均F1 | skill F1 | Tool-Hit@10 | 最佳/停止更新 | 停止原因 |','|---|---:|---:|---:|---|---|']
 for method,r in summary.items():
  v=r['test']['task_macro'];t=r.get('training',{});lines.append(f"| {method} | {v['Set-F1@1']:.2%} | {v['Skill-F1@1']:.2%} | {v['Tool-Hit@10']:.2%} | {t.get('best_step',0)}/{t.get('stopped_step',0)} | {t.get('stop_reason','zero update')} |")
 lines+=['','外层模板原本相同。B仅澄清任务分配/输出指令，C为原域复习对照；D为目标域监督。固定候选描述可见，未见组件泛化不等于知识存储于参数。验证推荐指标仅评估选出的checkpoint；完整loss曲线在runs目录。','所有指标为轨迹gold匹配，不是执行成功率。']
 (W/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
