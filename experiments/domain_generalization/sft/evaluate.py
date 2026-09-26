from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json,hashlib,shutil,statistics
from pathlib import Path
R=Path(str(AC_ROOT));W=R/'experiments/domain_generalization/sft';O=R/'outputs/SkillBench_SFT_v1_NO_CRITIC';D=R/'datasets/SkillBench_SFT_v1';S=R/'outputs/SkillBench_FROZEN_97'
def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(D/'split_manifest.json');ids=manifest['query_ids']['test'];assert len(ids)==12
methods=['finetuned','frozen_v10','gpt-5.4-2026-03-05','gpt-5.4-mini-2026-03-17','gpt-5.4-nano-2026-03-17','finetuned_no_length_penalty','frozen_no_length_penalty']
for m in methods:
 for sid in ids:
  if m.startswith('finetuned'):continue
  if m.startswith('gpt-'):src=S/m/'per_sample'/f'{sid}.json'
  else:src=R/'outputs/SkillBench_NO_CRITIC_97'/('without_critic' if m=='frozen_v10' else 'generator_no_length_penalty')/'per_sample'/f'{sid}.json'
  r=read(src)
  if not m.startswith('gpt-'):r.pop('api',None)
  else:assert r['api']['returned_model']==m
  r['source_prediction']=str(src);r['source_sha256']=sha(src);write(O/m/'per_sample'/f'{sid}.json',r)
# Only reuse metric arithmetic. Evaluate all methods on the identical held-out12.
s=(R/'experiments/domain_generalization/frozen/evaluate.py').read_text().split("if '--gpt-only' in sys.argv:\n write",1)[0]
s=s.replace("OUT=ROOT/'outputs/SkillBench_FROZEN_97'","OUT=ROOT/'outputs/SkillBench_SFT_v1_NO_CRITIC'")
s=s.replace("methods=['ours','gpt-5.4-2026-03-05','gpt-5.4-mini-2026-03-17','gpt-5.4-nano-2026-03-17']",f"qs=[q for q in qs if q['sample_id'] in {ids!r}]\nmethods={methods!r}")
s=s.replace("if method!='ours':","if method.startswith('gpt-'):")
ns={};exec(compile(s,str(W/'evaluate.py'),'exec'),ns);summary=ns['summary'];write(O/'metrics.json',summary)
train_tasks=set(manifest['task_ids']['train']);val_tasks=set(manifest['task_ids']['val']);test_tasks=set(manifest['task_ids']['test']);assert not train_tasks&test_tasks and not val_tasks&test_tasks and not train_tasks&val_tasks
for m in methods:
 assert {p.stem for p in (O/m/'per_sample').glob('*.json')}==set(ids)
for sid in ids:
 r=read(O/'finetuned/generation'/f'{sid}.json');g=r['generation'];assert not g['search_trace']['search_critic_enabled'] and not g['search_trace']['final_critic_reranking'] and g['critic_api_events']==[]
 old=read(S/'ours/generation'/f'{sid}.json');assert r['prompt']['text']==old['prompt']['text']
 for key in ['controlled','beam_min_size','beam_max_size','llm_branch_factor','tool_branch_factor','max_tools','num_results','search_length_penalty']:assert r['config'][key]==old['config'][key],key
 assert r['config']['controlled'] and g['search_trace']['llm_candidate_count']==19 and g['search_trace']['tool_candidate_count']==175
 pp=read(O/'finetuned/per_sample'/f'{sid}.json');assert len(pp['results'])==10 and all(c['llm'] for c in pp['results'])
 for n in g['search_trace']['final_rerank_pool_before_cap']:assert n['critic_raw'] is None
prep=read(D/'preparation_report.json');assert sha(D/'train.jsonl')==prep['train_sha256'] and sha(D/'val.jsonl')==prep['val_sha256']
train_rows=[json.loads(l) for l in (D/'train.jsonl').read_text().splitlines()];val_rows=[json.loads(l) for l in (D/'val.jsonl').read_text().splitlines()]
assert not {x['sample_id'] for x in train_rows+val_rows}&set(ids)
complete=read(R/'checkpoints/skillbench_v1_from_v10/COMPLETE.json');protocol=read(R/'checkpoints/skillbench_v1_from_v10/protocol.json');write(O/'training_complete.json',complete);write(O/'training_protocol.json',protocol);write(O/'split_manifest.json',manifest)
audit={'status':'PASS','test_tasks':9,'test_queries':12,'task_and_query_overlap':False,'critic_calls':0,'constrained_generation':True,'beam_sizes':[1,2],'llm_branch_factor':4,'tool_branch_factor':5,'ranked_outputs_per_query':10,'same_prompts_before_and_after':True,'training_data_hashes_verified':True,'checkpoint_selection':'validation completion loss only','initial_checkpoint':'generative_v10_compact','fine_tuned_checkpoint':'skillbench_v1_from_v10/best'};write(O/'AUDIT.json',audit)
# Dataset scope diagnostic: public candidate catalog is available, usage edges are held out.
qd=read(R/'datasets/SkillBench/canonical/questions/merge.json');rank=read(R/'datasets/SkillBench/canonical/rankings/merge.json')['rankings'];agents=read(R/'datasets/SkillBench/agents/merge.json');seen={t for k in manifest['query_ids']['train'] for aid in rank[k] for t in agents[aid]['T']['tools']};test={t for k in ids for aid in rank[k] for t in agents[aid]['T']['tools']};coverage={'train_used_components':len(seen),'test_used_components':len(test),'test_components_without_training_positive':len(test-seen)};train_sets={tuple(sorted(agents[aid]['T']['tools'])) for k in manifest['query_ids']['train'] for aid in rank[k]}
coverage['test_queries_with_any_seen_gold_set']=sum(any(tuple(sorted(agents[aid]['T']['tools'])) in train_sets for aid in rank[k]) for k in ids)
coverage['test_queries_with_only_new_gold_sets']=len(ids)-coverage['test_queries_with_any_seen_gold_set']
write(O/'component_coverage.json',coverage)
# Separate semantic skills from generic execution actions as a diagnostic.
comp=read(R/'experiments/domain_generalization/frozen/inputs.json')['components'];token_to_id={c['token']:c['id'] for c in comp};skill_ids={c['id'] for c in comp if c['type']=='skill'}
skill_diagnostic={}
for m in methods:
 values=[]
 for sid in ids:
  gold_skills=[set(agents[g]['T']['tools'])&skill_ids for g in rank[sid]];gold_skills=[g for g in gold_skills if g]
  if not gold_skills:continue
  row=read(O/m/'per_sample'/f'{sid}.json');pred=set()
  try:
   ts=row['results'][0]['tools'];assert isinstance(ts,list) and 1<=len(ts)<=10 and len(set(ts))==len(ts) and all(t in token_to_id for t in ts)
   pred={token_to_id[t] for t in ts}&skill_ids
  except (AssertionError,IndexError,KeyError,TypeError):pass
  values.append(max(2*len(pred&g)/(len(pred)+len(g)) for g in gold_skills))
 skill_diagnostic[m]={'queries_with_nonempty_skill_gold':len(values),'skill_only_F1@1':statistics.fmean(values) if values else None}
write(O/'skill_only_diagnostic.json',skill_diagnostic)
labels={'finetuned':'SkillBench 微调后（无critic）','frozen_v10':'微调前 v10（无critic）','finetuned_no_length_penalty':'微调后（无critic、无长度惩罚）','frozen_no_length_penalty':'微调前（无critic、无长度惩罚）'}
lines=['# SkillBench：继续微调生成器，不加 critic','','这是小规模先导实验：63/7/9个任务分别用于训练/验证/测试，对应78/7/12条query。从v10生成器继续LoRA微调3轮，保留原词嵌入和输出层；按验证completion loss选checkpoint。没有使用测试标签进行优化或模型选择。','', '| 方法 | 集合F1@1 | ToolR@1 | Tool-Hit@10 | CR-Hit@10 | CR-MRR@10 |','|---|---:|---:|---:|---:|---:|']
for m in methods:
 v=summary[m]['metrics'];lines.append('| '+labels.get(m,m)+' | '+' | '.join(f'{100*v[k]:.2f}%' for k in ['Set-F1@1','Legacy-ToolR@1','SetCoverage-Hit@10','AgentCoverage-Hit@10'])+f" | {v['AgentCoverage-MRR@10']:.4f} |")
lines+=['',f"最佳验证loss：{complete['best_eval_loss']:.6f}；checkpoint：{complete['best_checkpoint']}。",'', '主比较均使用同一beam搜索、原长度惩罚0.1、同一候选目录与完整query。无长度惩罚行仅对各自保存的完成池重新排序，是辅助结果，不用于按测试集选择模型。GPT沿用原始首次返回预测，仅取相同12条query，没有新API调用。GPT是没有历史示例的直接推荐基线，本轮尚未比较GPT＋历史检索。','', '本轮为v10在SkillBench上的继续微调，不能描述为仅使用SkillBench从原始基础模型开始训练。测试含37个实际使用组件，其中15个未在训练正例中使用过，因此混合了新任务与部分组件冷启动；候选描述始终对所有方法公开。样本只有9个独立任务，分数不能直接与此前97条全量结果比较。','', '所有指标是与轨迹gold的匹配，不是实际执行成功率。覆盖允许额外组件；无效标识和缺失排名按原协议处理。完整指标与逐条预测见metrics.json和各方法per_sample目录。']
lines+=['','技能子集诊断：仅保留skill组件，去除通用tool执行动作；仅统计存在非空skill gold的query。']
for m,v in skill_diagnostic.items():lines.append(f"- {labels.get(m,m)}：skill-only F1@1={100*v['skill_only_F1@1']:.2f}%（n={v['queries_with_nonempty_skill_gold']}）")
(O/'RESULTS.md').write_text('\n'.join(lines)+'\n')
code=O/'reproduction';code.mkdir(exist_ok=True)
for f in ['prepare.py','train.py','patch_inference.py','infer.py','evaluate.py','pipeline.py','inference_patch.json']:shutil.copy2(W/f,code/f)
shutil.copy2(R/'checkpoints/skillbench_v1_from_v10/trainer_state.json',O/'trainer_state.json')
write(O/'checksums.json',{str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and p.name!='checksums.json'})
print('\n'.join(lines),flush=True);print('ARCHIVE',shutil.make_archive(str(O),'zip',root_dir=O.parent,base_dir=O.name),flush=True)
