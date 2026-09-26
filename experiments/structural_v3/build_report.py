import argparse,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();p=a.root
s=json.loads((p/'SUMMARY.json').read_text());audit=json.loads((p/'FINAL_AUDIT.json').read_text())
assert s['complete'] and audit['passed']
order=['Free','Greedy','Beam','Critic','Beam+Critic','Critic+Cartesian']
four=['SUCR@10','OGR@10','Oracle-RDCR@10','RDCR@10']
seven=['ToolR@1','Tool-Hit@10','CompR@1','CR-Hit@1','CR-Hit@10','CR-MRR@10','RDCR@10']
v={k:r['percent'] for k,r in s['variants'].items()}
def table(fields):
 return '\n'.join(['| Variant | '+' | '.join(fields)+' |','|---|'+'---:|'*len(fields)]+['| '+k+' | '+' | '.join(f'{v[k][f]:.2f}' for f in fields)+' |' for k in order])
report='''# V10 + critic v3 六组结构消融

全部六组完成，每组固定 100 条测试。本文所有表格数值均为百分数，包括 CR-MRR@10；SUMMARY.json 同时保留 [0,1] 原始值。

## 四个阶段指标

'''+table(four)+'''

## 七个主表指标

'''+table(seven)+'''

Table 1 的新 Ours 对应 **Critic+Cartesian**，不是不扩展候选的 Critic 组。

## 关键对照

'''+f'''- Beam → Critic：RDCR@10 从 {v['Beam']['RDCR@10']:.2f} 到 {v['Critic']['RDCR@10']:.2f}，变化 {v['Critic']['RDCR@10']-v['Beam']['RDCR@10']:+.2f} 个百分点。
- Critic → Beam+Critic：RDCR@10 变化 {v['Beam+Critic']['RDCR@10']-v['Critic']['RDCR@10']:+.2f} 个百分点；Oracle-RDCR@10 变化 {v['Beam+Critic']['Oracle-RDCR@10']-v['Critic']['Oracle-RDCR@10']:+.2f} 个百分点。两组最终都使用纯 v3 critic 排序，差异来自搜索期 critic 参与后返回的候选及其排名，不属于单纯改变同一候选列表的顺序。
- Critic → Critic+Cartesian：CR-Hit@10 从 {v['Critic']['CR-Hit@10']:.2f} 到 {v['Critic+Cartesian']['CR-Hit@10']:.2f}；RDCR@10 从 {v['Critic']['RDCR@10']:.2f} 到 {v['Critic+Cartesian']['RDCR@10']:.2f}。该对照只改变候选扩展，最终评分规则相同。

以上为固定测试集上的描述性对照，没有据此调参、挑选 checkpoint 或进行显著性检验。

## 配置与可复现性

- 生成器：generative_v10_compact；训练时相同的 compact_context，prompt 预算 6144。
- 固定测试 100 条，seed 42，同一检索缓存、查询、target 和顺序；不重新检索，不进行 query rewrite。
- Beam (1,2)，retain .4，LLM/tool branch 4/5，工具数 1–6，complete pool cap 50，返回至多 10 个不同 agent。
- Critic：bundle_critic_terra_v3，epoch 2，端口 8013；权重 SHA256 d7eb8069dc30484f97c1b4679b3228bcbc386b10fde2a195e947d5a41fc1c23f。
- 三个带最终 critic 的组统一按纯 v3 raw score 排序。Beam+Critic 搜索阶段从两个真实工具开始调用 critic，使用原混合评分 .5 critic / .15 generator / .1 长度惩罚；根节点及单工具阶段仍用 generator。
- Free 和 Greedy 均只生成 1 个候选，未复制补足到 10；RDCR 和 Oracle-RDCR 的分母固定为 K=10，其余名次记零。因候选数量不同，Greedy → Beam 的 RDCR 增幅不能全解释为首选候选质量提高，应结合 CompR@1、SUCR 和 OGR 阅读。
- SUCR、OGR、Oracle-RDCR 都基于每组实际返回的同一个 Top-10 列表；Oracle 只重排该列表，不从全池重新选候选。
- Beam 与 Critic 复用已核验的 generator-only 完整完成候选池，移除旧 critic 评分与选出的 Top-10，再重新排序；Cartesian 从本次 v3 Critic Top-10 的去重工具集合 × 检索 Top-10 LLM 构造。原 v3 主表仍复用了旧 critic 选出的工具集合，所以本次完整对齐后的 Ours 不应强行等同于历史的 81.38/78.00/60.02/8.00/24.00/12.18/59.96。
- v3 缓存评分仅在来源结果和 v3 模型指纹完全一致时复用；没有使用旧 critic 分数进行本次候选选择。
- 原 agentrec 环境目录已经缺失。使用现有 tongyi_vllm 环境（torch 2.7.1、transformers 4.56.1、peft 0.18.1）；三个固定样本重新执行 generator-only 搜索，完整候选集合、生成节点数和 generator 分数与原记录完全一致，最大分数差 0。

## 核验

- 六组均为 100 条，样本内容与顺序一致，输出 rank 连续且 agent 去重。
- 独立集合计算与原官方评估器的七项主表指标最大差：{audit['official_evaluator_max_difference']:.3g}。
- 实际 critic 调用阶段、最终排序、beam 上限及 Cartesian 工具集合来源均经过检查，详见 FINAL_AUDIT.json。
- 独立指标检查覆盖：union coverage 与 best-candidate recall 的区分、单候选的固定 K=10 分母、Oracle 对同一列表排列的不变性。

## 产物

- SUMMARY.json / results_percent.csv：全组汇总，前者同时包含原始值和百分数。
- structural_ablation_v10_v3.png / .pdf / .svg：六组四指标图。
- table1_ours_v3.tex：新版主表 Ours 行，MRR 按百分数呈现。
- structural_table.tex：六组四指标 LaTeX 表格行。
- 各组 per_sample、results.jsonl、config.json：完整预测与配置；evaluation/per_sample.json：逐样本指标。

服务器目录：/root/yunxshi/NIPS2026/EXP/structural_v10_criticv3_beam1-2_fixed100。原论文和历史结果未覆盖。
'''
(p/'RESULTS.md').write_text(report)
(p/'table1_ours_v3.tex').write_text('Ours & '+' & '.join(f'{v["Critic+Cartesian"][f]:.2f}' for f in seven)+r' \\'+'\n')
(p/'structural_table.tex').write_text('\n'.join(k.replace('+',' + ')+' & '+' & '.join(f'{v[k][f]:.2f}' for f in four)+r' \\' for k in order)+'\n')
print(p/'RESULTS.md')
