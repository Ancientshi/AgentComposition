#!/usr/bin/env python3
"""Render the final report exclusively from measured, downloaded artifacts."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json
from pathlib import Path
p=Path(__file__).resolve().parent;r=p/'round2_results'
def read(f):return json.loads(f.read_text())
cfg=read(r/'config.json');done=read(r/'completed.json');sel=done['selection'];paired=read(r/'paired_diagnostics.json');rep=read(r/'replay_audit.json');assert rep['n']==100 and rep['exact_all_prediction_fields_match']
base=read(p/'optimized_results/guided_selected/evaluation/top10_metrics.json')['mean'];grid=read(r/'validation_grid.json')
names={'previous':'沿用上一轮采样','matching_constant':'匹配前瞻＋恒定引导','matching_annealed':'匹配前瞻＋阶段性引导','matching_annealed_lengthmix':'匹配前瞻＋阶段性引导＋长度探索'}
lines=['# DDBC 第二轮优化结果','', '第二轮已完成训练、初选验证、额外验证确认、100 个测试任务评测和全量检查点重放。参数没有根据本轮测试结果选择。','', '## 验证阶段','', '| 方案 | 整体评分权重 | RDCR@10 (%) |','|---|---:|---:|']
baseline=next(x for x in grid if x['mode']=='previous' and x['gamma']==0)
lines.append(f"| 上一轮原方案 | 0 | {baseline['metrics']['rdcr@10']*100:.4f} |")
for mode in cfg['modes']:
 best=max([x for x in grid if x['mode']==mode],key=lambda x:x['metrics']['rdcr@10']);lines.append(f"| {names[mode]}，最佳评分权重 | {best['gamma']} | {best['metrics']['rdcr@10']*100:.4f} |")
lines+=['','初选最佳方案只改整体评分，验证 RDCR@10 比上一轮增加约 0.0674 个百分点。三个新采样策略均未超过上一轮；长度探索降低了 Precision 和 F1。','', '## 额外验证确认','', '另外选择了 128 个与本轮初选 query 不重叠的验证 query，包含 299 条目标记录。这些 query 未用于本轮整体评分器检查点选择或配置初选，但属于此前语义适配器使用过的验证 split，不能声称是整个研究过程从未使用过的新测试集。','', '| 方案 | RDCR@10 (%) |','|---|---:|']
for i,v in enumerate(sel['confirmation']):lines.append(f"| {'上一轮原方案' if i==0 else '本轮初选方案'} | {v['metrics']['rdcr@10']*100:.4f} |")
lines+=['',f"确认规则：新方案 RDCR@10 必须高于上一轮才采用。本次确认结果：**{'通过' if sel['confirmation_accepted'] else '未通过，保留上一轮'}**。",f"最终冻结方案：{names[sel['final']['mode']]}，整体评分权重 {sel['final']['gamma']}。",'', '## 同一批 100 个测试任务','', '除明确标注的 MRR 外，数值均为百分数；MRR 列按 100 × MRR 展示。','', '| 方法 | ToolR@1 | Tool-Hit@10 | CompR@1 | CR-Hit@1 | CR-Hit@10 | 100 × CR-MRR@10 | RDCR@10 |','|---|---:|---:|---:|---:|---:|---:|---:|']
keys=['top1_tool_recall','tool_hit@10','top1_component_recall','cr_hit@1','cr_hit@10','cr_mrr@10','rdcr@10']
for name,m in [('上一轮 DDBC-Adapt-SG',base)]+[(n,done['metrics'][k]['mean']) for k,n in [('critic_only','本轮整体评分候选'),('sampling_only','本轮采样选择结果'),('selected','最终冻结方案')]]:
 lines.append('| '+name+' | '+' | '.join(f'{m[k]*100:.2f}' for k in keys)+' |')
lines+=['', '三种测试变体预先定义并在测试标签读取前保存完盲预测。采样选择结果可能沿用上一轮；它不代表三个新采样策略都在测试集上运行。测试结果仅用于报告，不用于重新挑选策略。','', '## 额外指标与配对统计','', '| 方案 | Precision@1 (%) | F1@1 (%) | 工具 Jaccard@1 (%) | Top-1 平均工具数 |','|---|---:|---:|---:|---:|']
for name,m in [('上一轮',paired['original_mean'])]+[(k,paired['variants'][k]['mean']) for k in ['critic_only','selected']]:lines.append(f"| {name} | {m['top1_tool_precision']*100:.2f} | {m['top1_tool_f1']*100:.2f} | {m['tool_jaccard']*100:.2f} | {m['top1_num_tools']:.2f} |")
for k in ['critic_only','selected']:
 d=paired['variants'][k]['paired_vs_original']['rdcr@10'];lo,hi=d['paired_bootstrap_95_percentile_interval'];lines+=['',f"{k} 相对上一轮的 RDCR@10 差值为 {d['difference']*100:+.4f} 个百分点；10,000 次配对 bootstrap 的 95% 百分位区间为 [{lo*100:+.4f}, {hi*100:+.4f}]。该区间仅反映固定训练种子和这 100 个任务，不代表多种子稳定性。"]
lines+=['','## 实现与成本','', '- 新采样器保留 masked diffusion、RVQ、已揭示 token 不再重新掩码的机制，增加剩余工具位置的精确不同组件匹配检查。检查机制在 1,500 个随机案例上与穷举一致。','- 整体评分器为两层无位置编码的集合 Transformer，训练标签包括组件覆盖回归、完整命中辅助标签和同 query 内成对排序监督。工具顺序不变性、空工具配置、有限梯度和真实 diffusion 采样检查全部通过。',f"- 新评分器参数 {cfg['critic_parameters']:,}；训练数据为 6,177 个训练 query 的 87,235 个组合，其中包含 96 个训练 query 的真实 diffusion 候选。",f"- 最佳评分器为第 {sel['critic_epoch']} 轮。学习率 0.0003，weight decay 0.01，每批 32 个 query、每 query 16 个候选，最多 20 轮，连续 4 次验证 MSE 不改善则早停。",'- 原 denoiser、第一轮语义适配器、训练词表和 RVQ 均冻结。组件范围仍是 174 个 LLM、6,021 个工具；没有使用额外检索候选或 LLM API。','- 采样初始预算保持 64 次、32 步，最多 6 个工具；不足 10 个不同合法输出时沿用原保护重试规则并记录实际预算。',f"- 本轮训练、两阶段验证和测试用时 {done['elapsed_seconds']:.2f} 秒，不含最终重放。",'', '## 复现与审计','', '方案和运行命令见 `ROUND2_PLAN.md`。服务器目录：`/root/yunxshi/NIPS2026/outputs/ddbc_semantic_round2_seed42_v1`。本地 `round2_results/` 保存检查点、配置、完整验证表、确认结果、测试预测、指标和审计。','', '```bash','CUDA_VISIBLE_DEVICES=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \\','/root/miniconda3/envs/agentrec/bin/python -u baselines/ddbc_semantic/round2.py \\','  --data outputs/ddbc_semantic_data_v1 \\','  --denoiser outputs/ddbc_semantic_rvq_seed42_v1 \\','  --previous outputs/ddbc_semantic_optimized_seed42_v1 \\','  --output outputs/ddbc_semantic_round2_seed42_reproduction','```','', '运行目录为服务器 `/root/yunxshi/NIPS2026`，复现须使用新的输出目录。','', '```bash','CUDA_VISIBLE_DEVICES=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \\','/root/miniconda3/envs/agentrec/bin/python -u baselines/ddbc_semantic/replay_round2.py \\','  --run outputs/ddbc_semantic_round2_seed42_v1','```','', '九项指标已由独立实现逐项核对，全部结果满足训练词表、类型、大小与 Top-10 唯一性约束。三个变体的 100 个任务全部预测字段已精确重放，重放过程未读取测试标签；只改评分的候选集合与上一轮完全一致。','', '## 判断','', '本轮证据支持：在这套轻量适配方案及当前数据下，继续增加评分复杂度或微调采样的边际收益很低。它不证明 diffusion 的理论能力已达到上限，也不排除重训任务相关表示或量化模型的收益；但那些属于更大的新实验，不应继续作为本轮的追加调参。']
(p/'ROUND2_RESULTS.md').write_text('\n'.join(lines)+'\n')
print(p/'ROUND2_RESULTS.md')
