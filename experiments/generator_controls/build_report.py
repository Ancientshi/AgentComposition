from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json,pathlib
from analyze_results import label
P=pathlib.Path(__file__).resolve().parent
s=json.loads((P/'results/summary.json').read_text());sub=json.loads((P/'subgroups.json').read_text());matched=json.loads((P/'matched_pool_summary.json').read_text());audit=json.loads((P/'FINAL_AUDIT.json').read_text())
fields=['RDCR@10','RDCP@10','CR-Hit@10','Exact-Config-Hit@10']
main=['history_full_retrieval','history_full_v4','ours_no_expansion_v4','ours_cartesian_v4','baseline5_rag_gpt_5_6_terra_top10_seed42_n100_v4','baseline_siliconflow_rag_Pro_moonshotai_Kimi_K2_6_top10_seed42_n100_v4']
def table(names,data,counts=False):
 out='| 方法 | '+('平均候选数 | ' if counts else '')+' | '.join(fields)+' |\n|---|'+('---:|' if counts else '')+'---:|---:|---:|---:|\n'
 for n in names:
  r=data[n];out+='| '+label(n)+' | '+(f"{r['mean_pool_count']:.2f} | " if counts else '')+' | '.join(f"{r['metrics'][k]*100:.2f}" for k in fields)+' |\n'
 return out
report='''# 生成器、历史 bundle 与 RAG 的同 critic 对照

全部完成。使用用户已保留的 **critic V4 第 2 轮，raw score，alpha=1**，没有使用在线 V3 服务，没有重新训练、检索或调用生成 LLM。固定 100 条测试查询、reference 和检索证据；新文件单独保存，原论文和旧实验未覆盖。

## 主要判断

这组实验值得加入，但结论应收窄：**生成式候选在当前配置下提高了排名加权的组件召回；现有证据不足以说明生成器不可替代，或在新工具组合恢复上优于历史 bundle 检索。** 历史 bundle + V4 是必须保留的强对照。等候选数量时，生成式方案与历史组合的差距更小。

## 1. 同一 V4 下的主要结果

以下所有指标均乘 100。CR-Hit@10 沿用论文的 complete-recall 定义：LLM 正确且包含全部参考工具，允许额外工具。Exact-Config-Hit@10 是本次补充诊断，要求 LLM 与无序工具集合均完全相同；不能与原 CR-Hit 混称。

'''+table(main,s,True)+'''
- 历史组合使用 Top-5 历史记录中的完整工具 bundle × 相同 Top-10 LLM，再去重。V4 对其 RDCR 的点估计增益为 1.98 个百分点，95% 配对 bootstrap 区间为 [−0.20, 4.13]。
- 控制版 Ours + Cartesian 对历史组合 + V4 的 RDCR 增益为 **5.16 个百分点 [2.01, 8.34]**，但 RDCP 低 1.61 点，CR-Hit 低 3 点，Exact-Config-Hit 低 8 点。不能据此宣称全面领先。
- 两者排序前的候选池也不同：历史组合平均 33.50 个，生成式 Cartesian 平均 78.60 个。历史池 complete-recall 覆盖率为 40%，生成池为 37%；精确配置覆盖率分别为 38% 和 32%。生成器没有在这批样本上扩大完整正确配置的总体可达覆盖。
- 与 Kimi + V4 相比，Ours 的 RDCR 高 1.44 点，区间 [−1.70, 4.85]；RDCP 高 15.33 点 [11.77, 18.89]。应体现 recall/precision 的取舍，而非只报其中一项。

所有区间使用固定 seed=20260921、10,000 次逐查询配对重采样，为未作多重比较校正的描述性 95% 区间，不用于选模型或改权重。

## 2. 固定评分规则的 Cartesian 消融

两组使用同一 V4 raw score、同一 evidence 和 canonical tie break；不加 generator 分数或工具长度惩罚。先对生成器的完整完成候选池评分，取得当前 V4 的 Top-10，提取去重工具集合并与同源 LLM10 交叉组合，再以同一 V4 评分选 Top-10。

| 指标 | 不扩展 | Cartesian | 差异 / 百分点 | 95% 配对区间 |
|---|---:|---:|---:|---|
| RDCR@10 | 56.54 | 57.97 | +1.43 | [−0.09, 2.97] |
| RDCP@10 | 45.95 | 49.74 | +3.79 | [2.29, 5.31] |
| CR-Hit@10 | 21.00 | 27.00 | +6.00 | [−3.00, 15.00] |
| Exact-Config-Hit@10 | 17.00 | 20.00 | +3.00 | [−6.00, 11.00] |

RDCP 的区间为正；其余上述增益仍有较大不确定性。CR-Hit 的逐查询变化为新增命中 13 条、丢失命中 7 条，并非单调改善。CR-Hit@1 从 10% 降为 8%。

**与原 V4 行的区别：** 已有 V4 结果使用较早上游选出的工具集合，本次额外保留了该视图，并精确复现 RDCR=59.02、RDCP=48.18、CR-Hit=30%。本次固定评分规则的控制版使用当前 V4 自己选出的工具种子，结果为 57.97/49.74/27%。两个视图不能混作同一次消融，也没有因为旧视图某些指标更好而替换本次预定比较。

## 3. RAG 原输出与 V4 重排

每个模型保持原来的十个配置，V4 只改变顺序。所有 10 个模型的候选身份、CR-Hit@10、Tool-Hit@10 均逐查询保持不变。下表展示排序敏感指标。

| RAG 模型 | 原 RDCR | +V4 RDCR | 原 RDCP | +V4 RDCP |
|---|---:|---:|---:|---:|
'''
for name in [n[:-9] for n in s if n.endswith('_original')]:
 a=s[name+'_original']['metrics'];b=s[name+'_v4']['metrics'];report+=f"| {label(name)} | {a['RDCR@10']*100:.2f} | {b['RDCR@10']*100:.2f} | {a['RDCP@10']*100:.2f} | {b['RDCP@10']*100:.2f} |\n"
report+='''
重排对 RDCR 的变化介于 −0.39 与 +0.77 个百分点之间，未使这些保存的 RAG 列表整体大幅改善。这仅评价既有十候选的重排，不能推断 critic 从更大的 RAG 池选候选也无效。

## 4. 未见 bundle/configuration

已核对生成器 target、训练提示中可见的完整历史 bundles、V3 critic 训练及 V4 新训练候选。V3 实际用于训练的 pair endpoints 按原始阈值与随机种子重建；V4 endpoints 根据保存的 pairs 读取。同时报告更保守的完整候选池暴露。验证集单独统计，避免把“训练未见”误当作“整个开发过程未见”。

| 暴露口径 | 未见参考 bundle | 未见参考 configuration |
|---|---:|---:|
| 仅生成器训练 target | 75 | 89 |
| 生成器训练 target + V3/V4 实际训练候选 | 71 | 85 |
| 再计入生成器训练提示中的历史 bundles及 critic 全候选池 | 31 | 85 |
| 再计入相应验证集暴露 | 30 | 85 |

Configuration 暴露指显式配对的 LLM 与工具集合；提示中的独立 LLM 列表不被当作训练过所有笛卡尔配置。Bundle 暴露则纳入提示中的完整工具集合。这里只审计任务训练/验证数据，不声称这些内容从未出现在底座或上游 encoder 预训练中。对标签前缀大小写和空白作格式规范化后，上表计数不变。

最严格的 30 条未见 bundle 查询中，有 **16 条全部组件在任务训练中见过**；其中一条为空工具、一条为单工具目标，排除后得到 **14 条包含至少两个工具的组合泛化子集**。另外 14 条还包含未见组件。这 30 条中有 **20 条的完整 reference bundle 可以直接从本次 Top-5 历史检索中取到**，因此训练未见不等于必须重新组合。

### 全部严格未见 bundle：30 查询

'''+table(['history_full_v4','ours_no_expansion_v4','ours_cartesian_v4','baseline5_rag_gpt_5_6_terra_top10_seed42_n100_v4','baseline_siliconflow_rag_Pro_moonshotai_Kimi_K2_6_top10_seed42_n100_v4'],sub['results']['pipeline_train_valid_conservative_unseen_bundle'])+'''
### 严格未见多工具 bundle 且组件均见过：14 查询

'''+table(['history_full_v4','ours_cartesian_v4','baseline5_rag_gpt_5_6_terra_top10_seed42_n100_v4','baseline_siliconflow_rag_Pro_moonshotai_Kimi_K2_6_top10_seed42_n100_v4'],sub['results']['strict_unseen_multitool_bundle_known_components'])+'''
这些是小样本、事后分组的描述性结果，不是预先按 bundle 划分的留出实验。完整分组结果和 query IDs 已保存。

**最需要正视的边界：** 在“严格训练/验证未见且本次 Top-5 未直接返回 reference bundle”的 10 条查询中，Ours 的 RDCR 为 12.95，历史组合为 15.77，Ours 的 Exact-Config-Hit 为 0。若再要求全部组件见过且至少包含两个工具，只剩 2 条，不足以支撑强组合泛化结论。在全部 16 条 reference bundle 不在 Top-5 的查询中，控制版 Ours 也没有精确配置命中。不能把现有结果写成已验证从历史库外构造正确新 bundle 的优势。

## 5. 等数量候选池诊断（事后补充）

主实验后发现候选池数量不同，因此按独立于 reference 的固定规则追加诊断：每条查询两边均选相同 B 个工具集合，再与同一 LLM10 配对。历史集合按检索序；生成集合按原 generator-only 路径分数（平均 log probability 减 0.1×工具数）选取，不用 V4 分数提前筛选。B 为两边可用去重集合数的较小值，历史集合沿用 1–6 工具合法范围。两边平均均为 **33.40 个候选**。

| 方法 | RDCR@10 | RDCP@10 | CR-Hit@10 | Exact-Config-Hit@10 |
|---|---:|---:|---:|---:|
'''
for name,r in matched['results'].items():report+='| '+('历史组合 + V4' if name=='matched_history_v4' else '生成组合 + V4')+' | '+' | '.join(f"{r['metrics'][k]*100:.2f}" for k in fields)+' |\n'
report+='\n生成组合的 RDCR 高 1.88 点 [−1.57, 5.21]，RDCP 高 0.63 点 [−2.13, 3.24]，CR-Hit 高 1 点 [−5, 7]。这削弱了将主实验 RDCR 差距全部归因于生成器构造能力的解释。该诊断只匹配最终评分池大小，没有匹配前面的生成搜索成本；候选预选规则也不同，因此不提供纯粹的单因素因果结论。它不是新的主表优胜配置，未据此重新调参。\n\n## 6. 输入边界与验证\n\n- checkpoint SHA256：`becd7447dcac496c1216d55998cf399c2bb9efcc6a9f49a75dd5189e447bc061`。epoch=2、alpha=1，float16 autocast，batch=48；相同 query/candidate 在所有方法中复用同一分数。\n- 共评分 20,398 个按查询去重的配置，20,266 个在原 1–6 工具范围内；这些配置的序列化文本、token IDs 和元数据与原 serializer 逐项完全一致。\n- 其余为 122 个去重空工具配置及一个 20 工具历史 bundle 对应的 10 个 LLM 配置。只在隔离的内存副本中解除 cardinality assertion，保留原格式、512-token 上限和完整 identifier。全部可序列化，无截断候选身份。这些输入超出 critic 的训练工具数范围，不应暗示经过域内验证。\n- 历史完整池与删除整个 20 工具 bundle 后的合法池，最终 V4 Top-10 逐条完全相同。RAG 的无空工具子集对照另外保存，主要排序结论不变。\n- 缓存的 10 条查询存在空格或 `<Tool_`/`<TOOL_` wrapper 差异；底层检索 bundle 在格式规范化后对应。主结果保留各原方法的标识符和评测规则，不用 reference 修复名字，也不将未知 RAG 标识替换为 gold。\n- 27 组 × 100 条、共 2,700 个排名结果独立复算，全部指标最大误差 0；每条返回 10 个不同配置。10 个 RAG 模型的候选及 Hit 不变性通过；原 V4 历史视图所有既有指标精确复现。\n- 新增模型训练、检索、生成 LLM 调用均为 0。离线 GPU 打分及当轮逐样本评测约 17 秒，不包含输入准备、历史生成、模型加载和完整审计，不能作为端到端延迟。\n\n## 建议如何用于论文\n\n1. 加入历史 bundle×LLM + V4 基线，RAG+V4 放附录；同时保留 precision、recall 和 exact recovery 诊断。\n2. 用本次同评分规则的 Cartesian 对照支撑扩展贡献，保留不确定性；旧 V4 行与新控制版明确区分。\n3. 可以报告未见组合的事后分组结果，但将其定位为有限证据。若论文坚持“生成未见组合是主要优势”，仍需预先按组合留出的新协议，并区分完整目标 bundle 可直接检索与必须重新组合的样本。当前 2 条多工具纯组合、不可直接取回的样本不足以代替该实验。\n4. 不根据这 100 条测试选择新的候选配额、critic 权重或上游工具种子。\n\n## 文件\n\n- `all_results_percent.csv`：27 组的完整指标。\n- `results/predictions.jsonl`、`results/scores/`：逐查询输出及冻结分数。\n- `paired_bootstrap.json`：全部预定配对区间。\n- `subgroups.json`、`exposure/per_query.jsonl`：未见组合定义、样本列表及所有方法分组指标。\n- `matched_pool_summary.json`：等候选数量诊断及区间。\n- `rag_in_range_sensitivity.json`：不含空工具输出的配对子集。\n- `FINAL_AUDIT.json`、`legacy_v4_reproduction.json`：独立复算与历史结果复现。\n- `manuscript_snippet.tex`：可审阅的独立 LaTeX 片段，未插入原稿。\n\n服务器完整记录位于 `/root/yunxshi/NIPS2026/experiments/generator_controls/`，包含额外的训练暴露索引和来源文件哈希。\n'
(P/'REPORT.md').write_text(report)
tex=r'''\paragraph{Controlling for the critic.}
We compare candidate construction methods using the same frozen V4 critic
and raw-score ranking. Historical bundles are retained intact and crossed
with the same ten retrieved backbone LLMs. We also rerank each RAG model's
saved ten configurations without changing their identities. Table~\ref{tab:shared_critic_controls}
reports results on the same 100 queries. The critic uses the original
serialization for all one-to-six-tool configurations. Empty and oversized
bundles retain the same format and input budget, with the cardinality
restriction relaxed; they are outside the training tool-count range.
Removing the single oversized historical bundle leaves its final Top-10
outputs unchanged. Complete-recall hits allow extra
tools; exact configuration hits require both the backbone and tool set to match.

\begin{table}[t]
\centering
\caption{Shared-critic controls. All scores are percentages. Exact-Hit denotes
exact configuration recovery at rank 10. The two Ours rows use the same V4
ranking rule; Cartesian expansion uses tool sets from the current V4 Top-10.}
\label{tab:shared_critic_controls}
\small
\begin{tabular}{lrrrr}
\toprule
Method & RDCR@10 & RDCP@10 & CR-Hit@10 & Exact-Hit@10 \\
\midrule
'''
short={'history_full_retrieval':'History $\\times$ LLM (rank sum)','history_full_v4':'History $\\times$ LLM + V4','ours_no_expansion_v4':'Ours (no expansion)','ours_cartesian_v4':'Ours (Cartesian)'}
for n in main:tex+=short.get(n,label(n))+' & '+' & '.join(f"{s[n]['metrics'][k]*100:.2f}" for k in fields)+r' \\'+'\n'
tex+=r'''\bottomrule
\end{tabular}
\end{table}

Ours with expansion exceeds historical-bundle reranking by 5.16 points in
RDCR@10, but has lower RDCP@10 and exact configuration recovery.
The final candidate pools average 78.60 and 33.50 configurations,
respectively. In an additional post-hoc comparison with equal final pool
sizes, generated and historical candidates reach RDCR@10 of 54.69 and
52.81; the paired difference has a 95\% bootstrap interval of
$[-1.57,5.21]$ points. This comparison controls final pool size but not
upstream generation cost. The results support a ranked component-recall
benefit in the main setting, rather than uniform superiority in configuration recovery.

\paragraph{Unseen combinations.}
We additionally stratify the fixed test set by task-training exposure.
Thirty reference tool sets are absent from generator training and validation
targets, historical bundles shown in those prompts, and the V3/V4 critic
training and validation candidate pools. Sixteen of these queries use only
components observed in task training. Removing one empty-tool and one
single-tool reference leaves fourteen multi-tool queries. On these queries,
Ours and historical-bundle reranking obtain RDCR@10 of 64.91 and 54.44,
and both recover an exact configuration on 28.57\% of queries. However, twenty of the
thirty unseen reference bundles are directly available in the retrieved
Top-5. On the remaining ten queries, Ours obtains no exact configuration
hits. These small, post-hoc subsets provide limited evidence for generalizing
to new combinations and do not replace a prospective combination-disjoint evaluation.
'''
(P/'manuscript_snippet.tex').write_text(tex)
print(P/'REPORT.md')
