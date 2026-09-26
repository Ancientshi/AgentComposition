from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json
from pathlib import Path
O=Path(str(AC_ROOT / 'outputs/SkillBench_FROZEN_97'))
fields=['Legacy-ToolR@1','SetCoverage-Hit@10','Legacy-CompR@1','AgentCoverage-Hit@1','AgentCoverage-Hit@10','AgentCoverage-MRR@10','Legacy-RDCR@10']
text=['# SkillBench：四方法冻结推理结果','','数据：1,000 条成功轨迹，97 条去重 query，79 个底层任务。四种方法使用相同的完整 query 与候选组件说明，没有训练。','', 'gold：每个 query 最多 10 个不同的成功配置；轨迹 reward 均为 1.0，等分按固定哈希选择。集合是实际观察到的 skill/tool 访问集合，不是经过消融证明的最小配置。','', '下表沿用原实验的覆盖式 recall 定义，并扩展为对多个 gold 取最佳匹配。Tool-Hit 和 CR-Hit 允许预测集合含额外组件；严格集合相等指标另存于 RESULTS 文件。RDCR 的折扣为 1/log2(rank+1)。']
for suffix,label in [('', '主要结果：严格标识匹配'),('_identifier_extraction','辅助结果：提取明确标识')]:
 summary=json.loads((O/('metrics'+suffix+'.json')).read_text())
 text+=['','## '+label,'','| 方法 | ToolR@1 | Tool-Hit@10 | CompR@1 | CR-Hit@1 | CR-Hit@10 | CR-MRR@10 | RDCR@10 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for method,r in summary.items():
  name={'ours':'Ours','gpt-5.4-2026-03-05':'GPT-5.4','gpt-5.4-mini-2026-03-17':'GPT-5.4 mini','gpt-5.4-nano-2026-03-17':'GPT-5.4 nano'}[method]
  v=r['metrics'];text.append('| '+name+' | '+' | '.join(f'{v[f]:.4f}' if 'MRR' in f or 'RDCR' in f else f'{100*v[f]:.2f}%' for f in fields)+' |')
 text+=['', '输出质量：']
 for m,r in summary.items():text.append(f"- {m}: 无效集合/重复配置 {r['invalid_component_or_duplicate_slots']} 个；集合有效但 LLM 标识无效 {r['invalid_llm_slots_with_valid_sets']} 个；缺失排名 {r['missing_slots']} 个；JSON 格式错误 {r['format_error_responses']} 条。")
text+=['','辅助协议只从单个字段提取唯一、完整、已在候选目录中的标识，不纠错、不猜测、不重试 API。主要协议保留字面匹配；LLM 字段无效不影响有效集合的单独评分。无效集合、重复配置及缺失排名均保留原排名位置并计为未命中。','', 'Ours 使用冻结 v10 生成器、beam1–2、完成集合与全部19个 LLM 的笛卡尔组合、冻结 critic v3 重排。本次将集合上限6放宽到10；critic序列化只改动数量断言，512 token预算保持不变。175个组件均进入候选目录；没有使用目标 query 的 gold 组合做检索输入。','', '本实验是从成功轨迹构建候选目录后的闭集跨域诊断。未观察到的有效配置仍可能被判为不匹配，通用执行框架动作也可能抬高集合重叠分数，因此不能把这些结果直接当作原始 SkillsBench 执行成功率。多个 query 可能来自同一任务，97不是97个独立任务。','', '文件：每种方法 predictions.jsonl 已转换为原数据集的 M/T 格式；per_sample 保存原始预测；ours/generation 保存搜索全过程；AUDIT.json 保存完整性审计。']
(O/'TABLE1.md').write_text('\n'.join(text)+'\n')
print('\n'.join(text[:12]))
