"""Create the final report and per-trial viewer from verified observed results."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import collections
import csv
import html
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

STRATA=['top','middle','tail90','last']
LABELS=['第1名','50%位置','90%位置','最后一名']


def read(p):return json.loads(p.read_text())
def number(x):return '未定义' if x is None else f'{x:.3f}'
def interval(x):
    if x['mean'] is None:return '未定义'
    return number(x['mean'])+(' ['+', '.join(number(v) for v in x['ci95'])+']' if x['ci95'] else '')


def main():
    verify=read(ROOT/'final_verification.json');assert verify['status']=='verified'
    metrics=read(ROOT/'metrics.json');rows=read(ROOT/'per_query_results.json')
    components={c['token']:c for c in read(ROOT/'inventory.json')['components']}
    queries=read(ROOT/'queries.json');records={q['sample_id']:read(ROOT/'recommendations'/f'{q["sample_id"]}.json') for q in queries}
    ex=list(csv.DictReader((ROOT/'execution_scores.csv').open()))
    diagnostics=read(ROOT/'execution_diagnostics.json')
    sensitivity=read(ROOT/'sensitivity_excluding_harness_affected_queries.json')
    abort_count=sum(d['termination']=='harness_abort_invalid_tool_arguments' for d in diagnostics)
    verifier_issues=[d for d in diagnostics if d['verifier_exit_code'] not in [None,0,1]]
    verifier_note=('另有 '+str(len(verifier_issues))+' 次隐藏测试未取得完整结论（超时或运行异常）；其评测分数保留，但不记为测试通过。' if verifier_issues else '')
    pools={ds:[r['pool_size'] for r in rows if r['dataset']==ds] for ds in ['AgentSelect','SkillsBench']}
    overlap={}
    for ds in pools:
        rr=[records[q['sample_id']]['selected'] for q in queries if q['dataset']==ds]
        overlap[ds]={'same_backbone_all_four':sum(len({c['llm'] for c in r})==1 for r in rr),
                     'last_contains_all_top_components':sum(set(r[0]['tools'])<=set(r[-1]['tools']) for r in rr),
                     'n_queries':len(rr)}
    (ROOT/'configuration_overlap_audit.json').write_text(json.dumps(overlap,indent=2))
    rho=metrics['overall']['spearman'];ci=rho['ci95']
    if ci and ci[0]>0:conclusion='本次样本中，推荐名次与执行质量呈正向关系，整体 Spearman 的 bootstrap 区间在零以上。'
    elif ci and ci[1]<0:conclusion='本次样本中出现反向关系，整体 Spearman 的 bootstrap 区间在零以下。'
    else:conclusion='本次样本尚未显示稳定的正向排名关系，整体 Spearman 的 bootstrap 区间包含零。'
    lines=['# 扩展候选池端到端实验：Luna 模拟、Terra 评测','',
           '**已整理：30 条 query × 4 个配置 = 120 次执行记录及 120 份盲评。**','',conclusion,'',
           (f'其中 {abort_count} 次因无效工具参数触发执行器异常而中断，文件状态未能核验。主结果保留这些失败；文末另列排除受影响整条 query 的敏感性分析。' if abort_count else ''),'',verifier_note,'',
           '## 结果','',
           '正相关表示推荐更靠前的配置得到更高执行分数。下表是逐 query 指标的平均值；方括号为 10,000 次 query bootstrap 的 95% 区间。并列评分在成对一致率中计半分。','',
           '| 数据集 | Query数 | Spearman ρ | Kendall τ-b | 成对一致率 | 第1名−最后一名 |',
           '|---|---:|---:|---:|---:|---:|']
    for ds in ['AgentSelect','SkillsBench','overall']:
        m=metrics[ds]
        lines.append(f'| {ds} | {m["n_queries"]} | {interval(m["spearman"])} | {interval(m["kendall_tau_b"])} | {interval(m["pairwise_concordance"])} | {interval(m["top_bottom_delta"])} |')
    lines+=['','| 数据集 | 第1名 | 50%位置 | 90%位置 | 最后一名 | 全同分query数 |','|---|---:|---:|---:|---:|---:|']
    for ds in ['AgentSelect','SkillsBench','overall']:
        m=metrics[ds]
        lines.append('| '+ds+' | '+' | '.join(number(m[f'score_position_{i}']['mean']) for i in range(1,5))+' | '+str(m['spearman']['undefined_all_tied_queries'])+' |')
    lines+=['','评分范围为 0–10。全同分 query 的相关系数无定义，未从成对一致率和均分的分母删除；相关均值的有效数量和置零敏感性结果见 `metrics.json`。NDCG@4 仅涉及四个实际执行的配置，不能解释为整个候选池的 NDCG。','',
            '## 候选覆盖','',
            '同样的 30 条 query、固定 inventory 和推荐器检查点。保留 4 个 backbone 根节点，将 beam 从 1–2 扩为固定 4，每个父节点的工具分支数从 5 扩为 10。被剪枝的非空合法前缀补上 END，并由原生成器计算结束概率，再纳入完整配置池。AgentSelect 沿用原 V4 critic 的最终混合评分；SkillsBench 沿用平均生成 log-probability。','',
            '每条 query 在全部候选去重和排序后，固定选择第 1、ceil(0.5N)、ceil(0.9N)、N 名。N 为该 query 的池大小；每条都至少 100。这些是扩展搜索中的排名，不是原始窄 beam 的返回名次。所有 30 条推荐及选取位置均在任何一次新任务执行前冻结。','',
            '候选池只包含本次搜索实际生成的配置，未穷举 inventory 的所有可能组合。最后一名指该生成池的末位。','',
            '| 数据集 | 完整候选池范围 | 四个配置同backbone的query数 | 最后一名包含第1名全部组件的query数 |','|---|---:|---:|---:|']
    for ds,ns in pools.items():
        o=overlap[ds];lines.append(f'| {ds} | {min(ns)}–{max(ns)} | {o["same_backbone_all_four"]}/{o["n_queries"]} | {o["last_contains_all_top_components"]}/{o["n_queries"]} |')
    lines+=['','实际 backbone 分布：'+', '.join(k+' = '+str(v) for k,v in verify['backbone_counts'].items())+'。',
            '上述组件包含关系属于事后描述，不作为因果解释或筛选标准。','',
            '## 模拟器与评测员','',
            '- 模拟器：`gpt-5.6-luna`，推理强度 low；新建模拟缓存，没有复用第一轮 DeepSeek 生成的数据。',
            '- 评测员：`gpt-5.6-terra`，推理强度 low；完成度 0–4、正确性 0–4、约束遵守 0–2。原生 API 返回的模型名称逐条核验。',
            '- 评测输入仅含请求、答案、实际输出文件、终止条件、可观察工具证据和隐藏测试结果，不含推荐名次、推荐分数或 backbone 身份。工具来源与日期澄清从主评分开始统一给出。',
            '- 评测输入中的文件正文至多保留前 55,000 个字符，每次工具观测至多 7,000 个字符；完整文件和轨迹另行保存，原始隐藏测试检查实际文件。',
            '- 可部署工具实际运行。SkillsBench 使用原始文件、新容器和原始隐藏测试；不可直接访问的 API 使用明确标记的 Luna 模拟结果。公共功能替代服务的旧缓存如被复用，保留原采集时间。',
            '- SkillsBench 的所选技能决定加载哪些技能说明与资源；所有配置共享 shell、文件操作和已安装 Python 库。API 调用仅限该配置选中的端点。因此，缺少相关技能的配置仍可能凭 backbone 的通用编程能力完成任务。',
            '- 模拟器接收 endpoint、schema、调用参数及同服务历史，不单独接收 query 或推荐排名；调用参数可能含任务相关文本。',
            '- Backbone 请求设置为 temperature=0、enable_thinking=false；AgentSelect 与 SkillsBench 的单次输出上限分别为 4,000 与 6,000 tokens。',
            '- 与第一轮相同的 assistant turn、工具调用和 backbone 输出预算。墙钟保护时限从 900/1800 秒增加为 3600/5400 秒，以减轻服务延迟的影响。','',
            'SkillsBench 原始隐藏测试全部通过率：'+', '.join(LABELS[i]+': '+f'{metrics["SkillsBench"]["verifier_pass_by_stratum"][s]:.0%}' for i,s in enumerate(STRATA))+'。','',
            '## 审计','',
            '工具事件：'+', '.join(k+' = '+str(v) for k,v in verify['tool_event_modes'].items())+'。',
            '模型响应达到输出上限：'+str(verify['finish_reasons'].get('length',0))+' 次；墙钟终止：'+str(sum(d['termination']=='wall_time_budget' for d in diagnostics))+' 次；隐藏测试运行故障：'+str(len(verify['verifier_runtime_errors']))+' 次。这些记录没有因分数高低被剔除。',
            ('隐藏测试未得出完整结论的记录：'+', '.join(d['sample_id']+' 第'+str(d['rank'])+'名（退出码 '+str(d['verifier_exit_code'])+'）' for d in verifier_issues)+'。退出码124表示测试超时；原始评分、文件和部分测试输出均保留。' if verifier_issues else ''),
            '正式 END 评分使用原始完整前缀概率计算。预检中的共享提示词 KV 缓存优化未通过等价性检查，已弃用并在服务器归档。最终搜索预算在任务执行前根据计算开销设为 beam=4、工具分支10；未使用执行效果或 judge 分数调整候选。','',
            '## 与第一轮的关系及限制','',
            '第一轮只执行第 1、5、10、15 名，使用 DeepSeek-V3.2 模拟和 GPT-5.4 评测。这一轮同时改变搜索覆盖、模拟器、评测员和墙钟保护时限，不能把两轮评分差异全部归因于搜索扩展。第一轮所有结果原样保留。',
            'AgentSelect 样本沿用依据最新 Table 1 划分重建的 629-test 清单；正文另述的 200-query 清单未找到，无法确认这 25 条属于该子集。SkillsBench 样本属于已有的 test19 清单。',
            '延续第一轮的数据范围：AgentSelect 25 条没有生成器训练重叠，但曾用于生成器验证；5 个 SkillsBench 任务中 3 个有历史验证使用记录。固定 inventory 替代原逐 query 检索上下文，并根据所选任务需求整理。每个配置只完成一次正式执行，只有一名 LLM 评测员，API 模拟结果只在保存的模拟环境中解释。query bootstrap 不覆盖重复执行或不同模拟世界的不确定性。这是探索性实验，不能替代完全独立测试集上的验证。','',
            '## 文件','',
            '- `results.html`：30 条 query 的 120 个配置、实际名次、答案、评分依据、轨迹和文件。',
            '- `THREE_QUERY_EXAMPLES.md`：用户查看的三条示例，含原始英文 query 与四个实际推荐配置。',
            '- `INVENTORY.md`、`inventory.json`、`deployment_map.json`：固定的模型、66 个 API、16 个技能及部署方式。',
            '- `execution_scores.csv`、`per_query_results.json`、`metrics.json`：完整分数及统计。',
            '- `recommendations/`、`selected_configurations_manifest.json`：完整排序与执行前冻结摘要。',
            '- `trials/`、`simulator_api/`、`simulator_cache/`：实际响应、工具记录及模拟来源。',
            '- `PROTOCOL.md`、`execution_protocol.json`、`REPRODUCE.md`：协议、执行预算和复现方法。','']
    if abort_count:
        lines+=['## 执行器中断及敏感性分析','',
            f'{abort_count} 次执行在智能体调用 execute 时缺少必需 command 参数，触发执行器的未处理异常。容器被清理，输出文件在中断时是否存在无法核验，隐藏测试也未运行。原始异常、模型响应和已经完成的工具操作均保留；没有重跑 backbone 或重放操作。评测员使用相同提示词评估保存的失败轨迹，并被明确告知这些证据限制。主统计包含这些端到端中断结果，不能把全部失分归因于推荐质量。','',
            '在对这些中断记录评分前，记录了以下处理规则：另行排除所有含此类中断的整条 query，重新计算排名指标，以保留四个配置的配对结构。这是执行器缺陷触发的事后敏感性分析，未改写主结果或按分数选择样本。','',
            '| 数据集 | 剩余 query 数 | Spearman ρ | 成对一致率 |','|---|---:|---:|---:|']
        for ds in ['AgentSelect','SkillsBench','overall']:
            if ds in sensitivity['groups']:
                s=sensitivity['groups'][ds]
                lines.append(f'| {ds} | {s["n_queries"]} | {interval(s["spearman"])} | {interval(s["pairwise_concordance"])} |')
        lines+=['','受影响 query：'+', '.join(sensitivity['excluded_queries'])+'。','',
            '上述隐藏测试通过率以全部已分配配置为分母，只有实际测试全部通过才计入分子。未运行测试的数量分别为：'+', '.join(LABELS[i]+': '+str(metrics['SkillsBench']['verifier_unobserved_by_stratum'][s]) for i,s in enumerate(STRATA))+'。已启动但未得出完整结论的数量分别为：'+', '.join(LABELS[i]+': '+str(metrics['SkillsBench']['verifier_inconclusive_by_stratum'][s]) for i,s in enumerate(STRATA))+'。文件状态未知、测试超时与实际断言失败在逐次记录中分别标明。','']
    old=ROOT.parent/'experiments/end_to_end_initial'/'metrics.json'
    if old.exists():
        previous=read(old);(ROOT/'baseline_v1_metrics.json').write_text(json.dumps(previous,indent=2))
        lines+=['## 两轮描述性对照','',
                '下表采用第一轮原始主评分。评测环境不同，不能将差值当成受控消融的提升。','',
                '| 数据集 | 第一轮ρ | 本轮ρ | 第一轮成对一致率 | 本轮成对一致率 |','|---|---:|---:|---:|---:|']
        for ds in ['AgentSelect','SkillsBench','overall']:
            a,b=previous[ds],metrics[ds]
            lines.append('| '+ds+' | '+number(a['spearman']['mean'])+' | '+number(b['spearman']['mean'])+' | '+number(a['pairwise_concordance']['mean'])+' | '+number(b['pairwise_concordance']['mean'])+' |')
    (ROOT/'REPORT.md').write_text('\n'.join(lines))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    for ax,ds,color in zip(axes,['AgentSelect','SkillsBench'],['#286A9B','#C26838']):
        arr=np.array([r['scores'] for r in rows if r['dataset']==ds]);m=metrics[ds]
        for score in arr:ax.plot(range(4),score,color=color,alpha=.12,linewidth=.8)
        mean=arr.mean(axis=0);lo=np.array([m[f'score_position_{i}']['ci95'][0] for i in range(1,5)]);hi=np.array([m[f'score_position_{i}']['ci95'][1] for i in range(1,5)])
        ax.errorbar(range(4),mean,yerr=np.stack([mean-lo,hi-mean]),fmt='o-',color=color,capsize=4,linewidth=2)
        ax.set_xticks(range(4),['Rank 1','50%','90%','Last']);ax.set_ylim(-.2,10.5);ax.grid(axis='y',alpha=.18)
        ax.set_title(ds+f' (n = {len(arr)} queries)',fontweight='bold');ax.set_xlabel('Position in expanded recommendation pool')
    axes[0].set_ylabel('Terra blind judge score (0–10)')
    caption='Points: means; bars: 95% query-bootstrap intervals; pale lines: individual queries.'
    if abort_count:caption+='\nPrimary summary includes '+str(abort_count)+' harness-aborted trial(s); see sensitivity analysis in the report.'
    fig.text(.5,.02,caption,ha='center',fontsize=8.5)
    fig.tight_layout(rect=[0,.13 if abort_count else .07,1,1]);fig.savefig(ROOT/'score_by_position.png',dpi=240);fig.savefig(ROOT/'score_by_position.svg');plt.close(fig)
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>扩展候选池实验</title>',
           '<style>body{font:15px system-ui;max-width:1100px;margin:36px auto;color:#23303e;padding:0 22px}p{line-height:1.6}details{border:1px solid #d7e0e7;border-radius:9px;margin:12px 0;padding:14px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;word-break:break-word;background:#f5f7fa;padding:14px;max-height:500px;overflow:auto}a{color:#286A9B}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px;border-bottom:1px solid #d7e0e7;text-align:left;vertical-align:top}nav{margin:16px 0}</style>',
           '<h1>扩展候选池：30 queries × 4 configurations</h1><p>Luna 工具模拟 · Terra 盲评 · 第1名 / 50%位置 / 90%位置 / 最后一名</p>',
           '<p>'+html.escape(conclusion)+'</p>'+('<p><strong>'+str(abort_count)+' 次执行因无效参数触发执行器中断，文件状态未核验。主结果包含这些记录；排除受影响 query 的敏感性分析见 <a href="REPORT.md">完整报告</a>。</strong></p>' if abort_count else '')+'<img src="score_by_position.png" alt="实际得分与候选位置" style="width:100%">']
    parts.append('<nav><a href="REPORT.md">完整报告</a> · <a href="THREE_QUERY_EXAMPLES.md">三条推荐示例</a> · <a href="INVENTORY.md">模型与工具清单</a> · <a href="execution_scores.csv">全部分数</a></nav>')
    if verifier_note:parts.append('<p>'+html.escape(verifier_note)+'</p>')
    parts.append('<h2>排名一致性</h2><p>逐 query 平均；方括号为 query bootstrap 95% 区间。正相关表示推荐靠前的配置得分更高；同分在成对一致率中计半分。</p><table><tr><th>数据集</th><th>Query 数</th><th>Spearman ρ</th><th>Kendall τ-b</th><th>成对一致率</th><th>第1名−最后一名</th></tr>')
    for ds in ['AgentSelect','SkillsBench','overall']:
        m=metrics[ds]
        cells=[ds,str(m['n_queries'])]+[interval(m[k]) for k in ['spearman','kendall_tau_b','pairwise_concordance','top_bottom_delta']]
        parts.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells)+'</tr>')
    parts.append('</table>')
    if abort_count:
        parts.append('<h2>排除执行器受影响 query 的敏感性分析</h2><p>这是观察到执行器缺陷后增加的分析。按中断原因排除整条 query，主结果及全部失败记录仍保留；未按得分筛选。</p><table><tr><th>数据集</th><th>剩余 query 数</th><th>Spearman ρ</th><th>成对一致率</th></tr>')
        for ds in ['AgentSelect','SkillsBench','overall']:
            if ds in sensitivity['groups']:
                m=sensitivity['groups'][ds]
                cells=[ds,str(m['n_queries']),interval(m['spearman']),interval(m['pairwise_concordance'])]
                parts.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells)+'</tr>')
        parts.append('</table><p>排除的 query：'+html.escape(', '.join(sensitivity['excluded_queries']))+'</p>')
    parts.append('<h2>逐条 query 的四个配置与结果</h2>')
    for q in queries:
        sid=q['sample_id'];r=records[sid];row=next(x for x in rows if x['sample_id']==sid)
        preview=' '.join(q['query'].split())[:110]
        parts.append('<details><summary>'+html.escape(q['dataset']+' · '+q['task_id'])+' · N='+str(r['pool_size'])+' · 得分 '+html.escape(' / '.join(str(v) for v in row['scores']))+'<br>'+html.escape(preview)+'</summary><p>'+html.escape(q['query']).replace('\n','<br>')+'</p>')
        for label,c in zip(LABELS,r['selected']):
            rank=c['recommendation_rank'];path=Path('trials')/f'{sid}__r{rank:02d}'
            e,j=read(ROOT/path/'execution.json'),read(ROOT/path/'judgment.json')
            parts.append('<details><summary>'+label+f' · 实际第 {rank}/{r["pool_size"]} 名 · {j["total"]}/10 · '+html.escape(e['backbone'])+'</summary>')
            parts.append('<p>配置的工具与技能：</p><ul>'+''.join('<li>'+html.escape(('技能：' if components[t]['kind']=='skill' else 'API：')+components[t]['name'])+'</li>' for t in c['tools'])+'</ul><p>真实调用 '+str(e['native_tool_calls'])+' · 模拟调用 '+str(e['simulated_tool_calls'])+' · 终止条件 '+html.escape(e['termination'])+'</p>')
            parts.append('<p>Terra 评分依据：'+html.escape(j['rationale'])+'</p><pre>'+html.escape(e['final_answer'] or '未返回最终答案；请查看实际保存的文件和轨迹。')+'</pre>')
            if e.get('harness_incident'):
                parts.append('<p><strong>执行器中断；文件状态未核验：</strong>'+html.escape(e['harness_incident'])+'</p>')
            parts.append(f'<p><a href="{path}/execution.json">完整轨迹</a> · <a href="{path}/judgment.json">评分记录</a></p>')
            if e.get('verifier'):
                code=e['verifier']['exit_code']
                status={0:'全部通过',1:'未全部通过（存在测试失败）',124:'超时，未得出完整测试结论'}.get(code,'运行异常，退出码 '+str(code))
                parts.append('<p>原始隐藏测试：'+status+'</p>')
            for artifact in e['artifacts']:
                if artifact['exists']:parts.append(f'<p><a href="{path}/artifacts/{Path(artifact["requested_path"]).name}">'+html.escape(artifact['requested_path'])+'</a></p>')
            parts.append('</details>')
        parts.append('</details>')
    parts.append('</html>');(ROOT/'results.html').write_text('\n'.join(parts))
    (ROOT/'README.md').write_text('# Expanded-pool execution study\n\n30 queries, 120 new executions, Luna simulation, Terra blind judging.\n\nStart with [REPORT.md](REPORT.md) or open [results.html](results.html). The viewer requires its sibling directories. The original study remains separate and unchanged.\n')
    (ROOT/'REPRODUCE.md').write_text("# Recompute and inspect\n\nRun `python3 analyze_results.py .`, `python3 verify_final.py`, and `python3 build_report.py` from this directory. NumPy and Matplotlib are required; SciPy is used only by `python3 -m unittest test_metrics`. Final analysis rejects missing executions or scores.\n\nThe generator and critic checkpoints are the same as the original experiment. The expanded source, selection protocol, and execution budgets are included and frozen by separate SHA-256 manifests. Actual candidate ranks differ by query; four position labels are used for pooled score plots. Spearman uses the within-query order, which is equivalent for the actual ranks and their strictly monotone position labels.\n\n`selected_configurations_manifest.json` was written before the first of 120 trials started. `trials` contains all raw requests, responses, trajectories, actual files and blinded judgments. `simulator_api` and `simulator_cache` preserve Luna calls and labeled outputs. `generation` preserves search traces when included in the archive. The uncached generation-only preflight remains archived on the server.\n\nThe server workspace is `/root/yunxshi/NIPS2026/experiments/end_to_end`. Large weights, Linux dependencies and the pinned vulnerability database remain on the server. The runtime image and original assets are identified in the provenance records. Credentials are kept in the user's original proxy scripts and are not packaged.\n\nThis is not a controlled cross-round ablation: search, simulator, judge and wall-clock guard time differ from the first study. Sampling, prior validation exposure, fixed-context scope and simulator limits remain material. No completed low-scoring run is replaced.\n")


if __name__=='__main__':main()
