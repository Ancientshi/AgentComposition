"""Build a compact local review page for the uniform SkillsBench execution rerun."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "comparison.json").read_text())
rows = data["rows"]
assert len(rows) == data["summary"]["expected"] == data["summary"]["completed"] == 20
inventory = json.loads((ROOT / "inventory.json").read_text())
names = {item["token"]: item["name"] for item in inventory["components"]}


def esc(value):
    return html.escape(str(value), quote=True)


def trial_name(row):
    return f'{row["sample_id"]}__r{row["rank"]:02d}'


cards = []
for row in sorted(rows, key=lambda item: (item["task_id"],
                                          ("top", "middle", "tail90", "last").index(item["stratum"]))):
    name = trial_name(row)
    record_path = ROOT / "trials" / name / "execution.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else None
    passed = row["new_verifier_pass"]
    status = "通过" if passed else ("按用户要求截停" if record is None else "未通过")
    style = "ok" if passed else "fail"
    skills = ", ".join(names.get(token, token) for token in row["selected_skills"]) or "无"
    test_text = f'{row["new_tests_passed"]}/{row["new_tests_total"]}' if record else "未运行"
    artifact_text = f'{row["new_artifacts"]}/{len(record["artifacts"])}' if record else "—"
    verifier = (record or {}).get("verifier") or {}
    verifier_detail = ((verifier.get("stdout", "") + "\n" + verifier.get("stderr", ""))[:4200]
                       if record else "用户要求：超过 15 轮且未完成的低排名软件审计配置直接记失败，不再执行或运行验证器。")
    final = ((record or {}).get("final_answer") or "")[:1500]
    evidence_links = (f'<a href="trials/{esc(name)}/execution.json">完整轨迹与工具输出</a> · '
                      f'<a href="trials/{esc(name)}/verifier.json">原生验证结果</a>'
                      if record else '<a href="excluded.json">用户指定截停记录</a>')
    cards.append(f'''<tr>
      <td><b>{esc(row["task_id"])}</b><div class="mono sub">{esc(row["sample_id"])}</div></td>
      <td>{esc(row["stratum"])} <span class="sub">#{row["rank"]}</span></td>
      <td class="mono">{esc(row["backbone"])}</td>
      <td>{esc(skills)}</td>
      <td class="num">{esc(row["new_tool_calls"] if record else "—")}</td>
      <td class="num">{artifact_text}</td>
      <td class="num">{test_text}</td>
      <td class="num">{esc(row["new_quality_score"])}</td>
      <td class="num">{esc(row["new_grounded_score"])}</td>
      <td><span class="badge {style}">{status}</span></td>
    </tr><tr class="detail-row"><td colspan="10"><details>
      <summary>查看执行证据 · {esc(name)}</summary>
      <p>终止：{esc(row["new_termination"])}；纠正次数：{esc(row["new_recovery_attempts"])}；
      {evidence_links}</p>
      <div class="detail-grid"><div><h4>验证输出摘录</h4><pre>{esc(verifier_detail)}</pre></div>
      <div><h4>最终回答摘录</h4><pre>{esc(final)}</pre></div></div>
    </details></td></tr>''')

summary = data["summary"]
strata = "".join(f'<li>{esc(key)}：{value["passed"]}/{value["completed"]} 原生验证通过</li>'
                 for key, value in summary["by_stratum"].items())
page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SkillsBench 执行修复复跑</title>
<style>
body{{margin:0;background:#f7f9fc;color:#1d2a37;font:15px/1.55 system-ui,-apple-system,sans-serif}}
main{{max-width:1500px;margin:auto;padding:32px}}
h1{{font-size:28px;margin:0 0 8px}}h4{{margin:0 0 8px}}p{{margin:8px 0 16px}}
.lead{{color:#506276;max-width:1050px}}.metrics{{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}}
.metric{{background:white;border:1px solid #dbe3ed;border-radius:12px;padding:16px 22px;min-width:165px}}
.metric strong{{display:block;font-size:28px;color:#176591}}.metric span,.sub{{color:#64768a;font-size:12px}}
.panel{{background:white;border:1px solid #dbe3ed;border-radius:12px;padding:20px;margin:20px 0}}
table{{width:100%;border-collapse:collapse;background:white;font-size:13px}}
th,td{{text-align:left;border-bottom:1px solid #e7edf3;padding:10px;vertical-align:top}}
th{{background:#edf3f8;position:sticky;top:0;z-index:1}}.num{{text-align:right;white-space:nowrap}}
.mono{{font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px;overflow-wrap:anywhere}}
.badge{{border-radius:999px;padding:3px 9px;font-weight:700;white-space:nowrap}}
.ok{{background:#d9f3e7;color:#136441}}.fail{{background:#fbe4e3;color:#9a342d}}
.detail-row td{{padding:2px 12px 12px;background:#fbfcfe}}details summary{{cursor:pointer;color:#176591}}
.detail-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;max-height:300px;overflow:auto;background:#f1f4f8;padding:12px;border-radius:8px;font-size:11px}}
a{{color:#176591}}ul{{margin:8px 0 0;padding-left:20px}}@media(max-width:900px){{.detail-grid{{grid-template-columns:1fr}}table{{display:block;overflow-x:auto}}}}
</style></head><body><main>
<h1>SkillsBench：20 组配置执行复跑</h1>
<p class="lead">五条固定 query，每条取第 1、50% 位置、90% 位置和最后一名配置。推荐结果、模型端点、所选 skills 和原生验证器保持冻结；本页只比较执行层修复后的结果。盲评质量与工具证据分数均为 10 分制。</p>
<div class="metrics"><div class="metric"><strong>{summary["completed"]}/{summary["expected"]}</strong><span>配置已评估（含 1 组截停）</span></div>
<div class="metric"><strong>{summary["native_verifier_runs"]}/{summary["expected"]}</strong><span>实际执行及原生验证</span></div>
<div class="metric"><strong>{summary["new_native_pass"]}/{summary["completed"]}</strong><span>原生验证通过</span></div>
<div class="metric"><strong>{summary["new_artifact_present"]}/{summary["completed"]}</strong><span>至少交付一个文件</span></div>
<div class="metric"><strong>{summary["old_native_pass"]}/20</strong><span>原始运行通过</span></div></div>
<div class="panel"><b>按推荐位置</b><ul>{strata}</ul><p>文件存在不等于任务通过；每行保留原生测试、真实工具轨迹和两个独立评分。靠后配置可能使用无关 skill，某些成功也可能主要来自通用执行工具。</p>
<p><a href="../e2e_rerun_sf14_20260923/results.html">原始 30-query 页面</a> · <a href="comparison.csv">机器可读对照表</a> · <a href="RESULTS.md">结论与局限</a></p></div>
<table><thead><tr><th>任务</th><th>位置</th><th>实际模型端点</th><th>选中 skills</th><th class="num">调用</th><th class="num">文件</th><th class="num">测试</th><th class="num">质量</th><th class="num">证据</th><th>验证</th></tr></thead><tbody>{''.join(cards)}</tbody></table>
</main></body></html>'''
(ROOT / "results.html").write_text(page)
print(ROOT / "results.html")
