"""Compare repaired SkillsBench executions with the frozen earlier run."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "e2e_rerun_sf14_20260923"
queries = {q["sample_id"]: q for q in json.loads((ROOT / "queries.json").read_text())
           if q["dataset"] == "SkillsBench"}
excluded = {(item["sample_id"], item["rank"]): item for item in
            json.loads((ROOT / "excluded.json").read_text())}


def read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


rows = []
for sid, query in queries.items():
    recommendation = read(ROOT / "recommendations" / f"{sid}.json")
    for configuration in recommendation["selected"]:
        rank = configuration["recommendation_rank"]
        name = f"{sid}__r{rank:02d}"
        old = read(OLD / "trials" / name / "execution.json")
        new = read(ROOT / "trials" / name / "execution.json")
        excluded_item = excluded.get((sid, rank))
        assert not (new and excluded_item)
        assert old is not None
        if new is not None:
            assert new["configuration"] == old["configuration"] == configuration
        new_ctrf = read(ROOT / "trials" / name / "verifier_ctrf.json")
        old_ctrf = read(OLD / "trials" / name / "verifier_ctrf.json")
        row = {
            "sample_id": sid,
            "task_id": query["task_id"],
            "stratum": configuration["selection_stratum"],
            "rank": rank,
            "backbone": old["backbone"],
            "selected_skills": [t for t in configuration["tools"] if t.startswith("<<SB")],
            "old_tool_calls": old["tool_calls"],
            "new_tool_calls": new["tool_calls"] if new else None,
            "old_artifacts": sum(a["exists"] for a in old["artifacts"]),
            "new_artifacts": sum(a["exists"] for a in new["artifacts"]) if new else None,
            "old_verifier_pass": old["verifier"]["exit_code"] == 0,
            "new_verifier_pass": new["verifier"]["exit_code"] == 0 if new else (False if excluded_item else None),
            "new_verifier_run": bool(new),
            "new_execution_status": "completed" if new else ("user_cutoff_failure" if excluded_item else "missing"),
            "old_tests_passed": old_ctrf["results"]["summary"]["passed"] if old_ctrf else None,
            "new_tests_passed": new_ctrf["results"]["summary"]["passed"] if new_ctrf else None,
            "new_tests_total": new_ctrf["results"]["summary"]["tests"] if new_ctrf else None,
            "new_termination": new["termination"] if new else ("user_cutoff" if excluded_item else None),
            "new_recovery_attempts": new.get("recovery_attempts") if new else None,
        }
        quality = read(ROOT / "trials" / name / "judgment.json")
        grounded = read(ROOT / "trials" / name / "judgment_grounded.json")
        row["new_quality_score"] = quality["total"] if quality else None
        row["new_grounded_score"] = grounded["total"] if grounded else None
        rows.append(row)

completed = [r for r in rows if r["new_verifier_pass"] is not None]
summary = {
    "expected": len(rows),
    "completed": len(completed),
    "native_verifier_runs": sum(r["new_verifier_run"] for r in completed),
    "user_cutoff_failures": sum(r["new_execution_status"] == "user_cutoff_failure" for r in completed),
    "old_native_pass": sum(r["old_verifier_pass"] for r in rows),
    "new_native_pass": sum(r["new_verifier_pass"] for r in completed),
    "new_artifact_present": sum(bool(r["new_artifacts"]) for r in completed),
    "old_artifact_present": sum(bool(r["old_artifacts"]) for r in rows),
    "by_stratum": {s: {
        "completed": sum(r["stratum"] == s for r in completed),
        "passed": sum(r["stratum"] == s and r["new_verifier_pass"] for r in completed),
    } for s in ("top", "middle", "tail90", "last")},
    "termination": dict(Counter(r["new_termination"] for r in completed)),
    "fixed_recommendations_unchanged": True,
    "interpretation": "Execution-only intervention; do not merge with original 30-query scores.",
}
(ROOT / "comparison.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n")
fields = [k for k in rows[0] if k != "selected_skills"]
with (ROOT / "comparison.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k: row[k] for k in fields} for row in rows)
print(json.dumps(summary, ensure_ascii=False))
