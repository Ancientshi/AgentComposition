"""Extract the paper case from the saved end-to-end experiment records."""

import json
from pathlib import Path


from agentcomposition.paths import ROOT, env
SOURCE_ROOT = Path(env('E2E_ROOT', str(ROOT/'outputs/end_to_end')))
HERE = ROOT / 'outputs/case_study'
HERE.mkdir(parents=True, exist_ok=True)
FOLLOWUP_ROOT = ROOT / 'outputs/case_study/followup'

S = "<<Whois Lookup_v3&&Check Similarity>>"
D = "<<Whois Lookup_v3&&DNS Lookup>>"
N = "<<Whois Lookup_v3&&NS Lookup>>"
MODEL = "<LLM_deepseek_deepseek-v4-pro>"
OTHER_MODELS = (
    "<LLM_Qwen_Qwen3.8-27B>",
    "<LLM_kimi-k2.6>",
)


def main() -> None:
    rec_path = SOURCE_ROOT / "recommendations" / "question_5656.json"
    judgment_path = SOURCE_ROOT / "trials" / "question_5656__r01" / "judgment.json"
    execution_path = SOURCE_ROOT / "trials" / "question_5656__r01" / "execution.json"

    record = json.loads(rec_path.read_text())
    judgment = json.loads(judgment_path.read_text())
    execution = json.loads(execution_path.read_text())
    gold = json.loads((HERE / "gold_label.json").read_text())
    assert record["sample_id"] == execution["sample_id"] == "question_5656"
    assert gold["qid"] == "question_5656" and set(gold["gold_tools"]) == {S, D, N}

    scored = {
        frozenset(row["tools"]): row
        for row in record["ranked_pool"]
        if row["llm"] == MODEL and row["critic_raw"] is not None
    }
    bundle_keys = {
        "S": frozenset((S,)),
        "SD": frozenset((S, D)),
        "SN": frozenset((S, N)),
        "SDN": frozenset((S, D, N)),
    }
    cells = {
        key: {
            "tools": sorted(bundle),
            "critic_raw": scored[bundle]["critic_raw"],
            "final_rerank": scored[bundle]["search_score"],
            "rank": scored[bundle]["stage_rank"],
        }
        for key, bundle in bundle_keys.items()
    }
    followup_manifest = FOLLOWUP_ROOT / "manifest.json"
    manifest = json.loads(followup_manifest.read_text())
    assert manifest["purpose"].startswith("Post-hoc q5656")
    followup_trials = {}
    for key, bundle in bundle_keys.items():
        trial_dir = FOLLOWUP_ROOT / key / "trials" / f"question_5656__r{cells[key]['rank']:02d}"
        followup_execution = json.loads((trial_dir / "execution.json").read_text())
        followup_judgment = json.loads((trial_dir / "judgment.json").read_text())
        assert followup_execution["sample_id"] == "question_5656"
        assert followup_execution["recommendation_rank"] == cells[key]["rank"]
        assert followup_execution["backbone"] == "deepseek-ai/DeepSeek-V4-Pro"
        assert set(followup_execution["configuration"]["tools"]) == set(bundle)
        assert followup_execution["query"] == execution["query"]
        assert followup_judgment["judge_model"] == "gpt-5.6-terra"
        assert followup_judgment["total"] == sum(
            followup_judgment[name] for name in ("completion", "correctness", "constraints")
        )
        assert followup_execution["termination"] == "final_answer"
        cells[key]["e2e_total"] = followup_judgment["total"]
        cells[key]["e2e_subscores"] = {
            name: followup_judgment[name]
            for name in ("completion", "correctness", "constraints")
        }
        cells[key]["e2e_tool_calls"] = followup_execution["tool_calls"]
        cells[key]["e2e_native_calls"] = followup_execution["native_tool_calls"]
        cells[key]["e2e_simulated_calls"] = followup_execution["simulated_tool_calls"]
        cells[key]["e2e_used_tool_functions"] = [
            event["tool_name"] for event in followup_execution["events"]
        ]
        followup_trials[key] = str(trial_dir)
    scores = {key: cell["critic_raw"] for key, cell in cells.items()}
    delta = scores["SDN"] - scores["SD"] - scores["SN"] + scores["S"]
    additive_prediction = scores["SD"] + scores["SN"] - scores["S"]
    assert abs(delta - (scores["SDN"] - additive_prediction)) < 1e-9
    assert cells["SN"]["rank"] == 1
    assert execution["recommendation_rank"] == 1

    other_model_interactions = {}
    for model in OTHER_MODELS:
        model_scores = {
            frozenset(row["tools"]): row["critic_raw"]
            for row in record["ranked_pool"]
            if row["llm"] == model and row["critic_raw"] is not None
        }
        if all(bundle in model_scores for bundle in bundle_keys.values()):
            other_model_interactions[model] = (
                model_scores[bundle_keys["SDN"]]
                - model_scores[bundle_keys["SD"]]
                - model_scores[bundle_keys["SN"]]
                + model_scores[bundle_keys["S"]]
            )

    payload = {
        "sample_id": "question_5656",
        "dataset": record["dataset"],
        "query": execution["query"],
        "backbone": MODEL,
        "gold_label": gold,
        "context_sha256": record["context_sha256"],
        "sources": {
            "recommendations": str(rec_path),
            "judgment": str(judgment_path),
            "execution": str(execution_path),
        },
        "tool_legend": {
            "S": S,
            "D": D,
            "N": N,
        },
        "cells": cells,
        "interaction_raw": delta,
        "other_backbone_interactions_raw": other_model_interactions,
        "additive_prediction_raw": additive_prediction,
        "executed_top_judge_score": judgment["total"],
        "executed_top_native_calls": execution["native_tool_calls"],
        "executed_top_simulated_calls": execution["simulated_tool_calls"],
        "evidence_limit": judgment["evidence_limits"],
        "followup": {
            "label": "Post-hoc matched execution; excluded from original 30x4 aggregate",
            "manifest": str(followup_manifest),
            "trial_directories": followup_trials,
            "historical_SN_judge_total": judgment["total"],
            "descriptive_e2e_second_difference": (
                cells["SDN"]["e2e_total"]
                - cells["SD"]["e2e_total"]
                - cells["SN"]["e2e_total"]
                + cells["S"]["e2e_total"]
            ),
            "warning": "Single runs and a bounded LLM-judge rubric; this value is not a causal interaction estimate.",
        },
    }
    out = HERE / "case_data.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {out}")
    print(f"Interaction: {delta:+.6f}; additive prediction: {additive_prediction:+.6f}")


if __name__ == "__main__":
    main()
