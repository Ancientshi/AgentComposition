#!/usr/bin/env python3
"""Build PartI_pairs.jsonl from the merged PartI dataset.

For every question, the first ``--top-n`` agents in its ranking are treated as
positive examples and expanded into separate JSONL rows.
"""

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(str(AC_ROOT))
PART_ROOT = PROJECT_ROOT / 'datasets/PartI'
DEFAULT_OUTPUT = PROJECT_ROOT / 'data_preparation/rag_synthesis/scripts/PartI_pairs.jsonl'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=Path, default=PART_ROOT / "agents/merge.json")
    parser.add_argument("--questions", type=Path, default=PART_ROOT / "questions/merge.json")
    parser.add_argument("--rankings", type=Path, default=PART_ROOT / "rankings/merge.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of leading ranked agents treated as positives (default: 10).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N questions; 0 processes all questions.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def llm_token(agent: dict[str, Any]) -> str:
    model = agent.get("M", {})
    if isinstance(model, dict):
        name = model.get("name")
    elif isinstance(model, str):
        name = model
    else:
        name = None

    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Agent has no valid M.name: {agent!r}")

    name = name.strip()
    if name.startswith("<LLM_") and name.endswith(">"):
        return name
    return f"<LLM_{name.replace('/', '__')}>"


def serialize_target(agent: dict[str, Any]) -> str:
    # PartI contains LLM-only agents. The explicit empty-tool token is required
    # by the structured-agent parser for a valid zero-tool configuration.
    return f"{llm_token(agent)} <TOOL_SEP> <TOOL_EMPTY> <SPECIAL_END>"


def main() -> None:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be greater than zero.")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative.")

    agents = load_json(args.agents)
    questions = load_json(args.questions)
    rankings_doc = load_json(args.rankings)

    if not isinstance(agents, dict):
        raise TypeError("Agents JSON must contain an object.")
    if not isinstance(questions, dict):
        raise TypeError("Questions JSON must contain an object.")
    if not isinstance(rankings_doc, dict):
        raise TypeError("Rankings JSON must contain an object.")

    rankings = rankings_doc.get("rankings", rankings_doc)
    if not isinstance(rankings, dict):
        raise TypeError("The rankings field must be a question-to-agent-list object.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    questions_written = 0
    missing_questions: list[str] = []
    missing_agents: list[str] = []

    with args.output.open("w", encoding="utf-8") as output:
        for qid, ranked_agent_ids in rankings.items():
            if args.limit and questions_written >= args.limit:
                break

            question = questions.get(qid)
            if question is None:
                missing_questions.append(qid)
                continue
            if not isinstance(question, dict):
                raise TypeError(f"Question {qid!r} must be an object.")
            if not isinstance(ranked_agent_ids, list):
                raise TypeError(f"Ranking for {qid!r} must be a list.")

            query = question.get("input", "")
            if not isinstance(query, str):
                query = str(query)

            # Only the first ten (or --top-n) ranked agents are positives.
            for rank, agent_id in enumerate(ranked_agent_ids[: args.top_n], start=1):
                agent = agents.get(agent_id)
                if agent is None:
                    missing_agents.append(str(agent_id))
                    continue
                if not isinstance(agent, dict):
                    raise TypeError(f"Agent {agent_id!r} must be an object.")

                row = {
                    "qid": qid,
                    "agent_id": agent_id,
                    "part": "PartI",
                    "topk": rank,
                    "query": query,
                    "target": serialize_target(agent),
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1

            questions_written += 1

    print(
        f"Wrote {rows_written:,} positive rows from "
        f"{questions_written:,} questions to {args.output}"
    )
    if missing_questions:
        print(
            f"WARNING: {len(missing_questions):,} missing questions; "
            f"first 5: {missing_questions[:5]}"
        )
    if missing_agents:
        print(
            f"WARNING: {len(missing_agents):,} missing agents; "
            f"first 5: {missing_agents[:5]}"
        )


if __name__ == "__main__":
    main()
