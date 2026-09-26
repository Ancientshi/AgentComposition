#!/usr/bin/env python3
"""Build PartII_pairs.json (JSONL) from PartII merge.json files.

Each ranked question-agent association becomes one output row.  The ranking
position is written to ``topk`` using 1-based indexing.
"""

from __future__ import annotations
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOT = Path(str(AC_ROOT / 'datasets/PartII'))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=Path, default=DEFAULT_ROOT / "agents/merge.json")
    parser.add_argument("--questions", type=Path, default=DEFAULT_ROOT / "questions/merge.json")
    parser.add_argument("--rankings", type=Path, default=DEFAULT_ROOT / "rankings/merge.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT.parent / "PartII_pairs.jsonl",
        help="Output is newline-delimited JSON despite the .jsonl suffix.",
    )
    parser.add_argument(
        "--default-llm",
        default=None,
        help=(
            "Optional LLM used when agent['M'] is empty, e.g. "
            "mistralai/Mixtral-8x7B-v0.1. Without it, a tool-only target is emitted."
        ),
    )
    parser.add_argument(
        "--keep-part-prefix",
        action="store_true",
        help="Keep qid as PartII_question_N instead of normalizing it to question_N.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N ranked questions (0 means all; useful for testing).",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_qid(qid: str, keep_part_prefix: bool) -> str:
    if keep_part_prefix:
        return qid
    return re.sub(r"^PartII_(question_\d+)$", r"\1", qid)


def first_string(value: Any) -> str | None:
    """Extract the first non-empty model string from common M encodings."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        for item in value:
            found = first_string(item)
            if found:
                return found
        return None
    if isinstance(value, dict):
        # Prefer values of conventional fields before falling back to values/keys.
        for key in ("model", "models", "llm", "LLM", "name", "id"):
            if key in value:
                found = first_string(value[key])
                if found:
                    return found
        for item in value.values():
            found = first_string(item)
            if found:
                return found
        if len(value) == 1:
            only_key = next(iter(value))
            if isinstance(only_key, str) and only_key.strip():
                return only_key.strip()
    return None


def model_token(model: str) -> str:
    model = model.strip()
    if model.startswith("<LLM_") and model.endswith(">"):
        return model
    # Match the naming convention in the supplied example.
    encoded = model.replace("/", "__")
    return f"<LLM_{encoded}>"


def tool_token(tool: str) -> str:
    """Keep existing angle-bracket markup; otherwise use one bracket pair."""
    tool = tool.strip()
    if tool.startswith("<") and tool.endswith(">"):
        return tool
    return f"<{tool}>"


def extract_tools(agent: dict[str, Any]) -> list[str]:
    tool_section = agent.get("T", {})
    raw_tools = tool_section.get("tools", []) if isinstance(tool_section, dict) else tool_section
    if raw_tools is None:
        return []
    if isinstance(raw_tools, str):
        raw_tools = [raw_tools]
    if not isinstance(raw_tools, list):
        raise TypeError(f"Unsupported tools value: {raw_tools!r}")
    tools = [str(tool).strip() for tool in raw_tools if str(tool).strip()]
    # Preserve source order while preventing accidental duplicate components.
    return list(dict.fromkeys(tools))


def serialize_target(agent: dict[str, Any], default_llm: str | None) -> str:
    llm = first_string(agent.get("M")) or default_llm
    tools = extract_tools(agent)

    pieces: list[str] = []
    if llm:
        pieces.extend((model_token(llm), "<TOOL_SEP>"))
    pieces.extend(tool_token(tool) for tool in tools)
    pieces.append("<SPECIAL_END>")
    return " ".join(pieces)


def ranking_items(rankings_doc: Any) -> Iterable[tuple[str, Any]]:
    if not isinstance(rankings_doc, dict):
        raise TypeError("Rankings JSON must be an object.")
    rankings = rankings_doc.get("rankings", rankings_doc)
    if not isinstance(rankings, dict):
        raise TypeError("rankings must be a question-to-agent-list object.")
    return rankings.items()


def main() -> None:
    args = parse_args()
    agents = load_json(args.agents)
    questions = load_json(args.questions)
    rankings_doc = load_json(args.rankings)
    if not isinstance(agents, dict) or not isinstance(questions, dict):
        raise TypeError("Agents and questions JSON files must contain objects.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    questions_written = 0
    missing_questions: list[str] = []
    missing_agents: list[str] = []

    with args.output.open("w", encoding="utf-8") as output:
        for source_qid, ranked_agents in ranking_items(rankings_doc):
            if args.limit and questions_written >= args.limit:
                break
            question = questions.get(source_qid)
            if question is None:
                missing_questions.append(source_qid)
                continue
            if not isinstance(ranked_agents, list):
                raise TypeError(f"Ranking for {source_qid} is not a list.")

            query = question.get("input", "") if isinstance(question, dict) else str(question)
            qid = normalize_qid(source_qid, args.keep_part_prefix)
            for rank, agent_id in enumerate(ranked_agents, start=1):
                agent = agents.get(agent_id)
                if agent is None:
                    missing_agents.append(str(agent_id))
                    continue
                row = {
                    "qid": qid,
                    "agent_id": agent_id,
                    "part": "PartII",
                    "topk": rank,
                    "query": query,
                    "target": serialize_target(agent, args.default_llm),
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1
            questions_written += 1

    print(f"Wrote {rows_written:,} rows from {questions_written:,} questions to {args.output}")
    if missing_questions:
        print(f"WARNING: {len(missing_questions):,} missing question IDs; first 5: {missing_questions[:5]}")
    if missing_agents:
        print(f"WARNING: {len(missing_agents):,} missing agent IDs; first 5: {missing_agents[:5]}")


if __name__ == "__main__":
    main()
