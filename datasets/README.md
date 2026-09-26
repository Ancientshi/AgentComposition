# Data layout

Place experiment inputs in the following directories:

| Directory/input | Required contents when supplied privately |
|---|---|
| `PartII/{agents,questions,rankings}/merge.json` | Tool-label source, questions, and rankings. |
| `PartIII/{agents,questions,rankings}/merge.json` | LLM-label source, questions, and rankings. |
| `generative_v9_sft/sft_{train,valid}_PartII.jsonl` | Original retrieval-grounded records used by compact preparation. |
| `generative_v10_compact/` | Prepared SFT records, frozen test manifest, preparation report. |
| `critic_stage1/` | Candidate cases, teacher labels, encoded records, preparation/encoding metadata. |
| `critic_stage2/` | Reference-informed candidate cases and preparation metadata. |
| `SkillBench/` | Prepared observed candidates, canonical tasks/questions, positive agents, and task-disjoint splits. |
| `open_domain/` | Inputs for the separate open-domain variants. |

Generator source rows require query/qid, retrieval context or prompt, and target identifiers. Prepared rows contain token IDs, attention masks, completion-only labels, target/completion, and provenance. Critic cases contain query/qid/hash, candidate LLM/tool identifiers, and retrieval inventory; encoded cases include teacher scores and token IDs. Follow the corresponding preparation code for the exact schema.

Baseline caches and end-to-end inputs use `outputs/` or a configured workspace. Remove `.gitkeep` before running scripts that require empty output directories.
