# Experiment guide

The table maps each experimental question to its implementation and entry point.

| Family | Entry or directory | Interpretation |
|---|---|---|
| Main fixed-test inference | `main`; `main_cartesian` | Compact-context generator, frozen retrieval context, final ranking, and separately identified Cartesian pool. |
| Component/bundle retrieval | `baselines/baseline1_*`, `baseline2_*` | Retrieve individual components or intact historical bundles. |
| Text-to-bundle | `baselines/baseline3_*` | Disjoint data preparation, training, and inference. |
| RAG | `baselines/baseline4_*`, `baseline5_*`, `baseline_rag_top10.py`, `baseline_siliconflow_top10.py` | Retrieved evidence with different generation backbones. |
| Diffusion baselines | `baselines/ddbc/`, `baselines/ddbc_semantic/` | Project-specific adaptations using separately configured upstream source and weights. |
| Structural ablations | `structural`; `experiments/structural_v3/` | Free generation, greedy decoding, beam, critic, beam+critic, and Cartesian settings. Keep V3 and V4 results separate. |
| Generator controls | `generator_controls` | Historical bundles × LLMs and RAG outputs scored with the same critic; exposure and pool-size diagnostics. |
| Sensitivity | `experiments/sensitivity/` | Beam width and critic-weight reranking analyses. |
| Frozen domain generalization | `domain_frozen` | Source-trained generator evaluated on observed SkillsBench candidates. |
| Adapted domain generalization | `domain_quickadapt_prepare`, `domain_quickadapt_train`, `domain_quickadapt_infer` | Task-disjoint capped adaptation and frozen/base controls. |
| Domain execution | `experiments/domain_generalization/execution/` | SkillsBench task containers and verifier-based execution. |
| End-to-end | `end_to_end`; `end_to_end_analyze` | Independently selected configurations executed with observable tool provenance and blind judging. |
| Case study | `case_study`; `run_matched_followup.py` | Four-arm post-hoc comparison, isolated caches, and raw-logit contrast. |

## Main experiment and ranking

`training/generator/run_infer_table1_compact.py` imports the search implementation in `inference/run_infer_v13_batch_eval_sota.py`. Its main search uses generator-only beam pruning with beam bounds 1–2. Completed candidates are ranked with a hybrid score:

```text
0.5 × z(critic_raw) + 0.15 × z(generator_average_logprob) − 0.1 × tool_count
```

Scores are standardized within the completed candidate pool. `main_cartesian` crosses saved tool sets with retrieved LLMs and uses critic-only controls for the expanded pool.

## Structural ablations

`experiments/structural/run_variants.py` contains V4 Free, Greedy, Beam+Critic, and BeamReplay entries. Additional Beam/Critic/completed-pool variants and Cartesian aggregations are implemented in the corresponding scripts.

Search-time critic variants begin critic scoring at two tools. Cartesian comparisons use a fixed scoring rule across the original and expanded candidate pools.

## Domain generalization

Frozen source transfer, target-domain SFT, capped quick adaptation, and execution have separate configurations. Supply the SkillsBench inventory and task splits as inputs. Frozen transfer records its tool-count serialization and critic version separately from the AgentSelect V4 experiment.

Quick adaptation trains LoRA parameters from the source checkpoint. Recorded settings include learning rate `1e-5`, effective batch 4, at most 40 updates, validation every 2 updates, and task-macro validation completion loss, including step zero. Job names and task splits come from the prepared manifest. Adaptation inference uses generator scores.

## End-to-end evaluation

`experiments/end_to_end/run_experiment.py` uses `gpt-5.6-luna` for fallback tool simulation and `gpt-5.6-terra` for judging. Agent backbones come from the separate execution inventory. It records tool-call mode, termination, artifacts, verifier evidence, requests, and judgments locally.

The rubric scores completion 0–4, correctness 0–4, and constraints 0–2. Judging is blinded to recommendation rank and identity. Execution provenance records native tools, public-service counterparts, and LLM simulation separately.

Inputs under the end-to-end workspace are `inventory.json`, `queries.json`, `recommendations/`, and the relevant benchmark assets/task containers. `E2E_ROOT` overrides the default `outputs/end_to_end` workspace. Earlier execution variants are in `experiments/end_to_end_initial/`.

## Case study

The fixed case uses tool sets S, S+D, S+N, and S+D+N with the same backbone and query. `run_matched_followup.py` prepares isolated starting caches for the matched comparison.

The plotted contrast uses **raw critic logits**:

```text
delta = u(S+D+N) − u(S+D) − u(S+N) + u(S)
```

Final ranking scores and execution grades use separate scales. The case records one execution for each of the four bundles.
