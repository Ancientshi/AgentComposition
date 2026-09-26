# Code map

| Component | Main source | Role |
|---|---|---|
| Generator labels | `training/generator/prepare_sft.py` | Trace PartIII LLM labels and PartII tool labels, exclude fixed-test queries, rebuild completion targets. |
| Shared context | `training/generator/compact_context.py` | Retain query and candidate identifiers while compressing descriptions; shared by training and inference. |
| Generator SFT | `training/generator/train_sft.py` | Explicit completion masking, LoRA, checkpoint and training provenance. |
| Main inference wrapper | `training/generator/run_infer_table1_compact.py` | Frozen cached contexts and fixed-test inference with the compact prompt. |
| Search | `inference/run_infer_v13_batch_eval_sota.py` | Candidate-constrained tree search, generator scoring, completed-pool reranking. |
| Cartesian expansion | `inference/run_OURS_CARTESIAN_LLM10_CRITIC_RERANK.py` | Cross saved tool sets with retrieved LLM identifiers and rerank. |
| Critic architecture | `models/train_bundle_critic_improved.py` | EasyRec encoder and bundle scoring head. |
| Critic representation | `training/critic/stage1/compact_input.py` | Candidate-preserving compact representation, shared by both critic stages. |
| Critic stage 1 | `training/critic/stage1/{prepare_cases,label_terra,encode_dataset,train}.py` | Blinded teacher supervision and initial critic fitting. |
| Critic stage 2 | `training/critic/stage2/{prepare,core,train}.py` | Reference-informed redundancy, coverage, replacement, and teacher-LLM pairs. |
| Critic HTTP service | `services/critic/serve.py` | Foreground local scoring service. |
| Retrieval | `retrieval/` | Collaborative and semantic component retrieval implementations. |
| Metrics | `evaluation/reference/exp3/` | Existing evaluation scripts, including ranked component precision/recall. |
| Structural variants | `experiments/structural/` | V4 variants and results aggregation; V3 variants kept separately. |
| Generator controls | `experiments/generator_controls/` | Historical bundles and RAG configurations scored using the same critic. |
| Domain transfer | `experiments/domain_generalization/` | Frozen transfer, SFT, capped quick adaptation, and execution variants. |
| End-to-end execution | `experiments/end_to_end/` | Tool execution, fallback simulation, blind judging, and aggregate metrics. |
| Case study | `experiments/case_study/` | Four-arm bundle comparison and figure extraction. |

The numbered inference files support different search variants. Use the named configurations to locate each experiment.

`source_manifest.json` records source hashes and destination paths for the organized code.
