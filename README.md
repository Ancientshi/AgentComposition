# AgentComposition

## 🌐 [Project Page](https://atlas-k7m4q9.vercel.app/)

This repository contains the implementation of AgentComposition: retrieval-grounded generation of model, tool, and skill bundles for a target task. It includes generator supervised fine-tuning (SFT), two-stage bundle critic training, constrained search, retrieval methods, baselines, and evaluation scripts.

## Start here

- [Method and source map](docs/CODE_MAP.md): where the generator, search, critic, retrieval, and evaluation are implemented.
- [Training](docs/TRAINING.md): generator SFT and critic stage 1 → stage 2.
- [Experiments](docs/EXPERIMENTS.md): main experiments, structural ablations, domain generalization, end-to-end evaluation, and case study.
- [Data layout](datasets/README.md): empty dataset directories and required private inputs.
- [Environment](docs/ENVIRONMENT.md): historical runtime and configurable paths/services.
- [Evaluation notes](docs/REVIEW_SCOPE.md): experiment versions, scoring quantities, and data provenance.

## Repository layout

```text
agentcomposition/      shared paths and command catalogue
configs/               explicit review/run entries
training/generator/    label preparation, compact context, and generator SFT
training/critic/       stage1 teacher supervision; stage2 reference refinement
models/                bundle scoring architecture and EasyRec integration
inference/             constrained tree search and Cartesian reranking
retrieval/             component retrieval and collaborative-filtering models
baselines/             retrieval, text-to-bundle, RAG, and diffusion baselines
evaluation/            original ranked-recall evaluators
experiments/           ablations, generalization, execution, and case study
services/critic/       critic serving interface
data_preparation/      retrieval-grounded supervision construction
datasets/              input layout
checkpoints/           model and adapter paths
outputs/               runtime outputs
docs/                  method and experiment guides
```

## Inspect an experiment

From the source checkout, use Python 3.11 or later:

```bash
python -m agentcomposition --list
python -m agentcomposition generator_sft
python -m agentcomposition critic_stage1_train
python -m agentcomposition critic_stage2_train
python -m agentcomposition main
python -m agentcomposition structural
python -m agentcomposition domain_frozen
python -m agentcomposition end_to_end
python -m agentcomposition case_study
```

By default, each command prints its entry point and required inputs. Supply the corresponding data, models, and services, then add `--execute` to run it.

```bash
python -m pip install -r requirements-experiments.txt
python -m pip install -e .
# Set paths/services using .env.example as a reference.
# After configuring the experiment inputs:
python -m agentcomposition main --execute
```
