# Training

This guide summarizes the training entries and recorded experiment settings.

## Generator SFT

The compact-context generator uses Meta-Llama-3-8B with LoRA rank 16, alpha 32, dropout 0.05, and the attention/MLP projections. `embed_tokens` and `lm_head` are retained as trainable modules. The recorded training uses one epoch, learning rate `1e-4`, per-device batch size 1, accumulation 8, seed 42, a prompt budget of 6,144 tokens, and a sequence budget of 8,192 tokens. The launch entry uses four distributed processes; change it explicitly for other hardware.

```bash
python -m agentcomposition generator_prepare
python -m agentcomposition generator_sft
```

Preparation keeps exact candidate identifiers and the full query, excludes query/qid overlap with the fixed test manifest, groups train/dev by query, and supervises the target completion. PartII provides tool-set labels; the corresponding LLM labels are traced to PartIII.

Required private inputs are listed in `datasets/README.md`. The fixed 100-query manifest has historical SHA-256:

```text
4dc31d07fea9854d7e826f943f187d960e4f6a02dda2f4f194d5e7c2dd49bd5a
```

Use the fixed manifest for the corresponding experiment.

## Critic stage 1: teacher-supervised fitting (historical V3)

1. `prepare_cases.py` builds candidate pools from train/dev queries and retrieval inventories. Training references can seed tool sets; reference labels are not included in the teacher prompt.
2. `label_terra.py` asks `gpt-5.6-terra` to score randomized candidate identifiers for configuration suitability.
3. `encode_dataset.py` creates the shared compact representation and checks that distinct configurations do not collapse to identical token sequences.
4. `train.py` fits the EasyRec encoder and scoring head using pairwise supervision.

Recorded settings: max input length 512; head hidden size 512; dropout 0.1; normalized embeddings; four epochs; encoder LR `5e-6`; head LR `1e-4`; batch 16; accumulation 2; seed 42. Candidate pairs distinguish LLM-only, tool-only, and joint changes.

```bash
python -m agentcomposition critic_stage1_prepare
python -m agentcomposition critic_stage1_label
python -m agentcomposition critic_stage1_encode
python -m agentcomposition critic_stage1_train
```

Teacher labeling contacts a separately configured API and may incur usage charges only when explicitly executed.

## Critic stage 2: reference-informed refinement (historical V4)

Stage 2 initializes from the stage-1 checkpoint. It constructs within-reference redundancy, missing-required-tool, and replacement pairs, plus teacher-based LLM preferences.

Recorded stage-2 settings: batch 24; encoder LR `2e-6`; head LR `2e-5`; max input length 512; seed 42. The V4 evaluation uses the **author-selected raw epoch-2 checkpoint, alpha 1**. The training entry runs two epochs.

```bash
python -m agentcomposition critic_stage2_prepare
python -m agentcomposition critic_stage2_train
python -m agentcomposition critic_serve
```

**Checkpoint selection.** The V4 evaluation used the manually selected raw epoch-2 checkpoint. It is distinct from the checkpoint selected by the trainer's validation guard, so use `last_critic.pt` and the epoch metadata when restoring the reported ranking configuration.

## Weights

Place base models, adapters, and critic checkpoints under `checkpoints/`, or set `BASE_MODEL` and `EASYREC_MODEL`. Rebase any machine-specific paths in checkpoint metadata before serving.
