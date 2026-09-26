# Checkpoints

Place base models under `base/` or set their environment paths. Put the trained generator and critic weights in the corresponding named directories.

Use the matching tokenizer and resized identifier embeddings with the generator adapter. Both critic stages use the same compact-input serialization.

Remove `.gitkeep` before training scripts that require empty output directories. Stage-2 checkpoint selection is described in `docs/TRAINING.md`.
