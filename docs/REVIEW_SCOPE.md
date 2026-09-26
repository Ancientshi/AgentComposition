# Evaluation notes

The repository contains source code, configurations, and input layouts. Supply datasets, trained weights, and experiment assets through the paths documented in `datasets/README.md` and `checkpoints/README.md`.

- Generator targets come from reference configurations. Execution quality is evaluated separately.
- Stage-1 teacher scores, stage-2 reference preferences, raw critic logits, final ranking scores, and execution grades use distinct definitions.
- The reported V4 evaluation uses the author-selected raw epoch-2 checkpoint; see the checkpoint selection note in `TRAINING.md`.
- Record the candidate pool, tool limit, and critic version alongside each experiment.
- For retrieval coverage, use the pre-injection audit in `evaluation/reference/exp2/`. Earlier cached contexts can contain injected reference components.
- Configuration-unseen, bundle-unseen, and retrieval-unavailable subsets have separate audit definitions.
- The case study compares four bundles for one fixed query and backbone, with one execution per bundle.
