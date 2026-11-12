# Portable reviewer handoff validation

Validated on UbuntuPC RTX5080 using the existing moe-adapters environment, 2026-11-11. These are diagnostic checks only; no formal accuracy claim or completed reviewer study is made.

- All four graph/non-graph x fixed/adaptive native GPU preflights completed one batch update and checkpoint read-back. Fixed selected 2; adaptive selected 5 at C_t=28.
- Budget matrix checked C_t=2,5,8,10,28,46,100, clipping and rejection of missing adaptive-stage metadata.
- Seed0 initial adapter tensor hashes match across graph and non-graph arms.
- Graph preflight has finite gradients and nonzero gradients in 312 graph,48 router,276 adapter parameter tensors. Native first warmup update has zero learning rate; this verifies gradients, not training convergence.
- Strict checkpoint-only loading/evaluation completed with zero native training updates. Repeated gates, adjacency, image IDs and labels match exactly across all 12 vision layers after switching test preprocessing from legacy random crops to deterministic resize/center crop.
- Manifest initialization and trusted-checkpoint inventory commands passed.
- Analysis correctly excludes all smoke runs and reports zero scientific completed runs.
- Python compilation checks pass. Full ten-stage training and original-checkpoint formats are not yet validated; unknown formats require reviewed conversion, not guessed partial loading.

Training semantics retained: native CE and auxiliary-loss formulas, optimizer/scheduler and epochs*batches+1 updates. Opt-in controls provide equal fixed/adaptive rules and configured router regularization in the two arms; graph initialization RNG isolation is enabled only by the new driver. Evaluation explicitly uses the latest stage router and all-seen prompts.

Limitations: resource monitor samples GPU and main-process CPU/RSS; worker-tree peak RAM is not measured. The generated sample pairs are selected illustrations, not population proof. Manuscript U1-U8 mapping, submitted neighbor default and matched even/uneven evidence need original artifacts and human verification. No optional architecture sweep is launched.

The plotting integration was exercised in `/tmp/moce_portable_plot_validation` with an explicitly named diagnostic fixture derived from smoke data. It produced all three figure types. This fixture is excluded from the scientific study and is not a trained result. Stable image IDs were checked against original CIFAR100 fine labels. Final strict reload repeated all 12 layer records exactly.
