# MoCE rebuttal: code and analysis handoff

Updated 2026-11-11. Scope: code and analysis; no manuscript writing.
Run training on the other PC. **Do not launch computation on UbuntuPC.**

## To-do list

- [ ] Verify the completed adaptive-k four-way comparison: collect all four configurations, seeds, epochs/update counts, and results. Completion was reported by the researcher; the four-way evidence has not been verified in this checkout.
- [x] Add a reproducible graph-configuration audit (`cil/moce_analysis.py`).
- [ ] Resolve paper `m` versus `k_g` against the exact submitted-run config. The code parameter is `model.graph_adj_top_m`. There is no single default used by every experiment: the constructor defaults to dense; older N=8 configs set 8; the selected formal N=8 config sets 5. Top-m ranks diagonal and off-diagonal entries together, and identity mixing can add a self edge. Do not describe 5 as five non-self neighbors.
- [x] Add seeded reproduction and atomic, read-back-verified stage checkpoint saving (`cil/reproduce_moce.py`).
- [ ] Complete a fresh formal CIFAR-100 N=8 reproduction on the other PC. This is not a bitwise reproduction of the unavailable original checkpoint. Preserve its exact configuration/seed if it becomes available.
- [x] Add optional paired per-sample graph/gate capture (`cil/sample_capture.py`).
- [ ] Create an actual sample specialization figure: attach fine labels and stable dataset indices in the evaluation driver, identify examples with shared anchors and different graph neighborhoods, and show actual images/anchors/shared and distinct neighbors/weighted edges. CIFAR-100 has no airplane class: use actual classes, e.g. rocket and a bird species. The optional collector does not automatically attach image identities, and no figure-generation driver is provided yet.
- [ ] Analyze task size versus involvement using paired adjacency/gates, stratified by stage and layer. Use `sample_metrics` for structural unions and effective degree. Compare fixed-k runs separately from adaptive-k runs; report routing diversity and weighted neighborhood measures. Task-level union counts depend on sample count: use equal-size resampling. Dense support involves all experts, so positive-edge union size is uninformative there. Do not threshold averaged adjacency to invent individual neighborhoods.
- [ ] Verify U1–U8 schedules against actual configs/logs: check number of entries, sum of class counts, and any repeated schedule blocks. Do not assume five entries represent 20 steps.
- [ ] Audit input-conditioned adjacency: each ordered pair has its own output coordinate in the N*N adjacency head; optional logit symmetrization couples pairs. Learned expert embeddings already exist in node features; pairwise embedding-conditioned edge prediction is a different alternative. Distinguish node-state communication from executing neighboring adapters.
- [ ] Compare existing even/uneven MoCE-minus-MoE results with matched backbone, N, epochs, schedule, seeds, and anchor budgets. Keep exploratory one-epoch/smoke runs separate from formal results.
- [ ] Organize routing/utilization evidence by run provenance, evaluation stage, task, and layer; distinguish archived group-mean graphs from new sample graphs.
- [ ] Optional: implement a parameter-matched graph-free router control after agreeing on the parameter budget and matched training protocol. Not implemented or launched.

## What is ready

See [REPRODUCE_MOCE.md](REPRODUCE_MOCE.md) for setup, exact launch commands, outputs, and limitations.

The initial UbuntuPC audit examined 770 graph configs (529 dense) and 8,232 archived routing groups. Those counts describe the files available there, not 8,232 independent experiments. Generated analysis files and checkpoints are intentionally not versioned. Run the audit locally to regenerate reports from whatever results are present.

The short UbuntuPC training attempt was stopped at the user's request. No completed reproduction or trained checkpoint is claimed. Existing tracked training code/configs were not edited; the new launcher wraps them.

## Acceptance checks on the other PC

- [ ] `--cfg job` prints the intended formal configuration.
- [ ] GPU is available without interrupting existing jobs; dataset and backbone are present.
- [ ] Console logs show all ten increments completing without traceback or nonfinite loss.
- [ ] Ten `CHECKPOINT_VERIFIED` lines and ten `task_XX.pt` files exist.
- [ ] Final `metrics.json` has ten stage records plus the final summary.
- [ ] `provenance.json`, Hydra config, console log, and environment export are preserved with the run.
- [ ] Before interpreting a checkpoint, reconstruct the model using its stored config and load `model_state_dict` strictly; restore class order, current class names, text tokens, and the appropriate task router. A standalone checkpoint-evaluation/resume driver remains to be implemented.
