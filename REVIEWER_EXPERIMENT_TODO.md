# MoCE reviewer experiments: detailed task list

Updated 2026-11-11. This is a portable handoff for the PC holding the original checkpoints. UbuntuPC is the development/validation host and has no original trained checkpoints. The checkpoint host should reuse compatible artifacts first and rerun only missing or incompatible evidence. Scope follows the PI guidance: the four-way comparison, input-level expert relationships, stage-level routing behavior, and implementation/configuration clarification. Optional architecture redesign and broad sweeps are deferred.

## Start here on the PC holding checkpoints

This checklist and the code are one handoff. Pull the validated GitHub commit into a clean checkout. Do not overwrite dirty work or running jobs. Use a Python environment with the repository requirements, PyTorch/CUDA, OmegaConf, continuum, matplotlib and numpy. Install missing dependencies according to local policy. Commands below run from repository root; use that machine's dataset and checkpoint paths. Nothing requires the Ubuntu absolute paths listed later.

1. Inventory only trusted, locally owned checkpoint files. `torch.load` can deserialize Python. The audit recognizes full stage checkpoints containing model state, config, stage/class metadata and text tokens; unknown formats are explicit blockers, not guessed conversions.

```bash
python cil/reviewer_handoff.py audit --checkpoint-root /PATH/TO/CHECKPOINTS --output /PATH/TO/HANDOFF/checkpoint_inventory.json
```

2. Read the inventory alongside original configs/logs. Identify the four conditions, seeds, schedules, epochs, training budget and native objectives. Confirm whether original evaluation was task-conditioned. A checkpoint can be reevaluated task-agnostically, but missing full model state or unknown seed/architecture must be resolved before results enter the matched table. The runner also rejects incompatible epochs, learning rates, batch sizes and regularizer application. Old graph-free training without the configured regularizer cannot be relabeled as the new matched control. Do not assume one experiment folder contains all four arms.

3. Generate a manifest and machine-local configs without launching training. Set seeds to the actual matched evidence available. The default is seeds 0,1,2 and the verified descending schedule. Use `--task-sizes` and `--neighbor-budget` only to match verified evidence or a predeclared missing comparison; these flags are not a request for a sweep.

```bash
python cil/reviewer_handoff.py init --dataset-root /PATH/TO/datasets --output /PATH/TO/HANDOFF/study
```

Edit each `checkpoint_dir` in `study/jobs.json` to its compatible directory containing `task_00.pt` through `task_09.pt`. Keep it null only for genuinely missing training. Checkpoint aliases or conversions require a reviewed converter and source hashes, not renaming an unrelated state dictionary. If original config values differ, update the intended configs deliberately, document the reason and reseal `config_sha256` using SHA256 of the exact config bytes; run refuses changed locks. Preserve originals.

Before scientific execution, run diagnostic checks with new output directories (these are excluded from analysis):

```bash
python cil/reviewer_run.py --config /PATH/TO/HANDOFF/study/moce_adaptive_seed0.yaml --output /PATH/TO/HANDOFF/smoke_graph --smoke
python cil/reviewer_run.py --config /PATH/TO/HANDOFF/study/moce_adaptive_seed0.yaml --output /PATH/TO/HANDOFF/smoke_reload --checkpoint-dir /PATH/TO/HANDOFF/smoke_graph/checkpoints --smoke
```

Compare gates, adjacency and sample IDs under `sample_routing` across these two directories; they should agree exactly on a deterministic environment. Repeat the graph-free and fixed-budget preflights by selecting their configs. Smoke mode performs one native update at the native initial warmup learning rate; it verifies gradient plumbing/checkpoint IO, not convergence or a successful scientific optimization schedule.

4. Run mapped checkpoint evaluations first, sequentially. The runner checks configuration, stage, class order and seed and strictly loads every model tensor. It uses the latest stage router for every image and deterministic resize/center-crop preprocessing. An unknown format, missing stage or incompatible config stops the campaign. Existing incomplete outputs remain preserved; choose a new reviewed output path after diagnosing a failure.

```bash
python cil/reviewer_handoff.py run --manifest /PATH/TO/HANDOFF/study/jobs.json
```

If some conditions genuinely need training, explicitly use the following command after the protocol/configuration checks. Compatible mapped checkpoints are still reused, completed outputs are skipped only when their config lock matches, and null checkpoint mappings receive native training. Partial mapped checkpoints stop rather than triggering silent replacement training.

```bash
python cil/reviewer_handoff.py run --manifest /PATH/TO/HANDOFF/study/jobs.json --train-missing
```

5. Generate accuracy tables, paired seed differences, stage routing measures and illustrative sample figures from completed non-smoke runs. The CIFAR root below contains `cifar-100-python/test` and `meta`.

```bash
python cil/reviewer_analyze.py --runs /PATH/TO/HANDOFF/study/runs --cifar-root /PATH/TO/datasets/cifar100 --output /PATH/TO/HANDOFF/rebuttal_analysis
```

Expected outputs: `four_way_per_seed.csv`, `summary.json`, `paired_seed_differences.csv`, `stage_routing_behavior.csv`, sample relationship PNGs when qualifying pairs exist, stage-behavior plots and class-conditioned utilization heatmaps, and `ANALYSIS_LIMITATIONS.md`. Each run stores provenance, metrics, sample reservoirs and a COMPLETE marker. Training additionally stores verified stage checkpoints and a gradient check. The orchestration records GPU/process resource samples every five seconds; CPU percentages are process averages and RSS excludes worker processes, so do not label them total CPU-tree peak RAM.

6. Complete the evidence-dependent tasks below: submitted m/k_g interpretation, original U1-U8 mapping, even/uneven matched comparisons, qualitative interpretation and final reviewer claims. Code does not invent missing manuscript settings, statistical significance, original results or semantic communities. No figure is produced if a qualifying sample pair is absent; report that outcome rather than fabricate a pair. Parameter-matched and expert-aware redesigns remain optional.

### Handoff acceptance gates

- [ ] Checkpoint-host operator reads this checklist, inspects inventory and maps compatible artifacts.
- [ ] Confirm all four arms share intended schedule/training/evaluation settings and verified seed metadata.
- [ ] Confirm the deterministic test transform and latest-router protocol are applied to every arm.
- [ ] Pass smoke/checkpoint validation locally; compare repeated checkpoint outputs before scientific reporting.
- [ ] Finish only missing training under the locked protocol; preserve negative findings.
- [ ] Produce outputs for each requested evidence item, or record its exact unresolved blocker.
- [ ] Commit compact validated reports and source; exclude checkpoints, datasets and large raw outputs.

## Ubuntu development locations and safety

- Main project: `/home/qiqi/Documents/MoE-Adapters`.
- Isolated experimental worktree: `/home/qiqi/Documents/MoE-Adapters-rebuttal-20261111`, branch `reviewer/moce-controls-20261111`.
- Environment: `/home/qiqi/miniconda3/envs/moe-adapters/bin/python`.
- Data: main project `datasets/cifar100`; cached ViT-B/16 backbone is present.
- Formal outputs: isolated worktree `cil/experiments/reviewer_20261111`.
- [x] Verify available GPU: RTX 5080, 16 GB, idle before preflights.
- [x] Isolate new source from original checkout and preserve historical logs/data.
- [ ] Before formal launch, check other GPU jobs, prevent duplicate launches, enforce sequential execution and disk guard.
- [ ] Version validated source/configs and protocol; push scoped changes without checkpoints, data, credentials or large results.

## 1. Lock a valid matched four-way comparison — G2, W2, R2, R4

- [x] Prepare 12 configurations: graph off/on x fixed/adaptive execution x seeds 0/1/2.
- [x] Base settings on the available formal CIFAR100 config: ViT-B/16, N=8, batch32, LR0.001, weight decay0, 15 epochs per increment, schedule [28,19,13,11,8,6,5,4,3,3]. This is a fresh controlled study, not a claimed exact reconstruction of an unidentified submitted run.
- [x] Declare fixed k=2; adaptive k=clamp(2+floor((C_t-10)/5),1,8). Use the same rule in both adaptive methods, in training and evaluation.
- [x] Preserve native CE, load balance and configured router-logit regularizer. Correct unequal regularizer application across graph/non-graph arms; document it rather than calling it standard z-loss.
- [x] Preserve native epochs*batches+1 update count and report it explicitly.
- [x] Retain the formal graph implementation value graph_adj_top_m=5, identity mixing .7, hidden512, one GNN layer, head512 and residual scale100. The formal comment K_g=4 is inconsistent with stored top_m=5; do not relabel this as four non-self neighbors or submitted m8.
- [x] Verify paired seed0 diagnostic adapter hashes; verify each scientific seed separately. graph initialization must not consume the common adapter initialization RNG stream.
- [ ] Audit source/config diffs and freeze their hashes before formal launch.
- [ ] Run all four seed0 conditions sequentially, then paired seeds1 and2; no post-hoc extension or automatic architecture search.
- [ ] Report per-seed final accuracy, average incremental accuracy, forgetting and task accuracies; show mean and variation across seeds.
- [ ] Report graph benefit at fixed and adaptive k, adaptive benefit with and without graph, and factorial interaction. Do not treat images or stages as independent training seeds.
- [ ] Report actual expert counts per stage and image, training time/update, inference latency, GPU/CPU/RAM/VRAM and storage. Distinguish adapter execution from graph-node communication.

## 2. Task-agnostic evaluation — user-selected protocol

- [x] Identify legacy evaluation selecting routers using true test task IDs; it is incompatible with the claimed task-agnostic setting.
- [x] Implement direct inference with the latest learned stage router for every image and all seen class prompts.
- [x] Supply current-stage class count for adaptive budgets; never choose a router or budget from the true test-image task.
- [ ] Verify task-ID permutation invariance of policy outputs; use true task metadata only for label remapping and reporting.
- [ ] Keep historical oracle/task-conditioned results separate; do not silently compare them with new task-agnostic results.

## 3. Correctness and persistence before full training

- [x] Graph adaptive preflight: one native update, expected k=5 at the 28-class stage, checkpoint read-back and sample capture.
- [x] Graph-free adaptive preflight: one native update, expected k=5, checkpoint read-back and sample capture.
- [x] Add finite-gradient assertions; graph-free preflight completes with them enabled.
- [x] Graph preflight with finite-gradient guard: nonzero gradients in 312 graph, 48 router and 276 adapter parameter tensors.
- [x] Test fixed/adaptive budget matrix including clipping, low class counts and stage metadata requirement.
- [x] Diagnostic GPU preflights pass for graph/non-graph x fixed/adaptive arms; scientific runs retain runtime assertions.
- [x] Diagnostic CIFAR indices/fine labels match original data; aligned gates/adjacency repeat exactly across all 12 vision layers. Metadata enters only post-inference reporting.
- [x] Verify seed0 diagnostic initial adapter hash equality across graph and graph-free arms.
- [ ] Save atomic full checkpoints after every stage; read back every tensor before committing the checkpoint filename.
- [ ] Preserve config, class order/names, text tokens, stage, source hashes, seed and RNG state. A stage checkpoint is not an exact mid-update optimizer-resume checkpoint.
- [ ] Require 10 completed stages, verified checkpoints, metrics and a COMPLETE marker per formal run; stop and diagnose failures instead of silently skipping them.

## 4. Input-level expert relationship analysis — W3, R1, R3

- [x] Implement stable image-ID/fine-label capture and sample-level gate/adjacency reservoir across vision layers.
- [x] Verify graph preflight writes paired sample records from all 12 vision layers.
- [ ] Use trained formal checkpoints and records, not preflight outputs, for scientific figures.
- [ ] Identify representative image pairs with shared anchors and distinct weighted neighborhoods. State the selection rule and avoid implying handpicked pairs represent the population.
- [ ] Display actual CIFAR100 images and fine class names, selected anchor IDs, shared/distinct neighbors and edge weights at an identified stage/layer.
- [ ] Use genuine dataset classes: CIFAR100 has no airplane class; use available bird species and other actual classes.
- [ ] Quantify overlap/specialization over many samples, including class-conditioned selection frequencies, entropy and neighborhood variation.
- [ ] Distinguish executed anchors, graph-node information contributors, structural support and weighted effective involvement. Averaged adjacency cannot reconstruct sample neighborhoods.
- [ ] Describe observations as associations; graph visualization alone does not establish useful semantic communities or causal benefit.

## 5. Stage class count versus learned routing — R3, R5

- [ ] Compute stage/layer selection frequency, class-conditioned utilization, input variation, effective weighted neighborhood size and anchor-neighbor unions.
- [ ] Use equal-size sample resampling when comparing union-based statistics across stages.
- [ ] Analyze fixed-k conditions separately to distinguish learned changes from the deterministic adaptive-k formula.
- [ ] Explicitly report self edges and identity mixing; dense positive support trivially involves all nodes.
- [ ] Pair behavior plots with stage accuracy and forgetting; acknowledge class identity/order and training progression confound simple class-count correlations.

## 6. Method and schedule verification — G1, W1, R1

- [ ] Trace m and k_g through method equations, submitted config, constructor and logs. Do not assume notation equivalence or infer a default from comments.
- [x] Verify code ranks diagonal and off-diagonal candidates together; identity mixing may add a self edge.
- [ ] Explain separately total expert pool N, retained graph rows/neighbors, selected anchors k and executed adapter count.
- [ ] Explain pair-specific adjacency-head output parameters; learned expert embeddings already initialize node features. Expert-aware dynamic adjacency is an alternative, not equivalent to a static graph.
- [ ] Verify U1–U8 actual arrays, length and sum against available configs. Resolve 20-step/five-entry discrepancy from evidence; do not invent repetition rules.
- [ ] Prepare a compact configuration table containing alpha, top-m selection/self-edge semantics, adaptive formula, backbone, training budgets and full schedules.
- [ ] If submitted runs truly use all N neighbors, explicitly acknowledge the dense case; renaming m cannot establish variable neighbor count.
- [ ] Decide whether a small neighbor sensitivity experiment is necessary only after submitted-setting verification; no automatic sweep.

## 7. Even versus uneven continual learning — R5

- [ ] Recover compatible manuscript table values or rerun the missing comparison if no usable evidence exists.
- [ ] Match model, N, epochs, seed and execution policy; report MoCE-minus-MoE gains separately for even and uneven settings.
- [ ] Include stage accuracy and forgetting where supported; do not claim a larger uneven advantage before calculating it.
- [ ] Describe varying arriving class counts accurately; class count is not a direct measurement of semantic or visual difficulty.
- [ ] Acknowledge graph routing is general while explaining this CIL motivation. Do not redesign adjacency solely for rebuttal.

## 8. Final evidence package and GitHub updates

- [ ] Table A: verified four-way accuracy, variation, budgets and measured cost.
- [ ] Table B: verified neighbor settings and any justified existing/new sensitivity evidence.
- [ ] Figure A: actual shared/distinct sample relationships with quantitative population context.
- [ ] Figure B: stage routing behavior beyond mechanically assigned k, with accuracy/forgetting context.
- [ ] Verified U1–U8/configuration clarification and even/uneven comparison, with limitations.
- [ ] Map shared evidence once to G2/W2/R2/R4; avoid duplicate experiments for repeated concerns.
- [ ] Save negative results and unsupported claims explicitly. Do not claim graph-specific value from different routing probabilities alone.
- [ ] Push validated source, protocol, checklist and compact reports to GitHub; verify remote commit hash. Keep checkpoints/data/raw large outputs local.

## Current scientific status

The code is being validated on Ubuntu diagnostic checkpoints, not original trained results. Formal reviewer runs have not launched; this handoff does not require repeating compatible training already present on the checkpoint host. Preflight accuracy is diagnostic and must not appear in rebuttal result tables. The earlier proposed 27-run campaign is cancelled in favor of this focused four-way study and PI-directed analyses. Optional parameter-matched MLP and expert-aware adjacency controls are deferred.
