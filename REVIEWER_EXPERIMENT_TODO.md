# MoCE reviewer experiments: detailed task list

Updated 2026-11-11. UbuntuPC is a fresh execution host. Historical files present in the checkout are not assumed to establish completed formal results or usable checkpoints. Rerun required evidence on this machine. Scope follows the PI guidance: the four-way comparison, input-level expert relationships, stage-level routing behavior, and implementation/configuration clarification. Optional architecture redesign and broad sweeps are deferred.

## Locations and safety

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
- [ ] Verify paired initial adapter hashes for each seed; graph initialization must not consume the common adapter initialization RNG stream.
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
- [ ] Repeat graph preflight with the final finite-gradient guard and confirm nonzero graph gradients.
- [ ] Test fixed/adaptive budget matrix including clipping, low class counts and stage metadata requirement.
- [ ] Check actual selected counts and finite logits/gradients in both graph and graph-free paths.
- [ ] Verify stable CIFAR test indices/fine labels and aligned gates/adjacency, with no metadata entering policy decisions.
- [ ] Verify initial adapter hash equality across graph and graph-free arms.
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

Formal reviewer runs have not yet launched. Preflight accuracy is diagnostic and must not appear in rebuttal result tables. The earlier proposed 27-run campaign is cancelled in favor of this focused four-way study and PI-directed analyses. Optional parameter-matched MLP and expert-aware adjacency controls are deferred.
