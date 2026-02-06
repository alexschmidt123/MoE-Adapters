# Uneven CIFAR-100: Experiment Design (With vs Without GNN) and Difference from Original MoE

## What the current code is designed for

The uneven CIFAR-100 setup supports **two experiment types**:

| Config | GNN | Method name | Description |
|--------|-----|-------------|-------------|
| `cifar100_uneven10-MoE-Adapters-N4.yaml` | **No** | MoE-Adapters-N4 | **Original MoE**: sparse top-k routing + adapters only, same as even-split MoE but with uneven task sizes. |
| `cifar100_uneven10-MoE-Adapters-N4-GoE.yaml` | **Yes** | MoE-Adapters-N4-GoE | **GoE (Graph-over-Experts)**: GNN runs before routing; routing and experts use the GNN-refined representation. |

So the code is designed for **both**: (1) baseline MoE without GNN, and (2) MoE + GNN (GoE), on the **same** uneven task schedule `task_sizes: [5, 15, 8, 12, 10, 8, 15, 10, 10, 7]`.

---

## Difference between the two (no GNN vs GoE)

### 1. **MoE only (no GNN)** — `graph_mixer_enabled: false`

- **Flow (in each ResBlock):**  
  `x_re` (CLS token) → **Router** → gates → **Experts** (on original `x`) → combine → `y_output`.
- **Router:** Uses **fixed** `top_k` (e.g. 2); no task-adaptive top-k.
- **Experts:** Get the **original** transformer representation `x`; no graph processing.
- **No** GNN module is created; no extra parameters for graph/adjacency.

### 2. **GoE (with GNN)** — `graph_mixer_enabled: true` + proper GNN

- **Flow (in each ResBlock):**  
  `x_re` → **GNN** (`ProperGraphExpertMixer`) → `x_gnn` → **Router** → gates → **Experts** (on **GNN output** `x_gnn` expanded to sequence) → combine → `y_output`.
- **Router:** Can use **task-adaptive top_k** when `global_num_classes_current_task` is set (e.g. more experts for heavier tasks), via `task_adaptive_top_k_base` and `task_adaptive_top_k_step`.
- **Experts:** Get the **GNN-refined** representation, not the raw `x`.
- **GNN:** Builds an expert graph (adjacency from input), does message passing over expert nodes, aggregates back to a single vector, then projects to `d_model`; that vector is used for both routing and expert input.

So the main differences are: **whether a GNN runs before the router**, **what representation the router and experts see** (raw vs GNN-refined), and **optional task-adaptive top_k** in the GoE path.

---

## Difference from “original MoE” in the codebase

“Original MoE” here means: **MoE-Adapters with sparse top-k routing and adapters, no graph**. That is the same as the **no-GNN** uneven config above.

- **Same as original MoE:**
  - Sparse top-k gating (e.g. top_k=2), optional noisy gating.
  - Per-task router parameters (`router_list[task_id]`, `w_noise_list[task_id]`).
  - Same ResBlock layout: attention → MoE (router + SparseDispatcher + adapters) → residual.
  - Experts are adapters (e.g. `adaptmlp_list`); same training (adapters + router + noise; optionally graph_mixer when enabled).

- **What’s different for uneven CIFAR (vs even-split CIFAR) is the scenario, not the MoE core:**
  - **Scenario:** Uneven class split (`task_sizes: [5, 15, 8, ...]`) instead of fixed increment (e.g. 10-10 or 2-2).
  - **Number of tasks:** Still 10 tasks; only the number of classes per task changes.
  - **Training/eval:** Same CLIP + MoE training; only the dataset and class counts per task come from the uneven scenario.

So:

- **Uneven N4 (no GNN)** = original MoE + uneven task schedule.
- **Uneven N4-GoE** = original MoE + **GNN before router/experts** + optional task-adaptive top_k + same uneven schedule.

---

## Summary table

| Aspect | Original MoE (even or uneven, no GNN) | GoE (uneven, with GNN) |
|--------|--------------------------------------|-------------------------|
| Routing input | CLS token `x_re` | GNN output `x_gnn` |
| Expert input | Original sequence `x` | GNN output expanded to sequence |
| top_k | Fixed (e.g. 2) | Can be task-adaptive (more experts for more classes) |
| Extra module | None | `ProperGraphExpertMixer` (adjacency + GNN layers + proj) |
| Task schedule (uneven) | Same uneven `task_sizes` | Same uneven `task_sizes` |

The code is designed to run **both** on uneven CIFAR-100 so you can compare **original MoE** vs **GoE** under the same class schedule.
