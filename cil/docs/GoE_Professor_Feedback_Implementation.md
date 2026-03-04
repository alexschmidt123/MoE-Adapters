# GoE: Professor Feedback → Code & Config

This doc maps the professor’s suggested changes to the codebase and config keys. The **five high-ROI items** from the “do-this-next” list are implemented; the rest are listed for future work.

---

## 1) Baseline-safe GoE (curriculum / residualization)

### Change A: Residual logits — **implemented**

- **Idea:** `Z = Z_base + λ * Z_graph` with λ scheduled 0→1 so the router is baseline-driven early and graph-driven later.
- **Code:** `cil/clip/model.py` (ResidualAttentionBlock forward):
  - `Z_base = x_re @ router_list[taskid]`, `Z_graph = einsum(Y, goe_router_w)`.
  - `Z = Z_base + lam * Z_graph` when `goe_residual_lambda < 1`.
- **Config:**
  - `model.goe_residual_lambda` (float, default 1.0): 0 = baseline only, 1 = graph only. Use e.g. 0.2–0.5 early, then ramp to 1 (scheduling can be added later).

### Change B: Identity-biased adjacency — **implemented**

- **Idea:** `A = α*I + (1-α)*softmax(A_logits)` with α high early → lower later to avoid aggressive mixing before experts are meaningful.
- **Code:** `cil/graph_mixer_proper.py` (ProperGraphExpertMixer.forward): after row-softmax, blend with identity.
- **Config:**
  - `model.graph_identity_bias_alpha` (float, default 0.0): 1 = no mixing (identity only), 0 = current softmax. Try e.g. 0.7→0.3 over time (scheduling TBD).

---

## 2) More expressive router (Change C) — not implemented

- **Idea:** Bilinear `Z_i = (W_q x_re)^T (W_k Y_i) + b_i` or small MLP on `[x_re, Y_i, x_re⊙Y_i]` instead of only `Y_i @ w_i`.
- **Where:** Same router block in `cil/clip/model.py`; add optional bilinear/MLP branch and config flags.

---

## 3) Stability–plasticity knobs

### Change D: Separate learning rates — **implemented**

- **Idea:** `lr_experts` higher (plastic), `lr_gnn` lower, `lr_router` medium (e.g. with warmup).
- **Code:** `cil/continual_clip/models.py` (ClassIncremental.train): if `lr_experts`, `lr_gnn`, `lr_router` are all set, build three param groups (adaptmlp, graph_mixer, router/noise/goe_router_w) and pass per-group `base_lrs` to `cosine_lr`.
- **Config (top-level):**
  - `lr_experts` (float, e.g. 1e-3)
  - `lr_gnn` (float, e.g. 3e-4)
  - `lr_router` (float, e.g. 5e-4)  
  If any is missing, single `lr` is used for all.

### Change E: Noise schedule & top-k schedule — not implemented

- **Idea:** Noise warmup (low → slight increase → anneal); optional k schedule (e.g. k=2 then k=1). Log gate entropy and expert-load histogram.
- **Where:** `noisy_top_k_gating_from_logits` / `noisy_top_k_gating` and training loop; add step/task-dependent noise and k, plus logging.

---

## 4) MoE stabilizers

### Change F: Load-balancing loss — **present; works for GoE**

- **Code:** `cil/clip/model.py`: when not HMoE, `moe_load_balance_weight > 0` adds `cv_squared(importance) + cv_squared(load)` on the GoE path too.
- **Config:** `model.moe_load_balance_weight` (e.g. 0.01).

### Change G: Router z-loss — **implemented**

- **Idea:** Regularize logit magnitude to avoid brittle routing.
- **Code:** `cil/clip/model.py`: after GoE routing, store `_last_router_logits`; in extra_losses block add `0.5 * (log(1+exp(-|Z|))^2).mean()` scaled by `moe_router_z_loss_weight`.
- **Config:** `model.moe_router_z_loss_weight` (float, default 0.0). Try e.g. 1e-4.

---

## 5) CL routing drift (Changes H, I) — not implemented

- **Change H:** Slow-moving expert prototypes (EMA of mean hidden state per expert) instead of fully trainable `E_expert`; use prototypes as node features.
- **Change I:** Gate distillation: KL(prev_gate, curr_gate) on a small replay set.

---

## 6) Sparser adjacency (sweep follow-up)

### Change J: Sparsify adjacency (top-m) — **implemented**

- **Idea:** Keep only top-m edges per node, renormalize; less oversmoothing, clearer structure.
- **Code:** `cil/graph_mixer_proper.py`: after softmax, optionally take top-m per row, zero rest, renormalize; then apply identity bias if used.
- **Config:** `model.graph_adj_top_m` (int or null): number of neighbors per node (e.g. 4 or 8). Omit or null = dense.

---

## 7) “Do-this-next” checklist (all wired in code)

| # | Experiment | Config / code |
|---|------------|----------------|
| 1 | Residual logits, λ ramp 0→1 | `goe_residual_lambda` (fixed for now; ramp in script or callback later) |
| 2 | Identity-biased A, α ramp 1→0.3 | `graph_identity_bias_alpha` |
| 3 | Load balancing + z-loss | `moe_load_balance_weight`, `moe_router_z_loss_weight` |
| 4 | Sparsified adjacency, L=1, H=512, Head512 | `graph_adj_top_m`, `graph_num_layers: 1`, `graph_hidden_dim: 512`, `graph_head_layers: [512]` |
| 5 | LR split | `lr_experts`, `lr_gnn`, `lr_router` (all three required to enable) |

**Suggested logging (for your runs):** final “last”, “average”, expert usage entropy, routing distribution drift task-to-task (professor’s list).

---

## 8) Metric clarification (no code change)

- **Idea:** Report AA_weighted over time, forgetting per task (peak − final), optional forward transfer so “average” vs “last” is interpretable (e.g. imbalanced task sizes).

---

## Example config snippet (high-ROI GoE)

```yaml
# Top-level
lr: 1e-3
# Optional: separate LRs (uncomment all three to enable)
# lr_experts: 1e-3
# lr_gnn: 3e-4
# lr_router: 5e-4

model:
  graph_mixer_enabled: true
  goe_residual_lambda: 0.3        # baseline-safe early (ramp to 1 later)
  graph_identity_bias_alpha: 0.5  # less mixing early
  graph_adj_top_m: 8               # top-8 edges per node
  graph_num_layers: 1
  graph_hidden_dim: 512
  graph_head_layers: [512]
  moe_load_balance_weight: 0.01
  moe_router_z_loss_weight: 0.0001
  goe_experts_on_x: true
  goe_route_from_per_expert: true
```

All of the above keys are optional; omit any to keep current defaults.
