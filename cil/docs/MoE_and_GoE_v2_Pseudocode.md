# Pseudocode: MoE and MoE + GNN (GoE v2)

**Notation:** L = sequence length, B = batch size, D = model dimension, N = number of experts, H = GNN hidden size. Task index t selects router weights. Optional: task-adaptive top-k (heavier task → more experts).

---

## Part 1: MoE only (no GNN)

One transformer block. Input: x, shape (L, B, D).

```
1.  x = x + Attention( LayerNorm(x) )
2.  x_re = x[0, :, :]                    # CLS token, (B, D)
3.  Z = x_re @ W_gate(t)                 # router logits, (B, N)
4.  (train) Z = Z + noise;  noise from softplus(x_re @ W_noise(t))
5.  top-k of Z per row;  G = softmax on those k, rest 0   # gates (B, N)
6.  For each expert i:  U_i = rows of x where G[b,i] > 0   # token-level x
7.  For each expert i:  V_i = Expert_i(U_i)
8.  For each b:  y[b] = sum_i G[b,i] * V_i(b)
9.  x = x + MLP(LayerNorm(x)) + y
```

Output: x, shape (L, B, D).

---

## Part 2: MoE + GNN (GoE v2) — current default

Same block. Order: **CLS → GNN → Router (from per-expert states) → Experts (on token-level x) → Combine.**

- **Router:** Logits Z[b,i] come from per-expert GNN states Y[b,i], not from a mean-pooled vector.
- **Experts:** Receive the **original sequence x** (token-level), not a broadcast GNN output.

```
1.  x = x + Attention( LayerNorm(x) )
2.  x_re = x[0, :, :]                    # (B, D)

    --- GNN (per-expert node states) ---
3.  h = x_re @ W_proj_in                  # (B, H)
4.  X = h broadcast to (B, N, H) + E_expert   # node features, E_expert (N, H)
5.  A_logits = adj_head(x_re)             # (B, N, N)
6.  A_logits = (A_logits + A_logits')/2 + I;  A[b,i,:] = softmax(A_logits[b,i,:])
7.  For each GNN layer:  M = A @ X;  Y = GELU(LayerNorm(M @ W + b) + X)
8.  Y remains (B, N, H)                  # per-expert states, no mean-pool here

    --- Router (from per-expert states) ---
9.  W_goe(t) = per-task weights (N, H)
10. Z[b,i] = Y[b,i] @ W_goe(t)[i,:]      # logits (B, N), one score per expert per sample
11. (train) Z = Z + noise;  noise from softplus(x_re @ W_noise(t))
12. top-k of Z per row → sparse gates G (B, N)

    --- Experts (on original sequence x) ---
13. For each expert i:  U_i = rows of x where G[b,i] > 0   # x is (L, B, D), token-level
14. For each expert i:  V_i = Expert_i(U_i)
15. For each b:  y[b] = sum_i G[b,i] * V_i(b)
16. x = x + MLP(LayerNorm(x)) + y
```

Output: x, shape (L, B, D).

**Config flags (defaults):** `goe_experts_on_x: true`, `goe_route_from_per_expert: true`. Optional: `goe_residual_alpha` blends CLS with GNN for router input (when not using per-expert routing). Optional: freeze GNN after task index via `goe_freeze_gnn_after_task` to reduce drift.

---

## Part 3: GoE legacy (pooled routing + experts on GNN output)

For reference only; not the default. Router uses mean-pooled GNN output; experts receive broadcast x_gnn.

```
    --- GNN (same as above) ---
    ... Y (B, N, H) ...
    y_agg = mean(Y over expert dim)       # (B, H)
    x_gnn = y_agg @ W_proj_out            # (B, D)

    --- Router (on pooled x_gnn) ---
    Z = x_gnn @ W_gate(t);  (train) add noise;  top-k → G (B, N)

    --- Experts (on broadcast x_gnn) ---
    x_gnn_seq = broadcast x_gnn to (L, B, D)
    For each expert i:  U_i = rows of x_gnn_seq where G[b,i] > 0
    ...
```

Set `goe_experts_on_x: false` and `goe_route_from_per_expert: false` to approximate this behavior.

---

## Part 4: Building blocks

**Router (noisy top-k from features):**  
Z = x @ W_gate; Z = Z + noise(x @ W_noise); keep top-k per row, re-softmax → G.

**Router from logits (GoE per-expert):**  
Z already (B, N); noise from x_re @ W_noise; top-k → G.

**Dispatch:** For expert i, collect batch indices b with G[b,i] > 0; U_i = x[those b] (sequence slice).

**Combine:** y[b] = sum_i G[b,i] * Expert_i(x[b]).

**GraphConv (one layer):** Y = GELU( LayerNorm(A @ X @ W + b) + X ).

**Per-expert logits:** Z[b,i] = Y[b,i] @ W_goe[i]  (einsum `bnh,nh->bn`).

---

## Part 5: Shapes

| Variable   | Shape     |
|-----------|-----------|
| x (input) | (L, B, D) |
| x_re      | (B, D)   |
| x_gnn     | (B, D)   |
| Y (nodes) | (B, N, H) |
| Z (logits)| (B, N)   |
| G (gates) | (B, N)   |
| A         | (B, N, N) |
| W_goe(t)  | (N, H)   |
| y (MoE out)| (L, B, D) |

---

## Part 6: Task-adaptive top-k (optional)

For GoE with uneven task sizes: effective_k = min( N, max(1, top_k + (num_classes_current_task - base) / step ) ). Heavier tasks get more experts; small tasks keep smaller k.
