# Pseudocode: MoE and MoE + GNN (GoE)

**Notation:** L = sequence length, B = batch size, D = model dimension, N = number of experts, H = GNN hidden size. Task index t picks router weights W_gate(t), W_noise(t).

---

## Part 1: MoE only (no GNN)

One transformer block. Input: x, shape (L, B, D).

```
1.  x = x + Attention( LayerNorm(x) )
2.  x_re = x[0, :, :]                    # CLS token, shape (B, D)
3.  Z = x_re @ W_gate(t)                 # router logits, (B, N)
4.  (train) Z = Z + noise;  noise from softplus(x_re @ W_noise(t))
5.  take top-k of Z per row;  G = softmax on those k entries, rest 0   # gates (B, N), sparse
6.  For each expert i:  U_i = rows of x where G[b,i] > 0
7.  For each expert i:  V_i = Expert_i(U_i)
8.  For each b:  y[b] = sum over i of  G[b,i] * V_i(b)
9.  x = x + MLP(LayerNorm(x)) + y
```

Output: x, shape (L, B, D).

---

## Part 2: MoE + GNN (GoE)

Same block; order: CLS → GNN → Router → Experts (on GNN output) → Combine.

```
1.  x = x + Attention( LayerNorm(x) )
2.  x_re = x[0, :, :]                    # (B, D)

    --- GNN ---
3.  h = x_re @ W_proj_in                  # (B, H)
4.  X = h broadcast to (B, N, H) + E_expert   # node features, E_expert (N, H)
5.  A_logits = adj_head(x_re)             # (B, N, N)
6.  A_logits = (A_logits + A_logits')/2 + I
7.  A[b,i,:] = softmax(A_logits[b,i,:])   # row-stochastic
8.  For each GNN layer:  M = A @ X;  Y = GELU(LayerNorm(M @ W + b) + X)
9.  y_agg = mean(Y over expert dim)       # (B, H)
10. x_gnn = y_agg @ W_proj_out            # (B, D)

    --- Router (on x_gnn) ---
11. Z = x_gnn @ W_gate(t)
12. (train) add noise;  then top-k → sparse gates G (B, N)

    --- Experts (on GNN output) ---
13. x_gnn_seq = broadcast x_gnn to (L, B, D)
14. For each expert i:  U_i = rows of x_gnn_seq where G[b,i] > 0
15. For each expert i:  V_i = Expert_i(U_i)
16. For each b:  y[b] = sum over i of  G[b,i] * V_i(b)
17. x = x + MLP(LayerNorm(x)) + y
```

Output: x, shape (L, B, D).

---

## Part 3: Building blocks in one line each

**Router (noisy top-k):**
- Z = x @ W_gate
- Z = Z + noise (noise from x @ W_noise), then keep only top-k per row, re-softmax → G

**Dispatch:** For expert i, collect all batch indices b with G[b,i] > 0; U_i = x[those b].

**Combine:** For each b, y[b] = sum_i G[b,i] * Expert_i(x[b]).

**GraphConv (one layer):** Y = GELU( LayerNorm(A @ X @ W + b) + X ).

---

## Part 4: Shapes

| Variable   | Shape    |
|-----------|----------|
| x (input) | (L, B, D) |
| x_re      | (B, D)   |
| x_gnn     | (B, D)   |
| G (gates) | (B, N)   |
| X (nodes) | (B, N, H) |
| A         | (B, N, N) |
| y (MoE out) | (L, B, D) |
