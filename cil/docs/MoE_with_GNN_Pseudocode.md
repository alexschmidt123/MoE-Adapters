# Pseudocode: Current MoE with GNN

**Current structure:** N=4 experts, no coarse router, no proto layer. Order: CLS → GNN → Router → Experts → Output (sequential).

**Shapes:** L = sequence length, B = batch size, D = model dimension, N = number of experts, H = GNN hidden size.

---

## Single-page pseudocode (current structure)

```
// === One transformer block (MoE + GNN) ===
INPUT:  x [L, B, D]

  (1)  x = x + Attention( LayerNorm(x) )
  (2)  x_re = x[0, :, :]                      // CLS  [B, D]
  (3)  x_gnn = GNN(x_re)                      // [B, D]  (no coarse router; see GNN below)
  (4)  gates = Router(x_gnn)                   // top-k  [B, N], sparse
  (5)  x_gnn_seq = broadcast(x_gnn, L)        // [L, B, D]
  (6)  expert_inputs = dispatch(x_gnn_seq, gates)
       for i = 1 to N:  expert_outputs[i] = Expert_i(expert_inputs[i])
       y_output = combine(expert_outputs, gates)   // [L, B, D]
  (7)  x = x + MLP( LayerNorm(x) ) + y_output

OUTPUT:  x [L, B, D]

// === GNN (no coarse router, no proto) ===
  h = input_proj(x_sample)                    // [B, D] → [B, H]
  X = expand(h, [B,N,H]) + expert_embeddings  // [B, N, H]
  A_logits = adjacency_head(x_sample)         // [B, N, N]
  A = row_softmax( symmetrize(A_logits) + I )
  Y = X
  for layer = 1 to L:  Y = GraphConv(Y, A);  if layer>1: Y = Y + Y_prev
  x_gnn = output_proj( mean(Y, dim=1) )       // [B, H] → [B, D]

// === GraphConv ===
  Y = GELU( LayerNorm( Linear(A @ X) ) + X )  // optional residual
```

---

## 1. Main flow (one transformer block)

```
INPUT:  x  with shape [L, B, D]

  (1)  x = x + Attention( LayerNorm(x) )

  (2)  x_re = x[0, :, :]                      // CLS token only  →  [B, D]

  (3)  x_gnn = GNN(x_re)                      // see section 2   →  [B, D]

  (4)  gates = Router(x_gnn)                  // top-k gating     →  [B, N], sparse

  (5)  x_gnn_seq = broadcast(x_gnn, length=L)   // same vector at every position  →  [L, B, D]

  (6)  expert_inputs = dispatch(x_gnn_seq, gates)
       for i = 1 to N:
           expert_outputs[i] = Expert_i(expert_inputs[i])
       y_output = combine(expert_outputs, gates)   →  [L, B, D]

  (7)  x = x + MLP( LayerNorm(x) ) + y_output

OUTPUT:  x  with shape [L, B, D]
```

---

## 2. GNN (ProperGraphExpertMixer)

**Current structure uses N=4 experts; no coarse router.** (Coarse router exists only when N >= 8.)

INPUT:  x_sample  [B, D]  
OUTPUT:  x_gnn  [B, D]

```
  // Node features (no proto: one projection + expert embeddings)
  h = input_proj(x_sample)                     // [B, D] → [B, H]
  X = expand(h to [B, N, H]) + expert_embeddings   // [B, N, H]

  // Adjacency matrix
  A_logits = adjacency_head(x_sample)          // [B, D] → [B, N, N]
  if symmetrize:  A_logits = (A_logits + transpose(A_logits)) / 2
  if add_self_loop:  A_logits = A_logits + identity
  A = row_wise_softmax(A_logits)                // [B, N, N], row-stochastic

  // Message passing: L layers
  Y = X
  for layer = 1 to L:
      Y_new = GraphConv(Y, A)                  // see section 3
      if layer > 1 and use_residual:  Y = Y + Y_new
      else:  Y = Y_new

  // Aggregate and project back to model dim
  y_agg = mean(Y, dim=1)                       // [B, N, H] → [B, H]
  x_gnn = output_proj(y_agg)                   // [B, H] → [B, D]

  return x_gnn
```

---

## 3. GraphConv (one GNN layer)

INPUT:  X [B, N, H],  A [B, N, N]  
OUTPUT:  Y [B, N, H]

```
  messages = batch_matmul(A, X)                // [B, N, N] @ [B, N, H] → [B, N, H]
  Y = Linear(messages)
  Y = LayerNorm(Y)
  if use_residual:  Y = Y + X
  Y = GELU(Y)
  Y = Dropout(Y)
  return Y
```

---

## 4. Router (top-k gating)

INPUT:  x [B, D]  
OUTPUT:  gates [B, N]  (only k nonzero per row)

```
  logits = x @ W_router                        // [B, D] @ [D, N] → [B, N]
  if training:  logits = logits + noise
  probs = softmax(logits)
  (top_k_values, top_k_indices) = topk(probs, k)
  gates = zeros(B, N)
  gates[top_k_indices] = top_k_values
  return gates
```

---

## 5. Dispatch and experts

INPUT:  x_seq [L, B, D],  gates [B, N]  
OUTPUT:  y_output [L, B, D]

```
  // gates[b] says which k experts handle batch item b
  expert_inputs = dispatch(x_seq, gates)       // split tokens by gates

  for i = 1 to N:
      expert_outputs[i] = Expert_i(expert_inputs[i])

  y_output = combine(expert_outputs, gates)    // reorder and weight by gates
  return y_output
```

---

## 6. Shape summary

| Variable      | Shape   |
|---------------|---------|
| x (block in)  | [L,B,D] |
| x_re          | [B,D]   |
| x_gnn         | [B,D]   |
| gates         | [B,N]   |
| x_gnn_seq     | [L,B,D] |
| y_output      | [L,B,D] |
| X (nodes)     | [B,N,H] |
| A             | [B,N,N] |
