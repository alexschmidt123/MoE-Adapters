# MoE with GNN - Detailed Structure

## Overall Architecture

```
+===========================================================================+
|                        ResidualAttentionBlock                             |
+===========================================================================+
|                                                                           |
|   Input x [L, B, D]                                                       |
|        |                                                                  |
|        v                                                                  |
|   +------------------+                                                    |
|   | Multi-Head       |                                                    |
|   | Attention        |                                                    |
|   | + LayerNorm      |                                                    |
|   +--------+---------+                                                    |
|            |                                                              |
|            v                                                              |
|   x = x + Attention(LN(x))                                                |
|            |                                                              |
|            v                                                              |
|   x_re = x[0, :, :] [B, D]  (CLS token - pooled representation)           |
|            |                                                              |
|            |                                                              |
|   +--------+----------------------------------------------------------+   |
|   |        |                                                          |   |
|   |        v                                                          |   |
|   |   +==========+     +===========================================+  |   |
|   |   |  Router  |     |           GraphExpertMixer                |  |   |
|   |   +==========+     +===========================================+  |   |
|   |        |           |                                           |  |   |
|   |        |           |  +-------------+      +----------------+  |  |   |
|   |        |           |  | A_head      |      | Proto Gen x N  |  |  |   |
|   |        |           |  | (Adjacency) |      | (per expert)   |  |  |   |
|   |        |           |  +------+------+      +--------+-------+  |  |   |
|   |        |           |         |                      |          |  |   |
|   |        |           |         v                      v          |  |   |
|   |        |           |    A [B,N,N]             X_all [B,N,D]    |  |   |
|   |        |           |         |                      |          |  |   |
|   |        |           |         +----------+-----------+          |  |   |
|   |        |           |                    |                      |  |   |
|   |        |           |                    v                      |  |   |
|   |        |           |         +--------------------+            |  |   |
|   |        |           |         | Graph Msg Passing  |            |  |   |
|   |        |           |         | Y = proj(act(A@X)) |            |  |   |
|   |        |           |         +---------+----------+            |  |   |
|   |        |           |                   |                       |  |   |
|   |        |           |                   v                       |  |   |
|   |        |           |            Y_all [B, N, D]                |  |   |
|   |        |           +===========================================+  |   |
|   |        |                               |                          |   |
|   |        v                               v                          |   |
|   |   +----------+                   +----------+                     |   |
|   |   |  MoE     |                   |   GNN    |                     |   |
|   |   |  Path    |                   |   Path   |                     |   |
|   |   +----+-----+                   +----+-----+                     |   |
|   |        |                              |                           |   |
|   |        v                              v                           |   |
|   |   y_moe [B,L,D]                  y_graph [B,L,D]                  |   |
|   |   (top-k experts)                (ALL experts)                    |   |
|   |        |                              |                           |   |
|   |        +-------------+----------------+                           |   |
|   |                      |                                            |   |
|   |                      v                                            |   |
|   |              +---------------+                                    |   |
|   |              |    Fusion     |                                    |   |
|   |              | y_fused =     |                                    |   |
|   |              | y_moe + α*y_g |                                    |   |
|   |              +-------+-------+                                    |   |
|   |                      |                                            |   |
|   +----------------------+--------------------------------------------+   |
|                          |                                                |
|                          v                                                |
|   +-----------------------------------------------------------------+     |
|   |                    MLP Block                                    |     |
|   |   x_out = x + MLP(LayerNorm(x)) + y_fused                       |     |
|   +-----------------------------------------------------------------+     |
|                          |                                                |
|                          v                                                |
|                   Output [L, B, D]                                        |
|                                                                           |
+===========================================================================+
```

---

## Detailed Component Breakdown

### 1. Router (Top-k Gating)

```
+=========================================================================+
|                            Router                                       |
+=========================================================================+
|                                                                         |
|   Input: x_re [B, D]                                                    |
|                |                                                        |
|                v                                                        |
|   +------------------------+                                            |
|   | W_gate [D, N]          |  (learnable weights)                       |
|   | W_noise [D, N]         |  (learnable noise weights)                 |
|   +------------------------+                                            |
|                |                                                        |
|                v                                                        |
|   logits = x_re @ W_gate  [B, N]                                        |
|                |                                                        |
|                +------------------+                                     |
|                |                  |                                     |
|                v                  v                                     |
|   +------------------+    +------------------+                          |
|   | Top-k Selection  |    | Full Softmax     |                          |
|   | (for MoE path)   |    | (for GNN path)   |                          |
|   +--------+---------+    +--------+---------+                          |
|            |                       |                                    |
|            v                       v                                    |
|   +------------------+    +------------------+                          |
|   | top_k_logits     |    | full_gates       |                          |
|   | [B, k]           |    | [B, N] DENSE     |                          |
|   +--------+---------+    | all > 0          |                          |
|            |              +------------------+                          |
|            v                                                            |
|   top_k_gates = softmax(top_k_logits) [B, k]                            |
|            |                                                            |
|            v                                                            |
|   gates = scatter(zeros[B,N], indices, top_k_gates)                     |
|   [B, N] SPARSE (only k non-zero)                                       |
|                                                                         |
|   Output:                                                               |
|     - gates [B, N]: sparse, for MoE dispatch                            |
|     - full_gates [B, N]: dense, for GNN aggregation                     |
|                                                                         |
+=========================================================================+

Example (N=8, k=2):
    logits     = [2.1, 1.8, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5]
    
    gates      = [0.6, 0.4,  0,   0,   0,    0,    0,    0 ]  <- SPARSE
    full_gates = [.30, .25, .12, .10, .08, .06,  .05,  .04]  <- DENSE
```

---

### 2. MoE Path (Sparse Expert Computation)

```
+=========================================================================+
|                         MoE Path                                        |
+=========================================================================+
|                                                                         |
|   Input: x [L, B, D], gates [B, N] (sparse)                             |
|                |                                                        |
|                v                                                        |
|   +---------------------------+                                         |
|   |     SparseDispatcher      |                                         |
|   |   dispatch(x, gates)      |                                         |
|   +-------------+-------------+                                         |
|                 |                                                       |
|     +-----------+-----------+-----------+-----+                         |
|     |           |           |           |     |                         |
|     v           v           v           v     v                         |
|  +------+   +------+   +------+   +------+   +------+                   |
|  |Expert|   |Expert|   |Expert|   |Expert|   |Expert|  ... (N experts)  |
|  |  E1  |   |  E2  |   |  E3  |   |  E4  |   |  E5  |                   |
|  |ACTIVE|   |ACTIVE|   |EMPTY |   |EMPTY |   |EMPTY |                   |
|  +--+---+   +--+---+   +------+   +------+   +------+                   |
|     |          |          |          |          |                       |
|     v          v          v          v          v                       |
|    y1         y2       (none)     (none)     (none)                     |
|     |          |                                                        |
|     +----+-----+                                                        |
|          |                                                              |
|          v                                                              |
|   +---------------------------+                                         |
|   |     SparseDispatcher      |                                         |
|   |   combine(outputs, gates) |                                         |
|   +-------------+-------------+                                         |
|                 |                                                       |
|                 v                                                       |
|   y_moe = 0.6*y1 + 0.4*y2  [B, L, D]                                    |
|   (weighted sum of top-k expert outputs)                                |
|                                                                         |
+=========================================================================+

Expert Architecture (Adapter):
+------------------------------------------+
|              Expert (Adapter)            |
+------------------------------------------+
|                                          |
|   Input: x [batch, L, D]                 |
|           |                              |
|           v                              |
|   +------------------+                   |
|   | Linear (down)    |  D -> bottleneck  |
|   | [D, bottleneck]  |                   |
|   +--------+---------+                   |
|            |                             |
|            v                             |
|   +------------------+                   |
|   | GELU Activation  |                   |
|   +--------+---------+                   |
|            |                             |
|            v                             |
|   +------------------+                   |
|   | Linear (up)      |  bottleneck -> D  |
|   | [bottleneck, D]  |                   |
|   +--------+---------+                   |
|            |                             |
|            v                             |
|   Output: y [batch, L, D]                |
|                                          |
|   bottleneck = 64 (homogeneous)          |
|             or varies (HMoE)             |
+------------------------------------------+
```

---

### 3. GraphExpertMixer (GNN Path)

```
+=========================================================================+
|                       GraphExpertMixer                                  |
+=========================================================================+
|                                                                         |
|   Input: x_sample [B, D] (pooled CLS token)                             |
|                |                                                        |
|   +------------+------------------------------------------+             |
|   |            |                                          |             |
|   v            v                                          v             |
|                                                                         |
|  +==================+   +==================+   +=====================+  |
|  |    A_head        |   |  adj_noise_head  |   |  Proto Generators   |  |
|  | (Adjacency MLP)  |   | (Noise MLP)      |   |  P1, P2, ..., PN    |  |
|  +==================+   +==================+   +=====================+  |
|                                                                         |
+=========================================================================+

### 3.1 Adjacency Predictor (A_head)

+------------------------------------------+
|            A_head (MLP)                  |
+------------------------------------------+
|                                          |
|   Input: x_sample [B, D]                 |
|           |                              |
|           v                              |
|   +------------------+                   |
|   | build_mlp(       |                   |
|   |   D -> N*N,      |  (configurable)   |
|   |   hidden_dims,   |                   |
|   |   activation     |                   |
|   | )                |                   |
|   +--------+---------+                   |
|            |                             |
|            v                             |
|   A_logits [B, N*N]                      |
|            |                             |
|            v                             |
|   reshape to [B, N, N]                   |
|            |                             |
|            v (if training & noisy)       |
|   +------------------+                   |
|   | + Gaussian noise |  (exploration)    |
|   +--------+---------+                   |
|            |                             |
|            v (if symmetrize)             |
|   +------------------+                   |
|   | A = (A + A^T)/2  |                   |
|   +--------+---------+                   |
|            |                             |
|            v (if add_self_loop)          |
|   +------------------+                   |
|   | A = A + I        |                   |
|   +--------+---------+                   |
|            |                             |
|            v                             |
|   +------------------+                   |
|   | softmax(dim=-1)  |  row-stochastic   |
|   +--------+---------+                   |
|            |                             |
|            v                             |
|   A [B, N, N]  (adjacency matrix)        |
|                                          |
+------------------------------------------+

Adjacency Matrix Example (N=4):
         E1    E2    E3    E4
    E1 [0.4,  0.3,  0.2,  0.1]  <- row sums to 1
    E2 [0.2,  0.5,  0.2,  0.1]
    E3 [0.1,  0.2,  0.5,  0.2]
    E4 [0.1,  0.1,  0.3,  0.5]

    A[i,j] = how much expert i receives from expert j


### 3.2 Proto-Feature Generators (N separate MLPs)

+------------------------------------------+
|      Proto Generator P_i (per expert)    |
+------------------------------------------+
|                                          |
|   Input: x_sample [B, D]                 |
|           |                              |
|           v                              |
|   +------------------+                   |
|   | build_mlp(       |                   |
|   |   D -> D,        |                   |
|   |   hidden_dims,   |  (configurable)   |
|   |   activation     |  e.g., depth=11   |
|   | )                |                   |
|   +--------+---------+                   |
|            |                             |
|            v                             |
|   h_i [B, D]  (proto-feature for E_i)    |
|                                          |
+------------------------------------------+

Proto Depth Configurations:
  - depth=2:  D -> D  (single linear)
  - depth=3:  D -> 512 -> D
  - depth=5:  D -> 512 -> 256 -> 128 -> D
  - depth=11: D -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 -> D

All N proto generators run in parallel:
  X_all = stack([P_1(x), P_2(x), ..., P_N(x)])  [B, N, D]


### 3.3 Graph Message Passing

+------------------------------------------+
|         Graph Message Passing            |
+------------------------------------------+
|                                          |
|   Input:                                 |
|     A [B, N, N]  (adjacency matrix)      |
|     X_all [B, N, D]  (proto-features)    |
|                                          |
|           |                              |
|           v                              |
|   +------------------+                   |
|   | messages =       |                   |
|   | bmm(A, X_all)    |  [B,N,N]@[B,N,D]  |
|   +--------+---------+  -> [B, N, D]     |
|            |                             |
|            v                             |
|   +------------------+                   |
|   | GELU activation  |                   |
|   +--------+---------+                   |
|            |                             |
|            v                             |
|   +------------------+                   |
|   | proj (MLP)       |  D -> D           |
|   +--------+---------+                   |
|            |                             |
|            v                             |
|   Y_all [B, N, D]                        |
|   (graph-mixed features for ALL experts) |
|                                          |
+------------------------------------------+

Message Passing Visualization (for expert E1):
    
    Y_all[E1] = proj(GELU(A[E1,:] @ X_all))
              = proj(GELU(0.4*h1 + 0.3*h2 + 0.2*h3 + 0.1*h4))
              
    Each expert receives weighted combination of ALL proto-features
```

---

### 3.4 How Experts Work in GNN Path (Detailed Flow)

This section shows the complete flow of how ALL N experts participate in the GNN path, even when only k experts are active in the MoE path.

```
+=========================================================================+
||              Expert Participation in GNN Path                          |
+=========================================================================+
||                                                                         |
||   Input: x_re [B, D] (pooled CLS token)                                 |
||           |                                                             |
||           +---------------------------------------------------+         |
||           |                                                   |         |
||           v                                                   v         |
||   +===============+                              +===================+ |
||   |  Router       |                              | GraphExpertMixer  | |
||   |  (Top-k)      |                              |                   | |
||   +===============+                              +===================+ |
||           |                                                   |         |
||           | (sparse gates)                                    |         |
||           v                                                   |         |
||   MoE Path: Only top-k experts                                |         |
||   (E1, E2 active, E3-E8 inactive)                            |         |
||                                                               |         |
||                                                               |         |
||   GNN Path: ALL N experts participate                        |         |
||                                                               |         |
||           +---------------------------------------------------+         |
||           |                                                   |         |
||           v                                                   v         |
||   +------------------+                              +----------------+ |
||   | Proto Generators |                              | Adjacency      | |
||   | (N separate MLPs)|                              | Predictor      | |
||   +------------------+                              +----------------+ |
||           |                                                   |         |
||   +-------+-------+-------+-------+-------+-------+-------+  |         |
||   |       |       |       |       |       |       |       |  |         |
||   v       v       v       v       v       v       v       v  v         |
||  P_1    P_2    P_3    P_4    P_5    P_6    P_7    P_8   A_head        |
||   |       |       |       |       |       |       |       |  |         |
||   v       v       v       v       v       v       v       v  v         |
||  h_1    h_2    h_3    h_4    h_5    h_6    h_7    h_8   A[B,N,N]      |
||   |       |       |       |       |       |       |       |  |         |
||   +-------+-------+-------+-------+-------+-------+-------+  |         |
||           |                                                   |         |
||           v                                                   |         |
||   X_all = stack([h_1, h_2, ..., h_8])  [B, N, D]            |         |
||   (Proto-features for ALL experts)                           |         |
||           |                                                   |         |
||           +-------------------+-------------------------------+         |
||                               |                                         |
||                               v                                         |
||                    +----------------------+                              |
||                    | Graph Message Pass  |                              |
||                    | Y = proj(act(A@X))  |                              |
||                    +----------+-----------+                              |
||                               |                                         |
||                               v                                         |
||   Y_all [B, N, D] = [Y_1, Y_2, Y_3, Y_4, Y_5, Y_6, Y_7, Y_8]            |
||   (Graph-mixed features for ALL experts)                                |
||                                                                         |
+=========================================================================+
```

**Key Points:**

1. **All Experts Generate Proto-Features:**
   - Each expert E_i has its own proto-feature generator P_i: ℝ^D → ℝ^D
   - All N proto-generators run in parallel on the same input x_re
   - Output: X_all = [h_1, h_2, ..., h_N] where h_i = P_i(x_re)

2. **Adjacency Matrix Connects All Experts:**
   - A[i,j] represents how much expert i receives from expert j
   - Each row sums to 1 (row-stochastic)
   - Allows inactive experts (E3-E8) to influence active ones (E1, E2)

3. **Message Passing for Each Expert:**
   ```
   For expert E_i:
     Y_i = proj(GELU(Σ_j A[i,j] * h_j))
         = proj(GELU(A[i,1]*h_1 + A[i,2]*h_2 + ... + A[i,N]*h_N))
   
   This means:
     - Y_1 (from E1) receives contributions from ALL experts
     - Y_2 (from E2) receives contributions from ALL experts
     - Y_3 (from E3, inactive in MoE) ALSO receives contributions from ALL experts
     - ... and so on for all N experts
   ```

4. **All Experts Contribute to Final Output:**
   ```
   y_graph = Σ_i (full_gates[i] * Y_i)
           = full_gates[1]*Y_1 + full_gates[2]*Y_2 + ... + full_gates[N]*Y_N
   
   Unlike MoE path (only top-k), GNN path uses ALL N experts
   ```

**Example: N=8, k=2 (top-k), MoE activates E1 and E2**

```
MoE Path (Sparse):
  - Only E1 and E2 process input
  - E3, E4, E5, E6, E7, E8 are inactive
  - y_moe = gates[1]*E1_output + gates[2]*E2_output

GNN Path (Dense):
  - ALL 8 experts generate proto-features:
    X_all = [h_1, h_2, h_3, h_4, h_5, h_6, h_7, h_8]
  
  - ALL 8 experts participate in message passing:
    Y_1 = proj(GELU(A[1,:] @ X_all))  <- receives from all 8
    Y_2 = proj(GELU(A[2,:] @ X_all))  <- receives from all 8
    Y_3 = proj(GELU(A[3,:] @ X_all))  <- receives from all 8 (inactive in MoE!)
    Y_4 = proj(GELU(A[4,:] @ X_all))  <- receives from all 8 (inactive in MoE!)
    ... (same for E5-E8)
  
  - ALL 8 experts contribute to final output:
    y_graph = full_gates[1]*Y_1 + full_gates[2]*Y_2 + ... + full_gates[8]*Y_8
            = 0.30*Y_1 + 0.25*Y_2 + 0.12*Y_3 + 0.10*Y_4 + 0.08*Y_5 + ...
```

**Why This Matters:**

- **MoE Path:** Only top-k experts (e.g., E1, E2) contribute directly
- **GNN Path:** ALL N experts contribute via graph mixing
- **Result:** Inactive experts (E3-E8) can still influence the output through:
  1. Their proto-features (h_3, h_4, ...) being used in message passing
  2. Their graph-mixed outputs (Y_3, Y_4, ...) being aggregated with full_gates

This is the key innovation: **GNN allows inactive experts to contribute without being directly activated by the router.**

---

### 4. GNN Aggregation (Using Dense Gates)

```
+=========================================================================+
|                     GNN Aggregation                                     |
+=========================================================================+
|                                                                         |
|   Input:                                                                |
|     Y_all [B, N, D]  (graph-mixed features for ALL experts)             |
|     full_gates [B, N]  (DENSE - all experts have non-zero weights)      |
|                                                                         |
|   +-----+-----+-----+-----+-----+-----+-----+-----+                     |
|   | .30 | .25 | .12 | .10 | .08 | .06 | .05 | .04 |  <- full_gates     |
|   +-----+-----+-----+-----+-----+-----+-----+-----+                     |
|   |  *  |  *  |  *  |  *  |  *  |  *  |  *  |  *  |                     |
|   +-----+-----+-----+-----+-----+-----+-----+-----+                     |
|   | Y1  | Y2  | Y3  | Y4  | Y5  | Y6  | Y7  | Y8  |  <- Y_all          |
|   +-----+-----+-----+-----+-----+-----+-----+-----+                     |
|                      |                                                  |
|                      v                                                  |
|   y_graph = einsum("bn,bnd->bd", full_gates, Y_all)                     |
|           = sum_i(full_gates[i] * Y_all[i])                             |
|           = .30*Y1 + .25*Y2 + .12*Y3 + .10*Y4 + .08*Y5 + ...            |
|                      |                                                  |
|                      v                                                  |
|   y_graph [B, D]                                                        |
|                      |                                                  |
|                      v                                                  |
|   y_graph = y_graph.unsqueeze(1).expand(-1, L, -1)                      |
|                      |                                                  |
|                      v                                                  |
|   y_graph [B, L, D]  (broadcast over sequence length)                   |
|                                                                         |
+=========================================================================+
```

---

### 5. Fusion

```
+=========================================================================+
|                          Fusion                                         |
+=========================================================================+
|                                                                         |
|   Inputs:                                                               |
|     y_moe [B, L, D]    (from top-k experts, sparse)                     |
|     y_graph [B, L, D]  (from ALL experts via GNN, dense)                |
|     alpha_graph        (learnable scalar, init=0)                       |
|                                                                         |
|                                                                         |
|   +------------------+          +------------------+                    |
|   |     y_moe        |          |    y_graph       |                    |
|   | (k=2 experts)    |          | (N=8 experts)    |                    |
|   +--------+---------+          +--------+---------+                    |
|            |                             |                              |
|            |                             v                              |
|            |                    +------------------+                    |
|            |                    |  * alpha_graph   |                    |
|            |                    +--------+---------+                    |
|            |                             |                              |
|            +-------------+---------------+                              |
|                          |                                              |
|                          v                                              |
|                  +---------------+                                      |
|                  |  y_fused =    |                                      |
|                  |  y_moe +      |                                      |
|                  |  alpha*y_graph|                                      |
|                  +-------+-------+                                      |
|                          |                                              |
|                          v                                              |
|                  y_fused [B, L, D]                                      |
|                                                                         |
|   alpha_graph starts at 0, learned during training                      |
|   This allows gradual integration of GNN collaboration                  |
|                                                                         |
+=========================================================================+
```

---

## Complete Data Flow Summary

```
Input x [L, B, D]
       |
       v
+------------------+
| Attention + LN   |
+--------+---------+
         |
         v
x_re = x[0,:,:] [B, D]  (CLS token)
         |
         +-----------------+-----------------+
         |                 |                 |
         v                 v                 v
    +---------+      +-----------+     +------------+
    | Router  |      | A_head    |     | Proto Gens |
    +---------+      +-----------+     | P1..PN     |
         |                 |           +------------+
         |                 |                 |
    +----+----+            v                 v
    |         |       A [B,N,N]         X_all [B,N,D]
    v         v            |                 |
  gates    full_gates      +---------+-------+
  (sparse)  (dense)                  |
    |         |                      v
    |         |            +------------------+
    |         |            | Graph Msg Pass   |
    |         |            | Y = proj(act(A@X)|
    |         |            +--------+---------+
    |         |                     |
    |         |                     v
    |         |               Y_all [B,N,D]
    |         |                     |
    v         +---------------------+
    |                               |
    v                               v
+--------+                  +---------------+
| MoE    |                  | GNN Aggregate |
| Path   |                  | einsum(       |
| sparse |                  | full_gates,   |
| k=2    |                  | Y_all)        |
+---+----+                  +-------+-------+
    |                               |
    v                               v
y_moe [B,L,D]               y_graph [B,L,D]
(2 experts)                 (8 experts)
    |                               |
    +---------------+---------------+
                    |
                    v
            +---------------+
            |    Fusion     |
            | y_moe + α*y_g |
            +-------+-------+
                    |
                    v
            y_fused [B,L,D]
                    |
                    v
            +---------------+
            | x + MLP(LN(x))|
            | + y_fused     |
            +-------+-------+
                    |
                    v
            Output [L,B,D]
```

---

## Parameter Summary

| Component | Parameters | Shape |
|-----------|------------|-------|
| Router W_gate | D × N | [768, 8] |
| Router W_noise | D × N | [768, 8] |
| Expert E_i | D × bottleneck + bottleneck × D | varies |
| A_head | D → N×N (MLP) | configurable |
| adj_noise_head | D → N×N (MLP) | configurable |
| Proto P_i (×N) | D → D (MLP) | configurable depth |
| proj | D → D (MLP) | configurable |
| alpha_graph | scalar | [1] |

---

## Key Design Choices

1. **Two types of gates**: 
   - Sparse gates for efficient MoE computation
   - Dense gates for comprehensive GNN aggregation

2. **Proto generators independent of experts**:
   - Take x_re directly (not expert outputs)
   - Enable graph collaboration regardless of routing

3. **Learnable adjacency**:
   - Per-sample A matrix learns expert relationships
   - Noisy adjacency during training for exploration

4. **Gradual fusion**:
   - alpha_graph initialized to 0
   - Model learns to balance MoE and GNN contributions
