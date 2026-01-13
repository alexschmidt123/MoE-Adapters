# Architecture Diagrams for MoE-Adapters

This document provides comprehensive architectural diagrams and technical descriptions for Mixture-of-Experts (MoE) adapter configurations, including standard MoE, Heterogeneous MoE (HMoE), and their extensions with Graph Neural Networks (GNN). All diagrams are designed for Word compatibility and publication use.

---

## 1. Standard MoE (Mixture of Experts)

### Architecture Flow

```
                    Input x (B x L x D)
                         |
                         v
                    +----------+
                    |  Router  |  g = softmax(W_g * x_pooled)
                    | (Top-k)  |  Select k=2 experts
                    +-----+----+
                          |
        +-----------------+-----------------+
        |                 |                 |
        v                 v                 v
    +-------+         +-------+         +-------+
    |Expert |         |Expert |         |Expert |
    |  E1   |         |  E2   |         |  E3   |
    |       |         |       |         |       |
    |Cap: M |         |Cap: M |         |Cap: M |
    |       |         |       |         |       |
    |y1=E1  |         |y2=E2  |         |y3=E3  |
    +---+---+         +---+---+         +---+---+
        |                 |                 |
        +-----------------+-----------------+
                          |
                          v
                    +----------+
                    |  Output  |  y_moe = sum(g_i * y_i)
                    +----------+
```

### Description

The standard Mixture of Experts (MoE) architecture implements a sparse expert routing mechanism that enables scalable model capacity while maintaining computational efficiency. The architecture consists of three principal components: a gating network (router), a set of homogeneous expert networks, and an output aggregation mechanism.

**Architecture Overview**: Given an input tensor **x** ∈ ℝ^(B×L×D), where B, L, and D denote batch size, sequence length, and model dimension respectively, the system first extracts a pooled representation **x_pooled** ∈ ℝ^(B×D) via CLS token pooling. This pooled representation serves as input to the gating network.

**Gating Mechanism**: The router implements a learned gating function **g** = softmax(**W_g** · **x_pooled** + **b_g**) ∈ ℝ^(B×N), where **W_g** ∈ ℝ^(D×N) and **b_g** ∈ ℝ^N are learnable parameters, and N denotes the number of experts. The router performs top-k selection (typically k=2) to obtain sparse gating weights **g_topk**, ensuring that only k experts are activated per input sample.

**Expert Networks**: Each expert E_i is implemented as an adapter MLP with uniform capacity M, processing dispatched tokens independently. The homogeneous design ensures that all experts have identical computational requirements and parameter counts, facilitating load balancing and parallelization.

**Output Aggregation**: The final output is computed as **y_moe** = Σ_{i=1}^N **g_i** ⊙ **y_i**, where **y_i** = E_i(**x_i**) represents the output of expert i, and ⊙ denotes element-wise multiplication. This weighted aggregation enables the model to leverage multiple experts while maintaining sparsity through the top-k selection mechanism.

**Key Features:**
- Homogeneous experts: All experts have same capacity M
- Sparse activation: Only top-k (k=2) experts process each token
- Weighted combination: Output = weighted sum of selected expert outputs
- Load balancing: Loss encourages uniform expert usage

**Mathematical Formulation:**
- Gating: g = softmax(W_g * x_pooled + b_g) in R^(B x N)
- Top-k selection: g_topk = top_k(g, k=2)
- Expert output: y_i = E_i(x_i) where E_i is adapter MLP
- Final output: y_moe = sum_{i=1}^N g_i * y_i

---

## 2. HMoE (Heterogeneous Mixture of Experts)

### Architecture Flow

```
                    Input x (B x L x D)
                         |
                         v
                    +----------+
                    |  Router  |  g = softmax(W_g * x_pooled)
                    | (Top-k)  |  Select k=2 experts
                    +-----+----+
                          |
        +-----------------+-----------------+
        |                 |                 |
        v                 v                 v
    +-------+         +-------+         +-------+
    |Expert |         |Expert |         |Expert |
    |  E1   |         |  E2   |         |  E3   |
    |       |         |       |         |       |
    |Cap:M1 |         |Cap:M2 |         |Cap:M3 |
    |(Small)|         |(Med)  |         |(Large)|
    |       |         |       |         |       |
    |M1<M2<M3|         |M1<M2<M3|         |M1<M2<M3|
    |       |         |       |         |       |
    |y1=E1  |         |y2=E2  |         |y3=E3  |
    +---+---+         +---+---+         +---+---+
        |                 |                 |
        +-----------------+-----------------+
                          |
                          v
                    +----------+
                    |  Output  |  y_hmoe = sum(g_i * y_i)
                    +----------+
```

### Description

The Heterogeneous Mixture of Experts (HMoE) architecture extends standard MoE by introducing variable expert capacities, enabling more efficient computational resource allocation. Unlike homogeneous MoE where all experts share capacity M, HMoE assigns distinct capacities M_i to each expert, creating a capacity hierarchy M₁ < M₂ < ... < M_N.

**Heterogeneous Capacity Design**: The architecture employs experts with varying capacities, where smaller experts (e.g., M₁ = 32) are designed to handle simpler tasks, while larger experts (e.g., M₃ = 64, M₄ = 80) process more complex inputs. This design principle enables adaptive resource allocation based on input complexity, reducing computational overhead for simple samples while maintaining capacity for complex ones.

**Gating and Routing**: The router maintains the same top-k selection mechanism as standard MoE, computing gating probabilities **g** = softmax(**W_g** · **x_pooled** + **b_g**). However, the dispatcher routes tokens to experts with consideration of their heterogeneous capacities, enabling capacity-aware token assignment.

**P-Penalty Regularization**: To prevent capacity imbalance and ensure fair expert utilization, HMoE incorporates a P-Penalty loss term: L_P = N × Σ_{i=1}^N (M_i × P̂_i), where P̂_i denotes the average gating probability for expert i. This regularization encourages balanced activation across experts of different capacities, preventing larger experts from dominating the routing distribution.

**Capacity Allocation Strategies**: Expert capacities can be determined via multiple strategies: (1) geometric progression M_i = M_base × r^(i-1), (2) arithmetic progression M_i = M_base + (i-1) × Δ, (3) hybrid strategies combining homogeneous and heterogeneous elements, or (4) manual specification based on domain expertise.

**Key Features:**
- Heterogeneous experts: Different capacities M1 < M2 < M3
- Efficient resource usage: Smaller experts for simple tasks
- P-Penalty loss: Encourages balanced expert activation
- Capacity strategies: Geometric, arithmetic, hybrid, or manual

**Mathematical Formulation:**
- Same gating as MoE
- Expert capacities: M1 != M2 != M3 (heterogeneous)
- P-Penalty: L_P = N * sum_i(M_i * P_hat_i) where P_hat_i is average gating probability
- Final output: y_hmoe = sum_{i=1}^N g_i * y_i

---

## 3. MoE with GNN (Graph-over-Experts)

### Architecture Flow

```
                    Input x [L, B, D]
                         |
                         v
                    Attention Block
                         |
                         v
                    x_re = x[:, 0, :]  [B, D]  (CLS token)
                         |
          +--------------+--------------+
          |                             |
          v                             v
    +----------+                  +-------------+
    |  Router  |                  |  Adjacency  |
    |  (MoE)   |                  |  Predictor |
    |g=softmax |                  |  A_head    |
    |(W_g*x_re)|                  |            |
    +-----+----+                  +------+------+
          |                             |
          | Top-k                       | A_logits
          | Selection                   | -> A [B,N,N]
          | (only k experts)            |
          |                             |
    +-----+-----+                       |
    |     |     |                       |
    v     v     v                       |
+-----+ +-----+ +-----+                 |
| E1  | | E2  | | E3  |                 |
|ACTIVE| |ACTIVE| |INACTIVE|             |
|Expert| |Expert| |Expert|                 |
|      | |      | |      |                 |
|y1    | |y2    | |(no   |                 |
|      | |      | |output)|                 |
+--+---+ +--+---+ +-----+                 |
   |       |       |                      |
   |       |       |  (Missing outputs    |
   +-------+-------+   from inactive)     |
           |                              |
           v                              |
    +----------+                          |
    | y_moe =  |                          |
    |combine(  |                          |
    |only     |                          |
    |active   |                          |
    |experts) |                          |
    |[B,L,D]  |                          |
    +-----+----+                          |
          |                               |
          |                               |
          |         x_re [B, D]           |
          |         (same as router)      |
          |               |               |
          |               v               |
          |      +----------------+       |
          |      | Proto Generators|      |
          |      |  P1, P2, ..., PN |     |
          |      |  (ALL experts)   |     |
          |      |  (from x_re)    |     |
          |      +--------+--------+      |
          |               |               |
          |               v               |
          |      X_all = [h1, h2, ..., hN]|
          |      (ALL experts) [B, N, D]  |
          |               |               |
          |               v               |
          |      +----------------+       |
          |      | Graph Message  |       |
          |      |   Passing      |       |
          |      | Y_all = proj(  |       |
          |      |   act(A @ X))  |       |
          |      +--------+--------+      |
          |               |               |
          |               v               |
          |      Y_all [B, N, D]         |
          |      (outputs for ALL        |
          |       experts, including     |
          |       inactive ones)         |
          |               |               |
          |               v               |
          |      y_graph = einsum(        |
          |        "bn,bnd->bd",         |
          |        gates, Y_all)         |
          |      (GNN replaces missing    |
          |       expert contributions)   |
          |      [B, D] -> [B, L, D]     |
          |               |               |
          +---------------+               |
                      |                   |
                      v                   |
                 +----------+             |
                 |  Fusion  |             |
                 |y_fused =|             |
                 |y_moe +  |             |
                 |alpha*   |             |
                 |y_graph  |             |
                 |(MoE: k  |             |
                 |experts +|             |
                 |GNN: all |             |
                 |experts) |             |
                 +-----+---+             |
                       |                 |
                       v                 |
                 +----------+            |
                 |  Output  |            |
                 |x = x +   |            |
                 |MLP(LN(x))|            |
                 |+ y_fused |            |
                 +----------+
```

### Description

The MoE with Graph Neural Network (MoE+GNN) architecture, also referred to as Graph-over-Experts (GoE), introduces a dual-path design that combines sparse MoE routing with dense graph-based expert collaboration. This architecture enables inactive experts to contribute to the final output through learned graph relationships, enhancing representation richness beyond sparse activation.

**Input Processing**: Given input tensor **x** ∈ ℝ^(L×B×D) (sequence length L, batch size B, model dimension D), the system first processes it through attention and extracts the pooled representation **x_re** = **x**[0, :, :] ∈ ℝ^(B×D) using the CLS token. This pooled representation serves as input to both the router and graph mixer.

**Dual-Path Architecture**: The system processes **x_re** through two parallel pathways: (1) a sparse MoE path implementing standard top-k expert routing, and (2) a dense GNN path enabling all experts to participate via graph message passing, regardless of routing selection.

**Sparse MoE Pathway**: The router computes gating probabilities **g** = softmax(**W_g** · **x_re** + **b_g**) ∈ ℝ^(B×N), performs top-k selection, and dispatches tokens to only the top-k selected experts. Each active expert E_i processes its assigned tokens independently, producing outputs **y_i** = E_i(**x_i**). Inactive experts (with gate value 0) produce no outputs. The MoE output is aggregated as **y_moe** = dispatcher.combine(expert_outputs) ∈ ℝ^(B×L×D), which only includes contributions from the k active experts.

**GNN Replacement Mechanism**: The GNN path provides outputs for ALL experts, including those that were not activated in the MoE path. This mechanism "replaces" the missing contributions from inactive experts. Proto-feature generators P_i: ℝ^D → ℝ^D transform the pooled input **x_re** into expert-specific representations **h_i** = P_i(**x_re**) for ALL N experts, regardless of routing selection. These proto-features are assembled into **X_all** = [**h₁**, ..., **h_N**] ∈ ℝ^(B×N×D), ensuring every expert has a representation.

**Adjacency Matrix Learning**: The adjacency predictor A_head: ℝ^D → ℝ^(N×N) learns per-sample adjacency matrices. The process involves: (1) predicting logits A_logits = A_head(**x_re**) ∈ ℝ^(B×N×N), (2) optionally adding noise during training for exploration, (3) optional symmetrization A_logits = (A_logits + A_logits^T) / 2, (4) optional self-loop addition, and (5) row-wise softmax normalization to obtain row-stochastic adjacency **A** = softmax(A_logits, dim=-1) ∈ ℝ^(B×N×N).

**Graph Message Passing**: Graph message passing is performed via matrix multiplication: **messages** = **A** @ **X_all** ∈ ℝ^(B×N×D), where **A** governs information flow between experts. The messages are activated and projected: **Y_all** = Proj(Act(**messages**)) ∈ ℝ^(B×N×D), where Act is typically GELU and Proj is a learned linear projection. Critically, **Y_all** contains outputs for ALL N experts, including those that were inactive in the MoE path.

**Gating-Based Aggregation**: The graph output is aggregated using the same gating weights from the router: **y_graph** = Σ_{i=1}^N **g_i** ⊙ **Y_all**[i] ∈ ℝ^(B×D), where ⊙ denotes element-wise multiplication. This weighted sum includes contributions from all experts (both active and inactive), with the graph path providing the missing outputs for inactive experts. This is broadcast over sequence length: **y_graph** = **y_graph**.unsqueeze(1).expand(-1, L, -1) ∈ ℝ^(B×L×D).

**Output Fusion**: The final output combines both pathways: **y_fused** = **y_moe** + α · **y_graph**, where **y_moe** contains contributions from only the k active experts, while **y_graph** contains contributions from all N experts (via graph message passing). The learnable fusion weight α ∈ ℝ (initialized to 0) enables gradual integration of graph collaboration during training. The final residual connection is: **x_out** = **x** + MLP(LN(**x**)) + **y_fused**.permute(1, 0, 2).

**Key Insight**: The GNN path replaces the missing expert contributions. In standard MoE, only top-k experts produce outputs, leaving (N-k) experts inactive. The GNN path generates outputs **Y_all** for all N experts through proto-features and graph message passing, effectively replacing the missing contributions from inactive experts and enabling full expert collaboration.

**Key Features:**
- Dual-path architecture: Sparse MoE path + dense GNN path
- Proto-feature generators: Create expert representations for graph mixing
- Per-sample adjacency: Learns expert relationships dynamically
- Graph message passing: Allows inactive experts to influence output
- Fusion: y_fused = y_moe + alpha * y_graph where alpha is learnable

**Mathematical Formulation:**
- MoE path: y_moe = sum_{i=1}^N g_i * E_i(x_i)
- Proto features: h_i = P_i(x_pooled) in R^D for each expert
- Adjacency: A = softmax(A_head(x_pooled)) in R^(B x N x N)
- Message passing: Y = A * X where X = [h1, ..., h_N]
- Graph output: y_graph = sum_{i=1}^N g_i * Y_proj[i]
- Final: y_fused = y_moe + alpha * y_graph

---

## 4. HMoE with GNN (Heterogeneous MoE + Graph-over-Experts)

### Architecture Flow

```
                    Input x [L, B, D]
                         |
                         v
                    Attention Block
                         |
                         v
                    x_re = x[:, 0, :]  [B, D]  (CLS token)
                         |
          +--------------+--------------+
          |                             |
          v                             v
    +----------+                  +-------------+
    |  Router  |                  |  Adjacency  |
    |  (HMoE)  |                  |  Predictor |
    |g=softmax |                  |  A_head    |
    |(W_g*x_re)|                  |            |
    +-----+----+                  +------+------+
          |                             |
          | Top-k                       | A_logits
          | Selection                   | -> A [B,N,N]
          |                             |
    +-----+-----+                       |
    |     |     |                       |
    v     v     v                       |
+-----+ +-----+ +-----+                 |
| E1  | | E2  | | E3  |                 |
|Cap: | |Cap: | |Cap: |                 |
| M1  | | M2  | | M3  |                 |
|(Small)|(Med) |(Large)|                 |
|ACTIVE| |ACTIVE| |INACTIVE|             |
|     | |     | |      |                 |
|y1   | |y2   | |(no   |                 |
|     | |     | |output)|                 |
+--+--+ +--+--+ +-----+                 |
   |       |       |                    |
   |       |       |  (Missing outputs   |
   +-------+-------+   from inactive)    |
           |                            |
           v                            |
    +----------+                        |
    | y_hmoe = |                        |
    |combine(  |                        |
    |only     |                        |
    |active   |                        |
    |experts) |                        |
    |[B,L,D]  |                        |
    +-----+----+                        |
          |                             |
          |         x_re [B, D]          |
          |         (same as router)     |
          |               |              |
          |               v              |
          |      +----------------+      |
          |      | Proto Generators|     |
          |      |  P1, P2, ..., PN |    |
          |      |  (ALL experts)   |    |
          |      |  (from x_re)    |    |
          |      +--------+--------+     |
          |               |              |
          |               v              |
          |      X_all = [h1, h2, ..., hN]|
          |      (ALL experts) [B, N, D] |
          |               |              |
          |               v              |
          |      +----------------+      |
          |      | Graph Message  |      |
          |      |   Passing      |      |
          |      | Y_all = proj(  |      |
          |      |   act(A @ X))  |      |
          |      +--------+--------+     |
          |               |              |
          |               v              |
          |      Y_all [B, N, D]         |
          |      (outputs for ALL       |
          |       experts, including     |
          |       inactive ones)         |
          |               |              |
          |               v              |
          |      y_graph = einsum(      |
          |        "bn,bnd->bd",        |
          |        gates, Y_all)        |
          |      (GNN replaces missing  |
          |       expert contributions) |
          |      [B, D] -> [B, L, D]     |
          |               |              |
          +---------------+              |
                      |                  |
                      v                  |
                 +----------+            |
                 |  Fusion  |            |
                 |y_fused =|            |
                 |y_hmoe + |            |
                 |alpha*   |            |
                 |y_graph  |            |
                 |(HMoE: k |            |
                 |experts +|            |
                 |GNN: all |            |
                 |experts) |            |
                 +-----+---+            |
                       |                |
                       v                |
                 +----------+           |
                 |  Output  |           |
                 |x = x +   |           |
                 |MLP(LN(x))|           |
                 |+ y_fused |           |
                 +----------+
```

### Description

The HMoE with GNN architecture integrates heterogeneous expert capacities with graph-based collaboration, achieving both computational efficiency and rich expert interaction. This design represents the most sophisticated variant, combining the resource efficiency of HMoE with the collaborative benefits of graph message passing.

**Input Processing**: Given input tensor **x** ∈ ℝ^(L×B×D), the system extracts pooled representation **x_re** = **x**[0, :, :] ∈ ℝ^(B×D) using the CLS token, which serves as input to both router and graph mixer.

**Integrated Architecture**: The system simultaneously employs (1) heterogeneous expert capacities M₁ ≠ M₂ ≠ ... ≠ M_N for adaptive resource allocation, and (2) graph-based expert collaboration enabling all experts to contribute regardless of routing selection. This dual mechanism provides optimal balance between efficiency and representation richness.

**Heterogeneous MoE Pathway**: The router computes gating probabilities **g** = softmax(**W_g** · **x_re** + **b_g**) and performs top-k selection. Tokens are dispatched to only the top-k selected experts with heterogeneous capacities M₁ < M₂ < ... < M_N, where smaller experts process simpler inputs efficiently. Inactive experts (with gate value 0) produce no outputs. The HMoE output is aggregated as **y_hmoe** = dispatcher.combine(expert_outputs) ∈ ℝ^(B×L×D), which only includes contributions from the k active experts, with capacity-aware token assignment.

**GNN Replacement Mechanism**: The GNN path provides outputs for ALL experts, including those that were not activated in the HMoE path. This mechanism "replaces" the missing contributions from inactive experts. Proto-feature generators P_i: ℝ^D → ℝ^D transform the pooled input **x_re** into expert-specific representations **h_i** = P_i(**x_re**) for ALL N experts, regardless of routing selection or capacity differences. Critically, proto generators are capacity-invariant: they transform inputs to uniform dimension D regardless of expert capacity M_i, ensuring consistent representation space for graph operations. Proto-features are assembled into **X_all** = [**h₁**, ..., **h_N**] ∈ ℝ^(B×N×D), ensuring every expert has a representation.

**Graph Collaboration Pathway**: The adjacency predictor learns per-sample adjacency matrices **A** ∈ ℝ^(B×N×N) from **x_re**. Graph message passing is performed: **messages** = **A** @ **X_all** ∈ ℝ^(B×N×D), followed by activation and projection: **Y_all** = Proj(Act(**messages**)) ∈ ℝ^(B×N×D). This enables all experts to collaborate through their proto-features, independent of sparse routing.

**Graph Message Passing**: Graph message passing is performed via matrix multiplication: **messages** = **A** @ **X_all** ∈ ℝ^(B×N×D), where **A** governs information flow between experts. The messages are activated and projected: **Y_all** = Proj(Act(**messages**)) ∈ ℝ^(B×N×D), where Act is typically GELU and Proj is a learned linear projection. Critically, **Y_all** contains outputs for ALL N experts, including those that were inactive in the HMoE path.

**Gating-Based Aggregation and Fusion**: The graph output is aggregated using router gates: **y_graph** = Σ_{i=1}^N **g_i** ⊙ **Y_all**[i] ∈ ℝ^(B×D), where this weighted sum includes contributions from all experts (both active and inactive), with the graph path providing the missing outputs for inactive experts. This is broadcast to **y_graph** ∈ ℝ^(B×L×D). The final output fuses both pathways: **y_fused** = **y_hmoe** + α · **y_graph**, where **y_hmoe** contains contributions from only the k active experts, while **y_graph** contains contributions from all N experts (via graph message passing). The learnable fusion parameter α enables gradual integration of graph collaboration. The total loss combines task loss, P-Penalty, and load balancing: L = L_task + λ_p × L_P + λ_lb × L_load.

**Key Insight**: The GNN path replaces the missing expert contributions. In HMoE, only top-k experts produce outputs, leaving (N-k) experts inactive. The GNN path generates outputs **Y_all** for all N experts through proto-features and graph message passing, effectively replacing the missing contributions from inactive experts and enabling full expert collaboration, regardless of capacity differences.

**Key Features:**
- Combines HMoE + GNN: Heterogeneous capacities with graph collaboration
- Efficient: Smaller experts for simple tasks, larger for complex
- Collaborative: Graph allows all experts to contribute
- Balanced: P-Penalty loss encourages fair expert usage

**Mathematical Formulation:**
- HMoE path: y_hmoe = sum_{i=1}^N g_i * E_i(x_i) with heterogeneous capacities
- GNN path: Same as MoE+GNN (proto generators independent of capacity)
- Final: y_fused = y_hmoe + alpha * y_graph
- Loss: L = L_task + lambda_p * L_P-Penalty + lambda_lb * L_load

---

## 5. Proto-Feature Generator Architectures

### Overview

Proto-feature generators are deep neural networks that transform the pooled input representation into expert-specific features for graph mixing. They serve as a bridge between the input and the graph message passing mechanism, creating rich representations that capture what each expert "knows" about the input.

### Proto Depth Comparison

```
Depth 2 (DeepProto):
  D -> 256 -> D
  (1 hidden layer)

Depth 3 (DeeperProto):
  D -> 256 -> 128 -> D
  (2 hidden layers)

Depth 4 (DeepestProto):
  D -> 512 -> 256 -> 128 -> D
  (3 hidden layers)

Depth 11:
  D -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 -> D
  (10 hidden layers)
```

### Description of Proto Generators

Proto-feature generators are essential components in GNN-based MoE architectures (MoE+GNN and HMoE+GNN), serving as the interface between input representations and graph-based expert collaboration. These generators transform pooled inputs into expert-specific features that enable graph message passing.

**Functional Role**: Each proto generator P_i: ℝ^D → ℝ^D implements a learned transformation that maps the pooled input **x_pooled** to an expert-specific representation **h_i** ∈ ℝ^D. These representations encode expert-specific knowledge about the input, computed independently of routing decisions, enabling inactive experts to contribute through graph operations.

**Architectural Design**: Proto generators employ a bottleneck architecture with progressive dimension reduction followed by expansion. The network compresses the input through hidden layers of decreasing dimensionality (e.g., D → 512 → 256 → ... → 1) and subsequently expands to the original dimension D. This bottleneck design enforces information compression, encouraging the learning of compact, semantically meaningful representations.

**Depth-Representation Trade-off**: The depth of proto generators presents a fundamental trade-off: (1) shallow networks (depth 2-4) offer reduced computational cost and parameter count but produce simpler representations, while (2) deep networks (depth 11) generate richer, more expressive features that better capture expert relationships at increased computational expense.

**Capacity Invariance**: Proto generators are designed to be capacity-invariant: regardless of expert capacity M_i, each generator P_i produces outputs of uniform dimension D. This design ensures consistent representation space for graph operations, enabling seamless mixing of features from experts with heterogeneous capacities.

### Detailed Architecture (ProtoDepth=11)

```
Input: x_pooled in R^D
    |
    v
+---------+
| 512 dim |  Linear + LayerNorm + GELU + Dropout
+----+----+
     |
     v
+---------+
| 256 dim |  Linear + LayerNorm + GELU + Dropout
+----+----+
     |
     v
+---------+
| 128 dim |  Linear + LayerNorm + GELU + Dropout
+----+----+
     |
     v
+---------+
|  64 dim |  Linear + LayerNorm + GELU + Dropout
+----+----+
     |
     v
+---------+
|  32 dim |  Linear + LayerNorm + GELU + Dropout
+----+----+
     |
     v
+---------+
|  16 dim |  Linear + LayerNorm + GELU + Dropout
+----+----+
     |
     v
+---------+
|   8 dim |  Linear + LayerNorm + GELU + Dropout
+----+----+
     |
     v
+---------+
|   4 dim |  Linear + LayerNorm + GELU + Dropout
+----+----+
     |
     v
+---------+
|   2 dim |  Linear + LayerNorm + GELU + Dropout
+----+----+
     |
     v
+---------+
|   1 dim |  Linear + LayerNorm + GELU + Dropout
+----+----+
     |
     v
+---------+
|   D dim |  Linear (output, no activation)
+---------+

Output: h in R^D
```

### Description of ProtoDepth=11 Architecture

The ProtoDepth=11 configuration implements the deepest proto-feature generator, providing the most expressive expert representations for graph-based collaboration. This architecture employs a 10-layer bottleneck with progressive dimension reduction followed by expansion.

**Architectural Specification**: The generator processes input **x_pooled** ∈ ℝ^D through a sequence of transformations: D → 512 → 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1 → D. The compression phase (D → 1) forces information abstraction through progressively narrower representations, while the expansion phase (1 → D) reconstructs expert-specific features.

**Layer Composition**: Each hidden layer implements the transformation: **h** = Dropout(GELU(LayerNorm(Linear(**x**)))), where Linear: ℝ^d_in → ℝ^d_out performs linear projection, LayerNorm normalizes activations, GELU provides non-linear activation, and Dropout(p=0.1) applies regularization. The output layer omits activation and normalization: **h_out** = Linear(**h**), producing the final proto-feature.

**Representation Learning**: The deep bottleneck architecture enables hierarchical feature learning across multiple abstraction levels. The compression phase forces the network to learn essential, compressed representations, while the expansion phase reconstructs expert-specific features that capture complex relationships. This design facilitates rich expert representations that enhance graph message passing effectiveness.

**Computational Complexity**: For D=512 and N=8 experts, each proto generator contains approximately 448K parameters, resulting in a total of ~3.58M parameters across all experts. The depth-11 configuration provides optimal representation richness for complex expert relationship modeling while maintaining reasonable parameter overhead.

**Layer Structure:**
- Each hidden layer: Linear -> LayerNorm -> GELU -> Dropout(0.1)
- Output layer: Linear only (no activation/normalization)
- Total: 11 layers (10 hidden + 1 output)

**Parameter Count (for D=512, N=8 experts):**
- Per proto generator: ~448K parameters
- Total for all experts: ~3.58M parameters

---

## Summary Table

| Architecture | Expert Type | Capacity | Graph | Key Innovation |
|--------------|-------------|----------|-------|----------------|
| MoE | Homogeneous | M (uniform) | No | Sparse top-k routing |
| HMoE | Heterogeneous | M1 != M2 != M3 | No | Variable capacities, P-Penalty |
| MoE+GNN | Homogeneous | M (uniform) | Yes | Graph-based expert collaboration |
| HMoE+GNN | Heterogeneous | M1 != M2 != M3 | Yes | Efficient + collaborative |

---

## Configuration Examples

### MoE (N=4, k=2)
```yaml
num_experts: 4
top_k: 2
graph_mixer_enabled: false
```

### HMoE (N=4, k=2, Hybrid)
```yaml
num_experts: 4
top_k: 2
hmoe_enabled: true
hmoe_strategy: "hybrid"
hmoe_base_capacity: 32
graph_mixer_enabled: false
```

### MoE + GNN (N=8, ProtoDepth=11)
```yaml
num_experts: 8
top_k: 2
graph_mixer_enabled: true
graph_proto_layers: [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
graph_use_noisy_adjacency: true
graph_noise_epsilon: 0.001
```

### HMoE + GNN (N=8, Hybrid, ProtoDepth=11)
```yaml
num_experts: 8
top_k: 2
hmoe_enabled: true
hmoe_strategy: "hybrid"
hmoe_base_capacity: 32
graph_mixer_enabled: true
graph_proto_layers: [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
graph_use_noisy_adjacency: true
graph_noise_epsilon: 0.001
```

---

## Design Trade-offs

| Configuration | Speed | Memory | Quality | Best For |
|---------------|-------|--------|---------|----------|
| MoE | Fast | Low | Good | Simple tasks, resource-constrained |
| HMoE | Fast | Very Low | Good | Efficient deployment |
| MoE + GNN | Moderate | Medium | Excellent | Complex tasks, expert collaboration |
| HMoE + GNN | Moderate | Low-Medium | Excellent | Best efficiency + collaboration |

---

## Notes for Word Import

1. **Copy and paste**: These diagrams use simple ASCII characters (+, -, |, v) that render well in Word
2. **Font**: Use a monospace font (Courier New, Consolas) for proper alignment
3. **Tables**: The summary tables will automatically format in Word
4. **Equations**: Mathematical notation uses standard ASCII characters for Word compatibility
5. **Formatting**: After pasting, you may need to adjust font size and spacing for optimal readability

---

*Diagrams designed for Word compatibility. All architectures are compatible with CLIP-based vision-language models.*
