# Graph-over-Experts (GoE) Mixer for MoE-Adapters

## Overview

This refactoring adds a **Graph-over-Experts (GoE) mixer** to the MoE-Adapters continual learning framework. The key innovation is that **inactive experts** (not selected by top-k routing) can still contribute to the final output through learned graph message passing.

## Key Features

### 1. **Configurable Number of Experts (N)**
- Easily change the number of experts via config files or command-line overrides
- No code changes required to experiment with different expert counts
- Default: `N=4` experts

### 2. **Fixed Top-K Selection (K)**
- Keep `k=2` for standard setup (easily configurable if needed)
- Ensures consistent routing behavior across experiments

### 3. **Graph-over-Experts Mixer**
- Learns a per-sample adjacency matrix **A ∈ ℝ^{B×N×N}** (row-stochastic)
- Propagates messages from ALL experts (including inactive ones)
- Lightweight proto-heads avoid full expert computation
- Optional symmetrization and self-loops
- Learnable fusion weight `α` (initialized to 0.0 for baseline compatibility)

### 4. **Backward Compatibility**
- With `graph_alpha_init=0.0`, reproduces baseline MoE behavior
- Existing training/routing logic remains unchanged
- All changes are additive and config-driven

## Architecture

### Graph Mixer Components

```
Input: x_re [B, D] (pooled token features)
├─> Adjacency Predictor: A_head(x_re) → A [B, N, N] (row-softmax)
├─> Proto Experts: {h_i(x_re)}_{i=1}^N → X_all [B, N, D]
├─> Graph Propagation: A @ X_all → Y_propagated [B, N, D]
└─> Output Projection: GELU(Y_propagated) @ Wg → Y_all [B, N, D]
```

### Fusion with Standard MoE

```
Standard MoE Output: y_moe = Σ_{e∈top-k} gates[e] * expert_e(x)
Graph Output:        y_graph = Σ_e gates[e] * Y_all[e]
Final Output:        y_total = y_moe + α * y_graph
```

## Configuration

### Config File: `cil/configs/experts.yaml`

```yaml
# Number of experts (N) - easily changeable for experiments
num_experts: 4

# Top-k selection (k) - keep at 2 for standard setup
top_k: 2

# Graph-over-Experts (GoE) Mixer Settings
graph_mixer_enabled: true        # Enable graph mixer
graph_symmetrize: true           # Symmetrize adjacency matrix A
graph_add_self_loop: true        # Add self-loops to adjacency
graph_alpha_init: 0.0            # Initial α (0.0 = baseline at init)
graph_entropy_weight: 0.0        # Entropy regularization on A

# Expert adapter parameters
ffn_bottleneck: 64              # Bottleneck dimension
```

### Overriding Config at Runtime

```bash
# Change number of experts to 8
python main.py experts.num_experts=8

# Disable graph mixer (baseline MoE)
python main.py experts.graph_mixer_enabled=false

# Enable entropy regularization
python main.py experts.graph_entropy_weight=0.01

# Change alpha initialization
python main.py experts.graph_alpha_init=0.5
```

## Usage

### 1. Run with Default Settings (N=4, k=2, Graph Mixer Enabled)

```bash
bash run_cifar100-2-2-MoE-GNN.sh
```

### 2. Run with Different Number of Experts

```bash
# Using environment variable
EXPERTS_N=8 bash run_cifar100-2-2-MoE-GNN.sh

# Or with direct override
python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml" \
    experts.num_experts=6
```

### 3. Baseline MoE (No Graph Mixer)

```bash
python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml" \
    experts.graph_mixer_enabled=false
```

## Code Structure

### New Files

1. **`cil/continual_clip/graph_mixer.py`**
   - `GraphExpertMixer` module
   - Adjacency prediction
   - Proto expert heads
   - Graph message passing
   - Diagnostic utilities

2. **`cil/configs/experts.yaml`**
   - Expert configuration
   - Graph mixer settings
   - Easy to modify for experiments

3. **`cil/run_cifar100-2-2-MoE-GNN.sh`**
   - Example run script with graph mixer
   - Shows how to override N via environment or CLI

### Modified Files

1. **`cil/clip/model.py`**
   - `ResidualAttentionBlock.__init__`: Accept cfg, create graph mixer
   - `ResidualAttentionBlock.forward`: Integrate graph mixer output
   - `Transformer.__init__`: Pass cfg to blocks
   - `VisualTransformer.__init__`: Pass cfg to transformer
   - `CLIP.__init__`: Accept and store cfg
   - `build_model`: Pass cfg to CLIP

2. **`cil/clip/clip.py`**
   - `load`: Accept cfg parameter, pass to build_model

3. **`cil/continual_clip/models.py`**
   - `ClassIncremental.__init__`: Pass cfg to clip.load
   - `ClassIncremental.train`: Accumulate graph mixer extra losses

4. **`cil/main.py`**
   - Add configuration logging at startup
   - Display k, n, and graph mixer settings

5. **`cil/configs/class/*.yaml`**
   - Add experts config group to defaults

## Training with Graph Mixer

### Loss Computation

The total loss includes:
1. **Cross-entropy loss** (standard)
2. **Graph entropy regularization** (optional, if `graph_entropy_weight > 0`)

```python
loss_total = loss_ce + Σ_blocks (graph_entropy_weight * entropy_loss)
```

### Learnable Parameters Added

Per `ResidualAttentionBlock`:
- **Adjacency predictor**: `A_head` (D → N²)
- **Proto expert heads**: N × (D → D)
- **Output projection**: `Wg` (D → D)
- **Fusion weight**: `α` (scalar, initialized to `graph_alpha_init`)

## Diagnostics

### Configuration Logging (Startup)

```
============================================================
Configuration: k=2, n=4 experts
Graph Mixer: ENABLED
  - Symmetrize: True
  - Self-loops: True
  - Alpha init: 0.0
  - Entropy weight: 0.0
============================================================
```

### Per-Block Logging (Model Initialization)

```
text transformer: k=2, n=4 experts
image transformer: k=2, n=4 experts
```

### Adjacency Statistics (Can be logged during training)

```python
stats = graph_mixer.get_adjacency_stats(A)
# Returns: {
#   'mean_diag': float,        # Average diagonal values
#   'mean_off_diag': float,    # Average off-diagonal values  
#   'mean_entropy': float      # Average row entropy
# }
```

## Acceptance Criteria ✓

- [x] Running `bash run_cifar100-2-2-MoE-GNN.sh` prints `Configuration: k=2, n=4 experts`
- [x] Changing experts with `EXPERTS_N=8` works without code edits
- [x] With `graph_alpha_init=0.0`, model reproduces baseline behavior at initialization
- [x] `alpha_graph` is learnable and can become >0 during training
- [x] No changes to `SparseDispatcher` or existing expert execution
- [x] Backward compatible with existing configs (via defaults)

## Experiment Suggestions

### 1. Varying Number of Experts
```bash
for N in 2 4 6 8; do
    EXPERTS_N=$N bash run_cifar100-2-2-MoE-GNN.sh
done
```

### 2. Graph Mixer vs Baseline
```bash
# With graph mixer
python main.py experts.graph_mixer_enabled=true

# Baseline (no graph mixer)
python main.py experts.graph_mixer_enabled=false
```

### 3. Different Alpha Initializations
```bash
for alpha in 0.0 0.1 0.5 1.0; do
    python main.py experts.graph_alpha_init=$alpha
done
```

### 4. Entropy Regularization
```bash
for weight in 0.0 0.001 0.01 0.1; do
    python main.py experts.graph_entropy_weight=$weight
done
```

## Implementation Details

### Why Proto Heads?
Running all N full experts would be expensive. Instead, we use lightweight linear projections (`proto` heads) to generate cheap approximations of expert outputs for graph propagation.

### Why α Initialized to 0.0?
Starting with `α=0.0` ensures the graph path has zero contribution initially, making the model numerically identical to the baseline MoE. As training progresses, `α` can learn to increase if the graph mixer is helpful.

### Symmetrization & Self-Loops
- **Symmetrization**: `A = 0.5 * (A + A^T)` encourages bidirectional expert relationships
- **Self-loops**: Adding identity before softmax helps experts retain their own information

## Debugging Tips

1. **Check if graph mixer is initialized**:
   ```python
   assert model.visual.transformer.resblocks[0].graph_mixer is not None
   ```

2. **Verify α is learning**:
   ```python
   alpha = model.visual.transformer.resblocks[0].alpha_graph.item()
   print(f"Current α: {alpha}")
   ```

3. **Monitor adjacency statistics**:
   ```python
   A, _, _ = graph_mixer(x_re)
   stats = graph_mixer.get_adjacency_stats(A)
   print(stats)
   ```

4. **Check extra losses are accumulated**:
   ```python
   extra_loss = model.visual.transformer.resblocks[0]._extra_losses
   print(f"Extra loss: {extra_loss}")
   ```

## Citation

If you use this Graph-over-Experts extension, please cite the original MoE-Adapters paper and acknowledge this extension.

---

**Last Updated**: October 2025  
**Compatible with**: MoE-Adapters4CL original codebase

