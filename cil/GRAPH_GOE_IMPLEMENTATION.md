# Graph-over-Experts (GoE) Implementation

## Overview

This document describes the implementation of the **Graph-over-Experts (GoE)** mixer for the CIL (Continual Incremental Learning) module. The GoE mixer allows inactive experts (not selected by top-k routing) to influence the final output through learned adjacency matrices and message passing.

## Architecture

### Core Components

1. **GraphExpertMixer** (`graph_mixer.py`)
   - Learns per-sample expert adjacency matrix **A ∈ ℝ^{B×N×N}**
   - Creates lightweight proto-features **X_all ∈ ℝ^{B×N×D}** for all experts
   - Performs message passing: **Y_all = proj(GELU(A @ X_all))**
   - Returns graph-mixed expert features

2. **ResidualAttentionBlock** (Modified in `clip/model.py`)
   - Integrates GraphExpertMixer when enabled
   - Fuses graph messages with standard MoE output
   - Supports optional entropy regularization on adjacency matrix
   - **Fully backward compatible** with existing configurations

3. **Training Loop** (Modified in `continual_clip/models.py`)
   - Collects and adds extra losses from graph mixer
   - Includes graph mixer parameters in optimizer
   - No impact on training when graph mixer is disabled

## Key Features

### 1. Configurable Number of Experts

The number of experts **N** is now fully configurable via Hydra config:

```yaml
model:
  num_experts: 4  # Easy to change via config or CLI
  top_k: 2        # Number of experts activated per sample
```

**Default:** `num_experts=2`, `top_k=2` (preserves original behavior)

### 2. Graph Mixer Parameters

```yaml
model:
  graph_mixer_enabled: true       # Enable/disable graph mixer
  graph_symmetrize: true          # Symmetrize adjacency: A = (A + A^T) / 2
  graph_add_self_loop: true       # Add identity before normalization
  graph_alpha_init: 0.0           # Initial fusion weight (0 = baseline)
  graph_entropy_weight: 0.0       # Entropy regularization (0 = disabled)
```

### 3. Message Passing Flow

```
Input x [L, B, D]
    ↓
Pool to x_re [B, D] (CLS token)
    ↓
┌───────────────────────┬──────────────────────┐
│   Standard MoE Path   │   Graph Mixer Path   │
├───────────────────────┼──────────────────────┤
│ Router: x_re → gates  │ Adjacency: A [B,N,N] │
│ Dispatch to top-k     │ Proto: X_all [B,N,D] │
│ Expert forward        │ Message: A @ X_all   │
│ Combine: y_moe        │ Output: Y_all [B,N,D]│
└───────────────────────┴──────────────────────┘
                ↓
    Fusion: y_fused = y_moe + α * y_graph
                ↓
         Output [L, B, D]
```

Where:
- **y_graph[b] = Σ_e gates[b,e] · Y_all[b,e]** (weighted by router gates)
- **α** is a learnable scalar (`alpha_graph`)

## Files Modified/Added

### New Files

1. **`cil/graph_mixer.py`**
   - GraphExpertMixer class implementation
   - ~100 lines, self-contained module

2. **`cil/configs/class/cifar100_2-2-MoE-Adapters-GoE.yaml`**
   - New configuration for Graph-over-Experts
   - Inherits from base config, overrides graph settings

3. **`cil/run_cifar100-2-2-MoE-GoE.sh`**
   - Bash script to run GoE experiments
   - Follows same pattern as existing scripts

4. **`cil/verify_graph_goe.py`**
   - Verification script to test implementation
   - Runs unit tests on GraphExpertMixer and backward compatibility

### Modified Files

1. **`cil/clip/model.py`**
   - Added optional `cfg` parameter to:
     - `ResidualAttentionBlock.__init__`
     - `Transformer.__init__`
     - `VisualTransformer.__init__`
     - `CLIP.__init__`
     - `build_model()`
   - Added graph mixer initialization in `ResidualAttentionBlock`
   - Added graph forward path in `ResidualAttentionBlock.forward()`
   - Added entropy regularization support
   - **Backward compatible:** All new parameters are optional with defaults

2. **`cil/clip/clip.py`**
   - Added optional `cfg` parameter to `load()`
   - Passes cfg to `build_model()`

3. **`cil/continual_clip/models.py`**
   - Passes cfg to `clip.load()` in `ClassIncremental.__init__`
   - Added graph mixer parameter collection in training loop
   - Added extra loss handling (entropy regularization)

## Backward Compatibility

### Old Configs (No Changes Required)

When using existing configurations like `cifar100_2-2-MoE-Adapters.yaml`:

```bash
# This still works exactly as before
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml"
```

**Behavior:**
- `cfg` is passed but graph settings are absent
- `getattr(cfg.model, 'graph_mixer_enabled', False)` returns `False`
- Graph mixer stays disabled (`graph_mixer = None`)
- Standard MoE path executes as before
- **Zero overhead, identical accuracy**

### Default Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_experts` | 2 | Number of experts in MoE |
| `top_k` | 2 | Number of experts activated |
| `graph_mixer_enabled` | False | Graph mixer disabled |
| `graph_symmetrize` | True | Symmetrize adjacency |
| `graph_add_self_loop` | True | Add self-loops |
| `graph_alpha_init` | 0.0 | Start at baseline |
| `graph_entropy_weight` | 0.0 | No regularization |

## Usage Examples

### 1. Run with Graph Mixer (New)

```bash
# Use the new GoE configuration
bash cil/run_cifar100-2-2-MoE-GoE.sh
```

This runs with:
- 4 experts (`num_experts=4`)
- top-2 routing (`top_k=2`)
- Graph mixer enabled
- Adjacency symmetrization and self-loops

### 2. Override Number of Experts via CLI

```bash
# Run with 8 experts instead of 4
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters-GoE.yaml \
    model.num_experts=8 \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml"
```

### 3. Enable Graph Mixer on Existing Config

```bash
# Add graph mixer to existing config without creating new YAML
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters.yaml \
    model.num_experts=4 \
    model.graph_mixer_enabled=true \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml"
```

### 4. Tune Graph Alpha and Entropy

```bash
# Start with higher alpha and add entropy regularization
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path configs/class \
    --config-name cifar100_2-2-MoE-Adapters-GoE.yaml \
    model.graph_alpha_init=0.1 \
    model.graph_entropy_weight=0.01 \
    dataset_root="../datasets/" \
    class_order="class_orders/cifar100.yaml"
```

## Verification

Run the verification script to test the implementation:

```bash
cd cil
python verify_graph_goe.py
```

This will test:
- ✓ GraphExpertMixer import
- ✓ GraphExpertMixer instantiation
- ✓ Forward pass and shape validation
- ✓ Row-stochastic adjacency matrix
- ✓ Backward compatibility (cfg=None)

## Technical Details

### Adjacency Matrix Construction

1. **Logits:** `A_logits = Linear(x_re)` → reshape to [B, N, N]
2. **Symmetrization (optional):** `A_logits = (A_logits + A_logits^T) / 2`
3. **Self-loops (optional):** `A_logits += I`
4. **Normalization:** `A = softmax(A_logits, dim=-1)` (row-wise)

**Result:** Row-stochastic adjacency where `A[b,i,:]` sums to 1

### Proto-Features vs Heavy Adapters

- **Proto-features:** Lightweight linear projections `X_all[i] = Linear_i(x_re)`
- **Adapters:** Heavy MLP blocks (only run for top-k experts)
- **Benefit:** Graph mixer doesn't increase expert forward cost

### Fusion Weight α

- Initialized to 0.0 (starts at baseline MoE)
- Learnable parameter (trained with SGD)
- Can increase to blend graph messages into output

### Entropy Regularization (Optional)

Encourages uniform attention distribution in adjacency:

```python
H(A) = -Σ_j A[i,j] * log(A[i,j])
loss = loss + λ * H(A)
```

**Use case:** Prevent over-concentration on single expert connections

## Trainable Parameters

With graph mixer enabled, the following new parameters are trained:

| Module | Parameters | Shape |
|--------|-----------|-------|
| `graph_mixer.A_head` | Weight, Bias | [D, N²], [N²] |
| `graph_mixer.proto[i]` | Weight (×N) | [D, D] ×N |
| `graph_mixer.proj` | Weight | [D, D] |
| `alpha_graph` | Scalar | [1] |

**Total:** ~N·D² + N² + D additional parameters

Example for D=512, N=4:
- Proto linears: 4 × 512² ≈ 1M params
- A_head: 512 × 16 ≈ 8K params
- Proj: 512² ≈ 262K params
- **Total:** ~1.3M params (small compared to full model)

## Performance Considerations

- **Inference:** Minimal overhead if `alpha_graph ≈ 0`
- **Training:** Extra forward/backward for graph mixer (~10-15% slower)
- **Memory:** Additional N×D proto-features stored per layer
- **Scaling:** Graph mixer is O(N²) in adjacency prediction

**Recommendation:** Use N ≤ 16 for efficient training

## Implementation Principles

1. **Additive only:** No deletions, only additions to codebase
2. **Backward compatible:** Old configs run unchanged
3. **Configurable:** All settings controllable via Hydra
4. **Modular:** Graph mixer is self-contained module
5. **Documented:** Extensive comments and docstrings

## Future Extensions

Potential enhancements (not implemented):

- [ ] Multi-head graph mixer (multiple adjacencies)
- [ ] Learned edge features (not just scalar weights)
- [ ] Task-specific adjacency parameters
- [ ] Sparse adjacency (top-k connections)
- [ ] Temporal smoothness for continual learning

## Troubleshooting

### Graph mixer not training

**Check:** Are graph parameters in optimizer?
```bash
# Look for "graph_mixer" and "alpha_graph" in trainable params
python main.py ... 2>&1 | grep -i graph
```

### Accuracy drop with graph mixer

**Solution:** Graph starts at α=0, needs training to improve
- Increase training iterations
- Try higher learning rate for graph params
- Monitor `alpha_graph` value during training

### Import errors

**Check:** Relative import path in `clip/model.py`:
```python
from ..graph_mixer import GraphExpertMixer  # Correct
```

## Summary

This implementation adds a powerful Graph-over-Experts mechanism to the CIL module while maintaining full backward compatibility. Users can:

- Run existing experiments unchanged
- Enable graph mixer via simple config change
- Scale number of experts via CLI override
- Experiment with different graph settings

The graph mixer allows **all experts to contribute** to the output through learned message passing, potentially improving continual learning performance by leveraging inactive expert knowledge.

