# GNN Configuration Options

Beyond noise, here are **all the configurable parameters** you can experiment with in the Graph-over-Experts (GoE) Mixer:

## 1. **Graph Fusion Weight (`graph_alpha_init`)**
- **Current value**: `0.0` (disabled by default)
- **What it does**: Controls how much the GNN output contributes to the final result
- **Formula**: `y_fused = y_moe + alpha_graph * y_graph`
- **Why change it**: 
  - Start at `0.0` to preserve baseline behavior initially
  - Gradually increase (e.g., `0.1`, `0.5`, `1.0`) to blend GNN contributions
  - This is a **learnable parameter** (it can be trained), but you set the initial value
- **Example values to try**: `0.0`, `0.1`, `0.5`, `1.0`, `2.0`

## 2. **Entropy Regularization (`graph_entropy_weight`)**
- **Current value**: `0.0` (disabled)
- **What it does**: Encourages the adjacency matrix to be more uniform/exploratory
- **Formula**: `loss += graph_entropy_weight * (-sum(p * log(p)))`
- **Why change it**: 
  - Higher values encourage more uniform expert connections (less sparse)
  - Lower values allow more specialized/sparse connections
  - Can help with exploration vs exploitation trade-off
- **Example values to try**: `0.0`, `0.01`, `0.1`, `0.5`, `1.0`

## 3. **Symmetrize Adjacency (`graph_symmetrize`)**
- **Current value**: `true`
- **What it does**: Makes the adjacency matrix symmetric: `A = (A + A^T) / 2`
- **Why change it**: 
  - `true`: Undirected graph (expert i → j same as j → i)
  - `false`: Directed graph (allows asymmetric connections)
  - Symmetric graphs are more stable but less expressive
- **Options**: `true` or `false`

## 4. **Self-Loops (`graph_add_self_loop`)**
- **Current value**: `true`
- **What it does**: Adds identity matrix to adjacency before normalization
- **Why change it**: 
  - `true`: Each expert can directly influence itself (preserves local information)
  - `false`: Experts only influence each other (purely collaborative)
  - Self-loops often help with gradient flow
- **Options**: `true` or `false`

## 5. **Noise Parameters** (you already know these)
- `graph_use_noisy_adjacency`: Enable/disable noise (`true`/`false`)
- `graph_noise_epsilon`: Base noise level (e.g., `0.0`, `0.001`, `0.01`)

## Summary Table

| Parameter | Current | Range/Options | Impact |
|-----------|---------|---------------|--------|
| `graph_alpha_init` | `0.0` | `0.0` to `2.0+` | **High** - Controls GNN contribution strength |
| `graph_entropy_weight` | `0.0` | `0.0` to `1.0+` | **Medium** - Controls adjacency sparsity |
| `graph_symmetrize` | `true` | `true`/`false` | **Medium** - Graph structure (directed vs undirected) |
| `graph_add_self_loop` | `true` | `true`/`false` | **Low-Medium** - Self-connection behavior |
| `graph_use_noisy_adjacency` | varies | `true`/`false` | **High** - Exploration during training |
| `graph_noise_epsilon` | `0.0` or `0.001` | `0.0` to `0.1+` | **High** - Noise magnitude |

## Recommended Experiments

### Experiment 1: Graph Fusion Weight
Create variants with different `graph_alpha_init` values:
- `N16-GoE-Alpha0.1.yaml` → `graph_alpha_init: 0.1`
- `N16-GoE-Alpha0.5.yaml` → `graph_alpha_init: 0.5`
- `N16-GoE-Alpha1.0.yaml` → `graph_alpha_init: 1.0`

### Experiment 2: Entropy Regularization
Create variants with entropy regularization:
- `N16-GoE-Entropy0.01.yaml` → `graph_entropy_weight: 0.01`
- `N16-GoE-Entropy0.1.yaml` → `graph_entropy_weight: 0.1`

### Experiment 3: Graph Structure
Test different graph topologies:
- `N16-GoE-Asymmetric.yaml` → `graph_symmetrize: false`
- `N16-GoE-NoSelfLoop.yaml` → `graph_add_self_loop: false`

### Experiment 4: Combined Settings
Test combinations:
- `N16-GoE-Alpha0.5-Entropy0.01.yaml` → Both alpha and entropy
- `N16-GoE-Alpha1.0-Noise001.yaml` → Alpha + noise

## Implementation Details

### Where These Are Used:

1. **`graph_alpha_init`** → `clip/model.py:339-340`
   - Creates learnable parameter `self.alpha_graph`
   - Used in fusion: `y_fused = y_moe + self.alpha_graph * y_graph`

2. **`graph_entropy_weight`** → `clip/model.py:342, 508-516`
   - Adds entropy loss: `loss += graph_entropy_weight * row_entropy`
   - Encourages uniform adjacency distribution

3. **`graph_symmetrize`** → `graph_mixer.py:97-98`
   - Symmetrizes: `A_logits = (A_logits + A_logits.transpose(-2, -1)) / 2.0`

4. **`graph_add_self_loop`** → `graph_mixer.py:101-104`
   - Adds identity: `A_logits = A_logits + eye.unsqueeze(0)`

## Notes

- **`graph_alpha_init`** is the most impactful parameter - it directly controls how much the GNN contributes
- **`graph_entropy_weight`** can help with training stability and exploration
- **`graph_symmetrize`** and **`graph_add_self_loop`** are structural choices that affect graph topology
- All parameters can be combined for more complex experiments

