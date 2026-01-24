# MoE with GNN (Corrected)

**Architecture**: `Input → GNN → Router → Experts → Output`

This project uses the corrected MoE+GNN flow where the GNN processes the pooled input first, and the router/experts operate on the GNN-enhanced representation.

## Corrected Flow

```
Input x [L,B,D]
  └─→ CLS pool x_re [B,D]
       └─→ GNN (ProperGraphExpertMixer)
            ├─ optional coarse router (N >= 8)
            └─ output x_gnn [B,D]
                 └─→ Router (top-k on x_gnn)
                      └─→ Experts (process x_gnn)
                           └─→ Output [L,B,D]
```

## Hybrid Routing

- **N < 8**: `Input → GNN (all experts) → Router → Experts → Output`
- **N >= 8**: `Input → Coarse Router → GNN (subset) → Router → Experts → Output`

## Where It Is Implemented

- `cil/graph_mixer_proper.py`: Proper GNN layer.
- `cil/clip/model.py`: Uses `x_gnn` for routing and expert inputs.

## How It Is Enabled

- Set `graph_mixer_enabled: true` in config.
- Proper GNN is used by default when available.
