# Testing After GoE Changes

## Config changes

The **base GoE config** `02052026_uneven_cifar100/GoE-L2-H768-HeadNone.yaml` was updated to include the new options. All 27 other GoE configs extend it, so they inherit these keys. Defaults are backward-compatible (new features off).

| Option | Default | Enable with (example) |
|--------|---------|------------------------|
| `goe_residual_lambda` | 1.0 | `model.goe_residual_lambda=0.3` |
| `graph_identity_bias_alpha` | 0.0 | `model.graph_identity_bias_alpha=0.5` |
| `graph_adj_top_m` | null | `model.graph_adj_top_m=8` |
| `moe_router_z_loss_weight` | 0.0 | `model.moe_router_z_loss_weight=0.0001` |

**Split LRs:** Set all three at once on the command line: `lr_experts=1e-3 lr_gnn=3e-4 lr_router=5e-4`.

---

## New configs (unified naming)

Configs in **`03032025_uneven_cifar100/`** use the **same naming** as the 02052026 grid (e.g. `GoE-L2-H768-HeadNone`). Folder distinguishes “new options” (03032025) from original (02052026).

### Quick (~2–5 min)

From `cil/`:

```bash
cd /Users/gaoming/Documents/Research/Codes/MoE-Adapters/cil

CUDA_VISIBLE_DEVICES=0 python main.py \
  --config-path configs/class \
  --config-name 03032025_uneven_cifar100/GoE-L2-H768-HeadNone \
  epochs=2 \
  dataset_root=../datasets \
  class_order=class_orders/cifar100.yaml
```

**Check:** Run finishes with no errors; startup log shows e.g. `residual_lambda=0.3`; `experiments/.../metrics.json` has one JSON line per task.

### Full run (15 epochs)

Same command without `epochs=2`. Check last line of `metrics.json` for `"last"` and `"avg"`.

### Alternative: overrides on existing config

Same setup (all new options) by overriding any GoE config from `02052026_uneven_cifar100/`:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config-path configs/class \
  --config-name 02052026_uneven_cifar100/GoE-L2-H768-HeadNone \
  epochs=2 \
  model.goe_residual_lambda=0.3 model.graph_identity_bias_alpha=0.5 \
  model.graph_adj_top_m=8 model.moe_router_z_loss_weight=0.0001 \
  dataset_root=../datasets class_order=class_orders/cifar100.yaml
```

---

## run_test.py / run_test.sh

Run **29 configs** (28 in 02052026 + 1 in 03032025), 3 runs each:

```bash
cd /Users/gaoming/Documents/Research/Codes/MoE-Adapters/cil
python run_test.py
# or: bash run_test.sh
```

The 03032025 config uses the same name as the base: `GoE-L2-H768-HeadNone` (all new options enabled).

---

## Optional: smoke test (defaults only)

Run base GoE with no overrides (new options stay at defaults):

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config-path configs/class \
  --config-name 02052026_uneven_cifar100/GoE-L2-H768-HeadNone \
  epochs=2 \
  dataset_root=../datasets \
  class_order=class_orders/cifar100.yaml
```

---

## Optional: regression vs baseline

1. Run **baseline** (no GNN), full epochs.
2. Run **GoE** with all new options (overrides above), full epochs.
3. Compare last line of each run’s `metrics.json`.

---

## Optional: professor’s 5 ablations

Use any GoE config (e.g. `02052026_uneven_cifar100/GoE-L2-H768-HeadNone`) and add overrides to isolate one change:

| # | Experiment | Overrides |
|---|------------|-----------|
| 1 | Residual logits only | `model.goe_residual_lambda=0.3` (others at default) |
| 2 | Identity bias only | `model.graph_identity_bias_alpha=0.5` |
| 3 | Load balance + z-loss | `model.moe_load_balance_weight=0.01 model.moe_router_z_loss_weight=0.0001` |
| 4 | Sparse adj + L=1, H=512 | `model.graph_adj_top_m=8 model.graph_num_layers=1 model.graph_hidden_dim=512 model.graph_head_layers=[512]` |
| 5 | Split LRs | `lr_experts=1e-3 lr_gnn=3e-4 lr_router=5e-4` |

Log for each: final “last”, “average”; optionally gate entropy and expert-load histogram.

---

## Quick reference

| Goal | What you run | Command / script |
|------|----------------|------------------|
| **Quick (new options)** | 03032025 GoE-L2-H768-HeadNone, 2 epochs | `03032025_uneven_cifar100/GoE-L2-H768-HeadNone` + `epochs=2` |
| **Full (new options)** | 03032025 GoE-L2-H768-HeadNone, 15 epochs | `03032025_uneven_cifar100/GoE-L2-H768-HeadNone` (no override) |
| **run_test** | 29 configs (28 + 1 new), 3 runs each | `python run_test.py` or `bash run_test.sh` |
| Smoke (defaults) | One GoE config, no overrides | `GoE-L2-H768-HeadNone` with `epochs=2` |

All commands assume you are in `cil/` and that `dataset_root` and `class_order` point to your data and class order.
