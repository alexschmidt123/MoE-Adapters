# CIL: Class-Incremental Learning with MoE-Adapters

Minimal layout for running class-incremental learning (CIFAR-100, Food-101, uneven tasks) with MoE and MoE-GNN.

## Layout

```
cil/
├── main.py                 # Entry point (Hydra)
├── run.sh                  # Run one experiment: bash run.sh <config>
├── continual_clip/         # CIL logic (scenarios, models, datasets)
├── clip/                   # CLIP + MoE / MoE-GNN
├── graph_mixer_proper.py   # GNN-over-experts (used by clip when enabled)
├── configs/
│   └── class/              # Experiment configs (*.yaml; _archive/ has bulk subdirs)
├── class_orders/           # Class order YAMLs per dataset
├── dataset_reqs/           # Class names and splits (e.g. imagenet100, food101)
├── scripts/                # Utilities (e.g. generate_class_orders.py)
├── experiments/            # Output dir (created by runs)
└── _archive/               # Unused/legacy code (datasets_ref, templates, batch runners)
```

## Data

- **CIFAR-100:** `dataset_root` should point to the parent of `cifar100/`.  
  Example: `dataset_root: "../datasets"` → data path `../datasets/cifar100`.  
  CIFAR-100 files live under that folder, e.g. `../datasets/cifar100/cifar-100-python/` (with `train`, `test`, `meta`).  
  Absolute example: `/home/grads/g/g.lin/Documents/MoE-Adapters/datasets` so that `cifar100/cifar-100-python` exists.
- **Food-101:** `../datasets/food-101/` with `images/` and `meta/` (train.json, test.json, classes.txt).

## Quick run (from `cil/`)

```bash
# CIFAR-100, 2-2, MoE-Adapters N4 + GNN
bash run.sh configs/class/cifar100_2-2-MoE-Adapters-N4-GoE.yaml

# Food-101, 11-10
bash run.sh configs/class/food101_11-10-MoE-Adapters-N4-GoE.yaml

# CIFAR-100 uneven 10 tasks
bash run.sh configs/class/cifar100_uneven10-MoE-Adapters-N4-GoE.yaml
```

Override data path if needed:

```bash
python main.py --config-path configs/class --config-name cifar100_2-2-MoE-Adapters-N4-GoE dataset_root=/home/grads/g/g.lin/Documents/MoE-Adapters/datasets
```

## Essential configs (minimal set)

- `configs/class/cifar100_2-2-MoE-Adapters-N2.yaml` — base (2-2 CIFAR-100)
- `configs/class/cifar100_2-2-MoE-Adapters-N4-GoE.yaml` — MoE-GNN N4
- `configs/class/cifar100_uneven10-MoE-Adapters-N4-GoE.yaml` — uneven 10 tasks
- `configs/class/food101_11-10-MoE-Adapters-N4-GoE.yaml` — Food-101

All other YAMLs in `configs/class/` are optional variants.

## Dependencies

Install from repo root (or use existing env): `continuum`, `hydra-core`, `omegaconf`, `torch`, `torchvision`, `PIL`, `tqdm`.
