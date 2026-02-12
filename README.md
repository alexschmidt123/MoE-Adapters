# MoE-Adapters for Continual Learning

This repository contains two frameworks for continual learning using Mixture of Experts (MoE) adapters with CLIP:

- **CIL** (Class-Incremental Learning): Continual learning on class-incremental scenarios
- **MTIL** (Multi-Task Incremental Learning): Continual learning across multiple datasets/tasks

## CIL: Class-Incremental Learning

CIL implements class-incremental continual learning where new classes arrive sequentially. It uses MoE adapters with optional Graph Neural Network (GNN) enhancement to improve expert routing and knowledge transfer across tasks.

**Key Features:**
- Supports CIFAR-100, TinyImageNet, ImageNet-100, ImageNet-1000
- MoE adapters with per-task routers
- Optional GNN enhancement for expert coordination
- HMoE (Hybrid MoE) support with dynamic capacity allocation

## MTIL: Multi-Task Incremental Learning

MTIL implements multi-task incremental learning where different datasets/tasks are learned sequentially. It uses MoE adapters with auto-choosers (routers) to select appropriate experts for each task.

**Key Features:**
- Supports 11 datasets: Aircraft, Caltech101, CIFAR100, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, SUN397
- Auto-chooser (router) training for task-specific expert selection
- Sequential task learning with frozen expert preservation
- Full-shot and few-shot learning modes

## GNN Integration in CIL

The GNN (Graph Neural Network) is integrated into the MoE system to enhance expert coordination. The architecture follows: **Input → GNN → Router → Experts → Output**. The coarse router has been removed; the GNN always uses all N experts (same workflow for any N). Only `cil/graph_mixer_proper.py` is used; the legacy `graph_mixer.py` has been removed.

### Architecture (GoE)

```
Input x [L, B, D]
  └─→ CLS: x_re [B, D]
       └─→ GNN (ProperGraphExpertMixer, all N experts)
            ├─→ Adjacency A [B, N, N], node features
            ├─→ Multi-layer message passing → x_gnn [B, D]
            └─→ Router (top-k on x_gnn) → Experts (on x_gnn) → Output [L, B, D]
```

### Enabling GNN

Set `graph_mixer_enabled: true` in your config. Implemented in `cil/graph_mixer_proper.py` and `cil/clip/model.py`.

## Running CIL

### Quick Start

```bash
cd cil

# Run a single config
python main.py \
    --config-path configs/class/cifar_configs \
    --config-name cifar100_2-2-MoE-Adapters-N4-GoE-ProtoDepth11.yaml \
    dataset_root=../datasets \
    class_order=class_orders/cifar100.yaml

# Run test suite (8 GNN configs)
python run_all_cifar_test.py
# or
bash run_all_cifar_test.sh
```

### Configuration

- Config files are in `cil/configs/class/`
- Use Hydra configs with `--config-path` and `--config-name`
- Set `dataset_root` and `class_order` as overrides

### Test Scripts

- `run_test.py` / `run_test.sh`: Run 28 configs (1 baseline MoE + 27 GoE grid) from `configs/class/02052026_uneven_cifar100/` (e.g. `baseline.yaml`, `GoE-L*-H*-Head*.yaml`); use `--config-path configs/class --config-name 02052026_uneven_cifar100/baseline` (etc.), 3 runs each; then write `summary.csv` (last_acc, avg_acc per run and per-config averages). `run_test.py` is Windows-friendly (pathlib, forward slashes for Hydra).
- `run_all_cifar_test.py` / `run_all_cifar_test.sh`: Quick test (8 configs)
- `run_all_cifar.py` / `run_all_cifar.sh`: Full CIFAR-100 suite
- `run_all_imagenet.py` / `run_all_imagenet.sh`: ImageNet suite

## Running MTIL

### Quick Start

```bash
cd mtil

# Set data location
export DATA_LOCATION="../datasets"
export GPU=0

# Full-shot training
bash scripts/train/train_full_shot_router11_experts22_1000iters.sh

# Testing
bash scripts/test/Full_Shot_order1.sh
```

### Training Process

1. **Train Auto-Choosers**: Router models for each dataset
2. **Train MoE-Adapters**: Sequential training on each dataset
3. **Evaluation**: Test all models

### Manual Training

```bash
# Single dataset training
CUDA_VISIBLE_DEVICES=0 python -m src.main \
    --train-mode=adapter \
    --train-dataset=Aircraft \
    --lr=5e-3 \
    --iterations=1000 \
    --method=finetune \
    --save=ckpt/my_experiment \
    --data-location=../datasets \
    --ffn_adapt_where=AdapterDoubleEncoder \
    --ffn_adapt \
    --apply_moe \
    --multi_experts \
    --experts_num=22 \
    --task_id=0 \
    --is_train
```

For detailed MTIL instructions, see `mtil/README_RUN.md`.

## Requirements

- Python 3.9+
- PyTorch
- CLIP
- Hydra (for CIL)
- See individual directories for specific requirements

## Directory Structure

```
MoE-Adapters/
├── cil/          # Class-Incremental Learning
│   ├── configs/  # Hydra configuration files
│   ├── clip/     # CLIP model with MoE+GNN
│   └── main.py   # Main entry point
├── mtil/         # Multi-Task Incremental Learning
│   ├── src/      # Source code
│   ├── scripts/  # Training/testing scripts
│   └── clip/     # CLIP model with MoE
└── datasets/     # Dataset storage (shared)
```
