# MTIL Training and Testing Guide

## Overview

MTIL (Multi-Task Incremental Learning) uses MoE (Mixture of Experts) adapters for continual learning across multiple datasets. The training process involves:
1. Training auto-choosers (routers) for each dataset
2. Training MoE-Adapters sequentially on each dataset
3. Testing on all datasets

## Prerequisites

1. **Environment Setup**
   ```bash
   # Activate your conda environment (if using conda)
   conda activate MoE_Adapters4CL
   ```

2. **Data Location**
   - Set `DATA_LOCATION` environment variable or modify scripts
   - Default: `../datasets/` (relative to mtil directory)
   - Datasets should be in: `{DATA_LOCATION}/Aircraft/`, `{DATA_LOCATION}/Caltech101/`, etc.

3. **Required Datasets**
   - Aircraft, Caltech101, CIFAR100, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, SUN397
   - Some datasets (EuroSAT, SUN397, StanfordCars) require manual download

## Training

### Full-Shot Training

**Script:** `scripts/train/train_full_shot_router11_experts22_1000iters.sh`

```bash
cd /home/grads/g/g.lin/Documents/MoE-Adapters/mtil

# Set data location (if different from default)
export DATA_LOCATION="/path/to/your/datasets"

# Set GPU (default: 0)
export GPU=0

# Run training script
bash scripts/train/train_full_shot_router11_experts22_1000iters.sh
```

**What it does:**
1. **Train Auto-Choosers** (lines 29-57): Trains router models for each dataset sequentially
2. **Train MoE-Adapters** (lines 59-114): Trains MoE adapters on each dataset sequentially
3. **Evaluation** (lines 117-137): Tests all models after training

**Key Parameters:**
- `exp_no`: Experiment name (default: `withFrozen_22experts_1000epoch_11`)
- `num`: Number of experts (default: 22)
- `iterations`: Training iterations per dataset (1000 for first, 1000 for subsequent)
- `frozen_path`: Path prefix for frozen expert configurations

### Few-Shot Training

**Script:** `scripts/train/train_few_shot_router11_experts22_1000iters.sh`

```bash
bash scripts/train/train_few_shot_router11_experts22_1000iters.sh
```

**Differences from full-shot:**
- Uses `--few_shot=5` flag
- Fewer iterations (500 instead of 1000)
- Different learning rates

## Testing/Evaluation

### Full-Shot Testing

**Script:** `scripts/test/Full_Shot_order1.sh`

```bash
cd /home/grads/g/g.lin/Documents/MoE-Adapters/mtil

# Set data location
export DATA_LOCATION="/path/to/your/datasets"

# Set GPU
export GPU=0

# Update checkpoint path in script (line 28)
# model_ckpt_path=ckpt/exp_withFrozen_22experts_1000epoch_11

# Run testing
bash scripts/test/Full_Shot_order1.sh
```

**What it does:**
- Loads trained models and auto-choosers
- Evaluates on all datasets
- Tests each dataset with each trained model checkpoint

### Few-Shot Testing

**Script:** `scripts/test/Few_Shot_order1_test.sh`

```bash
bash scripts/test/Few_Shot_order1_test.sh
```

## Manual Training/Testing

### Single Dataset Training

```bash
cd /home/grads/g/g.lin/Documents/MoE-Adapters/mtil

CUDA_VISIBLE_DEVICES=0 python -m src.main \
    --train-mode=adapter \
    --train-dataset=Aircraft \
    --lr=5e-3 \
    --ls=0.2 \
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

### Single Dataset Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.main --eval-only \
    --train-mode=adapter \
    --eval-datasets=Aircraft \
    --load=ckpt/my_experiment/Aircraft.pth \
    --load_autochooser=ckpt/my_experiment/Aircraft_autochooser.pth \
    --data-location=../datasets \
    --ffn_adapt_where=AdapterDoubleEncoder \
    --ffn_adapt \
    --apply_moe \
    --task_id=200 \
    --multi_experts \
    --experts_num=22 \
    --autorouter \
    --threshold=655e-4
```

## Key Arguments

### Training Arguments
- `--train-mode=adapter`: Use adapter mode
- `--train-dataset`: Dataset to train on
- `--lr`: Learning rate (varies by dataset)
- `--iterations`: Number of training iterations
- `--method=finetune`: Training method
- `--save`: Directory to save checkpoints
- `--load`: Checkpoint to load (for continual training)
- `--ffn_adapt`: Enable FFN adaptation
- `--apply_moe`: Enable MoE
- `--experts_num`: Number of experts (default: 22)
- `--task_id`: Task ID for continual learning
- `--is_train`: Training mode flag
- `--train_chooser`: Train auto-chooser/router
- `--frozen`: Use frozen experts
- `--frozen-path`: Path prefix for frozen expert configs

### Evaluation Arguments
- `--eval-only`: Evaluation mode
- `--eval-datasets`: Comma-separated list of datasets to evaluate
- `--load`: Path to model checkpoint
- `--load_autochooser`: Path to auto-chooser checkpoint
- `--autorouter`: Use auto-router for expert selection
- `--threshold`: Threshold for auto-router (varies by dataset)

## Dataset Order

Default order (as in scripts):
1. TinyImagenet (for chooser training only)
2. Aircraft
3. Caltech101
4. CIFAR100
5. DTD
6. EuroSAT
7. Flowers
8. Food
9. MNIST
10. OxfordPet
11. StanfordCars
12. SUN397

## Output Structure

After training, checkpoints are saved in:
```
ckpt/exp_{exp_no}/
├── Aircraft.pth
├── Aircraft_autochooser.pth
├── Caltech101.pth
├── Caltech101_autochooser.pth
├── ...
└── SUN397.pth
```

## Troubleshooting

1. **Data Location Issues**
   - Ensure `DATA_LOCATION` points to directory containing dataset folders
   - Check that datasets are properly downloaded (some require manual download)

2. **GPU Memory Issues**
   - Reduce `--batch-size` (default: 64)
   - SUN397 uses batch size 32 automatically in training script

3. **Checkpoint Loading**
   - Ensure checkpoint paths match your experiment name
   - Check that `--load` and `--load_autochooser` paths exist

4. **Import Errors**
   - Make sure you're running from the `mtil` directory
   - Use `python -m src.main` (not `python src/main.py`)

## Example Workflow

```bash
# 1. Navigate to mtil directory
cd /home/grads/g/g.lin/Documents/MoE-Adapters/mtil

# 2. Set environment variables
export DATA_LOCATION="../datasets"
export GPU=0

# 3. Run full training
bash scripts/train/train_full_shot_router11_experts22_1000iters.sh

# 4. After training completes, run evaluation
bash scripts/test/Full_Shot_order1.sh
```

## Notes

- Training is sequential: each dataset depends on the previous one
- Auto-choosers are trained first, then MoE adapters
- The `--frozen` flag is used for continual learning to preserve previous task knowledge
- Results are saved to `results.jsonl` by default (can be customized with `--results-db`)
