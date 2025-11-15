# How to Run Experiments

## Main Script: `run.sh`

**Use this single script to run any experiment:**

```bash
bash run.sh configs/class/xxxx.yaml
```

## Examples

### CIFAR-100 Experiments

```bash
# Baseline (N=2)
bash run.sh configs/class/cifar100_2-2-MoE-Adapters.yaml

# N=4 baseline
bash run.sh configs/class/cifar100_2-2-MoE-Adapters-N4.yaml

# N=8 baseline
bash run.sh configs/class/cifar100_2-2-MoE-Adapters-N8.yaml

# N=2 with GNN
bash run.sh configs/class/cifar100_2-2-MoE-Adapters-N2-GoE.yaml

# N=4 with GNN
bash run.sh configs/class/cifar100_2-2-MoE-Adapters-N4-GoE.yaml

# N=8 with GNN
bash run.sh configs/class/cifar100_2-2-MoE-Adapters-N8-GoE.yaml

# 5-5 split
bash run.sh configs/class/cifar100_5-5-MoE-Adapters.yaml

# 10-10 split
bash run.sh configs/class/cifar100_10-10-MoE-Adapters.yaml
```

### TinyImageNet Experiments

```bash
bash run.sh configs/class/tinyimagenet_100-10.yaml
bash run.sh configs/class/tinyimagenet_100-20.yaml
bash run.sh configs/class/tinyimagenet_100-5.yaml
```

## Experiment Directory Naming

Experiments are saved in:
```
experiments/class/{config_file_name}-{timestamp}/
```

**Format:**
- Config file name (without .yaml) + timestamp
- Timestamp: `MMDDYYYY-HHMMSS` (e.g., `12152024-143022`)

**Example:**
- Config: `cifar100_2-2-MoE-Adapters-N4-GoE.yaml`
- Experiment: `cifar100_2-2-MoE-Adapters-N4-GoE-12152024-143022`

## Reference Scripts

Individual run scripts are kept in `bash/` folder for reference only.
You can still use them if preferred, but `run.sh` is recommended.

See `bash/README.md` for the mapping between individual scripts and config files.

