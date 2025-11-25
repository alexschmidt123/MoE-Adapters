# Bash Scripts (Reference Only)

This folder contains individual run scripts for reference purposes.

## Main Script

**Use `run.sh` in the parent directory** to run any experiment:

```bash
bash run.sh configs/class/xxxx.yaml
```

### Epoch Override (Optional)

You can override the epoch number specified in the config file by passing it as a second argument:

```bash
# Run with default epochs from config file
bash run.sh configs/class/cifar100_2-2-MoE-Adapters-N4-GoE.yaml

# Run with epochs=5 (overrides config file)
bash run.sh configs/class/cifar100_2-2-MoE-Adapters-N4-GoE.yaml 5

# Run with epochs=10
bash run.sh configs/class/cifar100_2-2-MoE-Adapters-N4-GoE.yaml 10
```

**Note:** The epoch override is useful for:
- Testing different epoch numbers without modifying config files
- Running epoch studies (see `run_epoch_study.sh`)
- Quick experiments with different training durations

## Individual Scripts (Reference)

These scripts are kept here for reference but are **not needed** - you can use `run.sh` instead.

### CIFAR-100 Scripts

- `run_cifar100_2-2-MoE-Adapters.sh` → `bash run.sh configs/class/cifar100_2-2-MoE-Adapters.yaml`
- `run_cifar100_2-2-MoE-Adapters-N4.sh` → `bash run.sh configs/class/cifar100_2-2-MoE-Adapters-N4.yaml`
- `run_cifar100_2-2-MoE-Adapters-N8.sh` → `bash run.sh configs/class/cifar100_2-2-MoE-Adapters-N8.yaml`
- `run_cifar100_2-2-MoE-Adapters-N2-GoE.sh` → `bash run.sh configs/class/cifar100_2-2-MoE-Adapters-N2-GoE.yaml`
- `run_cifar100_2-2-MoE-Adapters-N4-GoE.sh` → `bash run.sh configs/class/cifar100_2-2-MoE-Adapters-N4-GoE.yaml`
- `run_cifar100_2-2-MoE-Adapters-N8-GoE.sh` → `bash run.sh configs/class/cifar100_2-2-MoE-Adapters-N8-GoE.yaml`
- `run_cifar100_5-5-MoE-Adapters.sh` → `bash run.sh configs/class/cifar100_5-5-MoE-Adapters.yaml`
- `run_cifar100_10-10-MoE-Adapters.sh` → `bash run.sh configs/class/cifar100_10-10-MoE-Adapters.yaml`

### TinyImageNet Scripts

- `run_tinyimagenet_100-10.sh` → `bash run.sh configs/class/tinyimagenet_100-10.yaml`
- `run_tinyimagenet_100-20.sh` → `bash run.sh configs/class/tinyimagenet_100-20.yaml`
- `run_tinyimagenet_100-5.sh` → `bash run.sh configs/class/tinyimagenet_100-5.yaml`

## Usage

**Recommended (using general script):**
```bash
cd /home/grads/g/g.lin/Documents/MoE-Adapters/cil
bash run.sh configs/class/cifar100_2-2-MoE-Adapters-N4-GoE.yaml
```

**With epoch override:**
```bash
bash run.sh configs/class/cifar100_2-2-MoE-Adapters-N4-GoE.yaml 5
```

**Alternative (using individual script):**
```bash
bash bash/run_cifar100_2-2-MoE-Adapters-N4-GoE.sh
```

Both methods work, but using `run.sh` is recommended for consistency.

## Epoch Study Script

For running multiple experiments with different epoch numbers, use `run_epoch_study.sh`:

```bash
# Run epoch study for N2, N4, N8 with epochs 3, 5, 10 (default)
bash run_epoch_study.sh

# Run with custom epoch values
bash run_epoch_study.sh "5 10"
```

This script runs 11 configs × 3 N values (N2, N4, N8) × epoch values = 99 experiments, each run 3 times.

