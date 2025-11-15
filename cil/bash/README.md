# Bash Scripts (Reference Only)

This folder contains individual run scripts for reference purposes.

## Main Script

**Use `run.sh` in the parent directory** to run any experiment:

```bash
bash run.sh configs/class/xxxx.yaml
```

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

**Alternative (using individual script):**
```bash
bash bash/run_cifar100_2-2-MoE-Adapters-N4-GoE.sh
```

Both methods work, but using `run.sh` is recommended for consistency.

