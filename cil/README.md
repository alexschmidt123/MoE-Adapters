# CIL Quickstart (Brief)

## 1) Install dependencies

From repo root:

```bash
pip install -r requirements.txt
```

If no `requirements.txt`, install minimum packages:

```bash
pip install torch torchvision hydra-core omegaconf continuum pillow tqdm
```

## 2) Run CIL

From `cil/`:

```bash
# Linux/macOS
bash run.sh -directory 03122026_uneven_cifar100

# Windows (or cross-platform)
python run.py -directory 03122026_uneven_cifar100
```

Optional: set repeats per config:

```bash
bash run.sh -directory 03122026_uneven_cifar100 -times 5
python run.py -directory 03122026_uneven_cifar100 -times 5
```
