# Reproduce a MoCE checkpoint on another PC

Training is intentionally not running on UbuntuPC. These commands are for the other Linux GPU PC.

## Get the code and environment

```bash
git clone https://github.com/alexschmidt123/MoE-Adapters.git
cd MoE-Adapters
# For an existing clean checkout instead: git pull --ff-only
conda create -n moe-adapters python=3.10 -y
conda activate moe-adapters
```

Install a PyTorch/torchvision build suitable for that PC's GPU and driver first. The environment validated on UbuntuPC used Python 3.10.20, torch 2.11.0+cu128 and torchvision 0.26.0+cu128. Then install the project requirements and the missing MTIL dependency:

```bash
python -m pip install -r cil/requirements.txt
python -m pip install wilds
python -m pip check
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

The requirement file is legacy and contains old pins and a Git-hosted timm fork. Installation on a fresh PC has not been tested; retain the Git fork and resolve build/compatibility errors explicitly rather than silently replacing dependencies. PyYAML and NumPy are needed by the audit tool. The provided code was import/smoke-tested in the existing UbuntuPC environment, not a new installation.

Place CIFAR-100 under `datasets/cifar100/cifar-100-python/` (train, test, meta). Keep datasets/checkpoints outside Git. CLIP uses `~/.cache/clip/ViT-B-16.pt`; it may download this if absent. Check GPU use before launching and do not interrupt other jobs.

## Inspect the configuration without training

```bash
cd cil
python reproduce_moce.py \
  --config-path configs/class/09082026_formal_moe_moce \
  --config-name s10-MoCE-N8-cifar100-ts28-19-13-11-8-6-5-4-3-3 \
  --cfg job
```

Formal config: ViT-B/16, N=8, 15 epochs/increment, batch 32, learning rate 0.001, class counts `[28,19,13,11,8,6,5,4,3,3]`, base anchor k=2, graph top-m=5, identity alpha=0.7, lambda=100. The graph top-m budget includes diagonal candidates; the config comment's K_g=4 is not an enforced four-off-diagonal-neighbor implementation.

**Adaptive-k remains enabled by the original code for graph models** when the class count is supplied. With the defaults, the training anchor schedule is `[5,3,2,2,1,1,1,1,1,1]`. This is not a fixed-k control. The existing evaluation call does not supply a per-task class count; evaluation can therefore use base k=2. Verify this behavior against the intended experiment before comparing fixed/adaptive budgets. The launcher intentionally retains it instead of changing the algorithm. The original loop also executes `epochs * batches + 1` updates per task.

## Launch a fresh run

Use a new output directory for every run. Do not reuse an existing directory: metrics would be overwritten and some outputs could be mixed.

```bash
export MOCE_SEED=0
export PYTHONHASHSEED=0
export TQDM_DISABLE=1
export HYDRA_FULL_ERROR=1
# Optional: paired adjacency/gates, capped per stage/task/layer. No image labels/IDs yet.
export MOCE_CAPTURE_SAMPLES=1
RUN_DIR="$PWD/experiments/moce_reproduction_seed0_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
python -m pip freeze > "$RUN_DIR/environment.txt"
nohup python -u reproduce_moce.py \
  --config-path configs/class/09082026_formal_moe_moce \
  --config-name s10-MoCE-N8-cifar100-ts28-19-13-11-8-6-5-4-3-3 \
  "hydra.run.dir=$RUN_DIR" \
  +routing_analysis_enabled=true \
  +routing_analysis_sample_limit=200 \
  > "$RUN_DIR/console.log" 2>&1 < /dev/null &
echo "PID=$! RUN_DIR=$RUN_DIR"
tail -f "$RUN_DIR/console.log"
```

No runtime estimate has been validated. Capture is optional and can increase memory/disk use. Every increment saves a full model state after adaptation, before evaluation, using a temporary file and exact tensor read-back checks. Expect multiple GB across ten checkpoints. CUDA visibility can be set explicitly before launch if needed.

Outputs: `checkpoints/task_00.pt` through `task_09.pt`, `provenance.json`, `metrics.json`, `training_loss.csv`, Hydra config, and routing reports. Optional `paired_stage*_task*_*.npz` files contain aligned `gates [B,N]` and `adjacency [B,N,N]`. They contain labels/IDs only if an evaluation driver explicitly passes those to the collector. Existing training calls do not.

Checkpoints contain all model weights (including graph and task routers), resolved config, class information, text tokens, provenance, and RNG states. They are for later evaluation and stage-level recovery development; no automatic resume or standalone inference command is implemented. Optimizer state is not saved; exact mid-task continuation is not supported. Loading with `weights_only=False` is appropriate only for your own trusted checkpoints.

## Regenerate the configuration/archived-result audit

From `cil/`:

```bash
python moce_analysis.py --root .. --out ../analysis/rebuttal_local
```

Produces `config_audit.csv`, `archived_routing_metrics.csv` when saved routing groups exist, and `audit.json`. A fresh checkout has no archived experiment outputs, so no archived routing CSV is produced until runs exist. The audit does not train a model or generate the final specialization figure.

For paired sample files, use the supplied function in an analysis driver:

```python
import numpy as np
from moce_analysis import sample_metrics
with np.load('path/to/paired_stage9_task0_L0.npz') as data:
    metrics = sample_metrics(data['gates'], data['adjacency'], depth=1)
    print({name: values.mean() for name, values in metrics.items()})
```

Use the configured graph depth. Structural involvement counts nonzero edges, including self paths; effective degree uses row-weight entropy. Neither metric executes neighboring adapters. For task-size analysis, compare matched evaluation stages/layers and hold anchor k/sample count constant or explicitly control for them. Final sample figure generation, metadata wiring, checkpoint evaluation, and the matched-router control remain on the to-do list.
