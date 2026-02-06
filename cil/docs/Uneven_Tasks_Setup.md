# Uneven Task Split for CIFAR-100

Split 100 classes unevenly into 10 tasks. MoE-GNN can learn task size and allocate more experts to heavier tasks.

## Step 1: utils.py

In `cil/continual_clip/utils.py`, at the start of `get_class_ids_per_task(args)` add:

```python
if getattr(args, "task_sizes", None) is not None:
    start = 0
    for n in args.task_sizes:
        yield args.class_order[start:start + n]
        start += n
    return
```

Then keep the existing `yield args.class_order[:args.initial_increment]` and the `for i in range(...)` loop.

## Step 2: datasets.py

In `build_cl_scenarios`, when `cfg.scenario == "class"` and `getattr(cfg, "task_sizes", None) is not None`, use UnevenClassIncremental instead of ClassIncremental. You need a new file `continual_clip/uneven_scenario.py` that defines UnevenClassIncremental: same interface as continuum Scenario (len, scenario[i], scenario[i:j]), built from dataset.get_data(), class_order, and task_sizes (list of sizes per task, sum = len(class_order)).

## Step 3: Config

Create a config with `task_sizes: [5, 15, 8, 12, 10, 8, 15, 10, 10, 7]` (sum 100). Example: `cifar100_uneven10-MoE-Adapters-N4-GoE.yaml` with defaults from cifar100_2-2-MoE-Adapters-N2 and overrides dataset, class_order, dataset_root, task_sizes.

## Step 4: Run

From cil: `python main.py --config-path configs/class --config-name cifar100_uneven10-MoE-Adapters-N4-GoE dataset_root=../datasets`

## Step 5: Model

Use `len(class_ids_per_task[task_id])` in the model to get number of classes per task and scale expert allocation (e.g. more capacity for heavier tasks).
