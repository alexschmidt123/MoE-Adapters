# Replacing CIFAR-100 with Another Dataset

Yes, you can replace CIFAR-100 with another dataset. The CIL pipeline is dataset-agnostic once the right config and data files are in place.

---

## Option A: Use an already-supported dataset

The codebase already supports:

| Dataset        | Config value      | Class order file           | Data path (under `dataset_root`) |
|----------------|-------------------|----------------------------|-----------------------------------|
| CIFAR-100      | `cifar100`        | `class_orders/cifar100.yaml` | `cifar100/`                      |
| TinyImageNet   | `tinyimagenet`    | `class_orders/tinyimagenet.yaml` | `tinyimagenet/`               |
| ImageNet-100   | `imagenet100`     | `class_orders/imagenet100.yaml` | `ImageNet/` (with split files) |
| ImageNet-1000  | `imagenet1000`    | `class_orders/imagenet1000.yaml` | `imagenet1000/` (train/, val/) |
| CORe50         | `core50`          | (built-in)                 | `core50/`                        |

### Steps to switch (e.g. to TinyImageNet)

1. **Config:** In your YAML (or Hydra overrides), set:
   - `dataset: "tinyimagenet"`
   - `class_order: "class_orders/tinyimagenet.yaml"`  (path relative to `workdir`, usually the `cil/` directory)
   - `dataset_root: "/path/to/your/datasets"`  (or leave empty and pass via CLI)

2. **Data:** Put the dataset under `dataset_root/tinyimagenet/` in the format expected by Continuum’s `TinyImageNet200` (see Continuum docs or existing usage).

3. **Class names:** Ensure `dataset_reqs/tinyimagenet_classes.txt` exists (format: `index\tid\tclass_name` per line). It is already in the repo.

4. **Run:** Same as CIFAR-100; pass `dataset_root` and `class_order` if you override them:
   ```bash
   python main.py --config-name your_config dataset_root=/path/to/data class_order=class_orders/tinyimagenet.yaml dataset=tinyimagenet
   ```

For **imagenet100**, data lives under `dataset_root/ImageNet/` and the split files in `dataset_reqs/imagenet100_splits/` are used automatically.

---

## Option B: Add a new dataset (not in the list above)

You need to wire the new dataset into the loader and provide class order + class names.

### 1. Add a branch in `continual_clip/datasets.py`

In `get_dataset(cfg, is_train, transforms=None)` add an `elif cfg.dataset == "your_dataset":` block that:

- Builds a Continuum-compatible dataset (or one that implements the same interface) from `data_path`.
- Sets `classes_names` (list of class names in **dataset order**, i.e. index 0, 1, 2, …).

Example pattern:

```python
elif cfg.dataset == "your_dataset":
    data_path = os.path.join(cfg.dataset_root, "your_dataset")
    dataset = YourContinuumDataset(data_path, train=is_train, ...)
    classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
```

If your dataset class exposes class names (like CIFAR-100), you can use those instead of `get_dataset_class_names`.

### 2. Create `class_orders/your_dataset.yaml`

Same format as `cifar100.yaml`: a permutation of class indices that defines the order of classes in incremental learning.

```yaml
class_order: [0, 5, 2, 9, ...]   # length = num_classes
```

You can use a fixed order or a random permutation for experiments.

### 3. Create `dataset_reqs/your_dataset_classes.txt`

Required if you use `get_dataset_class_names(cfg.workdir, cfg.dataset)`. One line per class:

```
0	class_id_0	class name 0
1	class_id_1	class name 1
...
```

The code uses the **last column** (after the final tab) as the class name. The first column is the index (0, 1, 2, …) used in `class_order`.

### 4. Config and run

In your config YAML:

```yaml
dataset: "your_dataset"
class_order: "class_orders/your_dataset.yaml"
dataset_root: ""
```

Then run as usual, overriding `dataset_root` (and optionally `class_order`) from the command line if needed.

---

## Summary

| Goal                         | What to do |
|-----------------------------|------------|
| Use TinyImageNet / ImageNet-100 / etc. | Set `dataset`, `class_order`, and `dataset_root`; ensure data and `dataset_reqs/*_classes.txt` (and splits for imagenet100) exist. |
| Add a new dataset           | Add a branch in `get_dataset()`, add `class_orders/your_dataset.yaml` and `dataset_reqs/your_dataset_classes.txt`, then set `dataset` and `class_order` in config. |

The pipeline (scenario, increments, evaluation) is the same for any dataset; only the config and the three pieces above (loader branch, class order YAML, class names file) are dataset-specific.
