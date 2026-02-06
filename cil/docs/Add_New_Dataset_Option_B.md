# Option B: Add a New Dataset (Step-by-Step)

Follow these steps to add a new dataset to the CIL pipeline. Replace `MY_DATASET` with your dataset name (e.g. `flowers102`, `food101`).

---

## Step 1: Decide data layout

**Option 1 — Folder-per-class (easiest)**  
If your data is organized as:

```
dataset_root/MY_DATASET/
  train/
    class_0/   img1.jpg, img2.jpg, ...
    class_1/   ...
  val/
    class_0/   ...
    class_1/   ...
```

you can use Continuum’s **ImageFolderDataset** (same idea as ImageNet1000). Class names are the folder names; indices are the order in which folders are read (alphabetical unless you control it). You still need a **class order** YAML that defines the incremental order (see Step 3).

**Option 2 — Custom loader**  
If your data is in another format (e.g. one folder + CSV, or a PyTorch Dataset), you need a Continuum-compatible dataset that implements `get_data()` returning `(x, y, t)` (numpy arrays: images, labels, task ids). Then you wire it in Step 2.

---

## Step 2: Add a branch in `continual_clip/datasets.py`

### 2a. If using folder-per-class (train/val with class subfolders)

At the top of the file, `ImageNet1000` already subclasses `ImageFolderDataset`. You can add a similar class for your dataset, or reuse `ImageFolderDataset` with the right path. Example for a dataset that has `train/` and `val/` under `dataset_root/MY_DATASET/`:

```python
# In get_dataset(), add BEFORE the "else:" branch:

elif cfg.dataset == "MY_DATASET":
    data_path = os.path.join(cfg.dataset_root, "MY_DATASET")
    dataset = CustomImageFolder(data_path=data_path, train=is_train, download=False)
    classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
```

You **must** have `dataset_reqs/MY_DATASET_classes.txt` so that `get_dataset_class_names` returns names in **index order** 0, 1, 2, … (see Step 3). Folder order might not match that order, so the _classes.txt file defines the mapping.

### 2b. If using a custom Continuum dataset class

If you have a custom class `MyContinuumDataset` that takes `data_path` and `train` and returns the same interface as other Continuum datasets:

```python
elif cfg.dataset == "MY_DATASET":
    data_path = os.path.join(cfg.dataset_root, "MY_DATASET")
    dataset = MyContinuumDataset(data_path=data_path, train=is_train)
    classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
```

Again, `dataset_reqs/MY_DATASET_classes.txt` must exist and list class names in **index order** (0, 1, 2, …).

### 2c. If your dataset object already has class names

If your dataset exposes a list of class names in index order (e.g. `dataset.classes`):

```python
elif cfg.dataset == "MY_DATASET":
    data_path = os.path.join(cfg.dataset_root, "MY_DATASET")
    dataset = MyContinuumDataset(data_path=data_path, train=is_train)
    classes_names = dataset.classes   # or dataset.dataset.classes, depending on your class
```

Then you can skip creating `MY_DATASET_classes.txt` (but you still need the class order YAML).

---

## Step 3: Create `dataset_reqs/MY_DATASET_classes.txt`

Required if you use `get_dataset_class_names(cfg.workdir, cfg.dataset)` in Step 2.

- **Path:** `cil/dataset_reqs/MY_DATASET_classes.txt`
- **Format:** One line per class: `index\tid_or_extra\tclass_name`
- The code uses the **last column** (after the last tab) as the class name. The **first column** must be the class index 0, 1, 2, … (used in `class_order`).

Example for 5 classes:

```
0	0	apple
1	1	banana
2	2	car
3	3	dog
4	4	elephant
```

Or with an extra id (like ImageNet):

```
0	n01440764	tench
1	n01443537	goldfish
...
```

Ensure the **first column** is 0, 1, 2, … in order.

---

## Step 4: Create `class_orders/MY_DATASET.yaml`

- **Path:** `cil/class_orders/MY_DATASET.yaml`
- **Content:** A permutation of class indices defining the order in which classes are presented in incremental learning.

Example for 5 classes (order: 0, 1, 2, 3, 4):

```yaml
class_order: [0, 1, 2, 3, 4]
```

Example random order:

```yaml
class_order: [2, 0, 4, 1, 3]
```

The list length must equal the number of classes. You can use a fixed order or generate a random permutation for experiments.

---

## Step 5: Config and run

In your config YAML (e.g. under `configs/class/`), set:

```yaml
dataset: "MY_DATASET"
class_order: "class_orders/MY_DATASET.yaml"
dataset_root: ""   # override from CLI: dataset_root=/path/to/your/data
# Keep the rest (scenario, initial_increment, increment, etc.) as in cifar100 configs.
```

Run from the **cil** directory:

```bash
python main.py --config-path configs/class --config-name your_config \
  dataset=MY_DATASET \
  class_order=class_orders/MY_DATASET.yaml \
  dataset_root=/path/to/your/datasets
```

If you use a run script (e.g. `run_test.py`), set the script’s `CLASS_ORDER` and `DATASET_ROOT` (or pass the same overrides) so that `class_order` points to `class_orders/MY_DATASET.yaml` and `dataset_root` points to the parent folder that contains the `MY_DATASET` directory.

---

## Checklist

| Step | Action |
|------|--------|
| 1 | Put data under `dataset_root/MY_DATASET/` (e.g. train/val with class subfolders, or as required by your loader). |
| 2 | In `continual_clip/datasets.py`, add `elif cfg.dataset == "MY_DATASET":` with the right dataset class and `classes_names`. |
| 3 | Create `dataset_reqs/MY_DATASET_classes.txt` (one line per class: index, then class name as last column). |
| 4 | Create `class_orders/MY_DATASET.yaml` with `class_order: [0, 1, ...]` (length = num_classes). |
| 5 | Set config `dataset: "MY_DATASET"`, `class_order: "class_orders/MY_DATASET.yaml"`, and pass `dataset_root` at run time. |

---

## Minimal code snippet (copy-paste and edit)

Add this in `continual_clip/datasets.py` in `get_dataset()`, **before** the `else:` that raises `ValueError`:

```python
elif cfg.dataset == "MY_DATASET":
    data_path = os.path.join(cfg.dataset_root, "MY_DATASET")
    dataset = CustomImageFolder(data_path=data_path, train=is_train, download=False)
    classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
```

Then create the two files:

- `dataset_reqs/MY_DATASET_classes.txt`
- `class_orders/MY_DATASET.yaml`

and use `dataset=MY_DATASET` and `class_order=class_orders/MY_DATASET.yaml` in your config/CLI.
th.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()
```

**2) Add the branch** in `get_dataset()`, **before** the `else:`:

```python
elif cfg.dataset == "MY_DATASET":
    data_path = os.path.join(cfg.dataset_root, "MY_DATASET")
    dataset = CustomImageFolder(data_path=data_path, train=is_train, download=False)
    classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
```

**3) Create:**

- `dataset_reqs/MY_DATASET_classes.txt` (one line per class: `index\tid\tclass_name`)
- `class_orders/MY_DATASET.yaml` (`class_order: [0, 1, ...]`)

**4) Run:** `dataset=MY_DATASET` and `class_order=class_orders/MY_DATASET.yaml` in config/CLI.

---

## Food101 (already wired)

Food101 is supported in CIL with:

- **Dataset branch:** `continual_clip/datasets.py` has `elif cfg.dataset == "food101"` using `CustomImageFolder`.
- **Data:** Place Food101 under `dataset_root/food101/` with `train/` and `val/` subfolders, each with one folder per class (e.g. `food101/train/class_name/*.jpg`). Ensure `dataset_reqs/food101_classes.txt` exists (one line per class: `index\tid\tname`).
- **Class order:** `class_orders/food101.yaml` (permutation of 0–100, seed 42).
- **Example config:** `configs/class/food101_11-10-MoE-Adapters-N4-GoE.yaml` (11 initial + 10 per task).

Run from `cil`:

```bash
python main.py --config-path configs/class --config-name food101_11-10-MoE-Adapters-N4-GoE \
  dataset_root=/path/to/datasets class_order=class_orders/food101.yaml
```
