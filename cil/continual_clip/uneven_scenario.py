"""
Uneven class-incremental scenario: split classes by task_sizes (list) instead of fixed increment.
Use so the MoE-GNN can see different task sizes and allocate more expert capacity to heavier tasks.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image


def _load_image(path):
    if isinstance(path, (str, bytes)) or (hasattr(path, "decode") and callable(path.decode)):
        path = path.decode() if hasattr(path, "decode") else str(path)
        img = Image.open(path).convert("RGB")
        return np.array(img)
    return path


class _TaskSubset(Dataset):
    """Single-task dataset: (x, y, t) filtered by task_id, with transform applied in __getitem__.
    Returns task-local labels in [0, num_classes_in_task) for the model."""

    def __init__(self, x, y, t, task_id, indices, transforms, task_class_ids):
        self.x = x
        self.y = y
        self.t = t
        self.task_id = task_id
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transforms = transforms
        # Map global class id -> local index 0..len(task_class_ids)-1 for this task
        self._class_to_local = {c: i for i, c in enumerate(task_class_ids)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        xi = self.x[real_idx]
        yi_global = int(self.y[real_idx])
        yi = self._class_to_local[yi_global]  # task-local 0-indexed for current task
        ti = self.t[real_idx]
        if self.transforms is not None:
            img = _load_image(xi)
            # Torchvision transforms (e.g. RandomResizedCrop) expect PIL; CIFAR-100 get_data() can return numpy
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img.astype(np.uint8))
            if hasattr(self.transforms, "__iter__") and not callable(self.transforms):
                for tr in self.transforms:
                    img = tr(img)
            else:
                img = self.transforms(img)
            return img, yi, ti
        return xi, yi, ti


class UnevenClassIncremental:
    """
    Scenario with uneven task sizes. Same interface as continuum Scenario: len(scenario), scenario[i], scenario[i:j].
    task_sizes: list of ints, e.g. [5, 15, 8, 12, 10, 8, 15, 10, 10, 7] for 10 tasks summing to 100.
    Each task must have at least 2 classes (min(task_sizes) >= 2).
    """

    def __init__(self, dataset, class_order, task_sizes, transformations):
        self.dataset = dataset
        self.class_order = list(class_order)
        self.task_sizes = list(task_sizes)
        self.transformations = transformations
        if sum(self.task_sizes) != len(self.class_order):
            raise ValueError(
                "task_sizes must sum to len(class_order): "
                f"sum(task_sizes)={sum(self.task_sizes)} != len(class_order)={len(self.class_order)}"
            )
        if min(self.task_sizes) < 2:
            raise ValueError(
                f"Each task must have at least 2 classes; got task_sizes={self.task_sizes}. "
                "Single-class tasks are not allowed."
            )
        x, y, t_old = dataset.get_data()
        # Map class_id -> task_id: class_order[pos] belongs to task_id where pos in [start, start+size)
        class_to_task = {}
        start = 0
        for task_id, size in enumerate(self.task_sizes):
            for j in range(size):
                class_to_task[self.class_order[start + j]] = task_id
            start += size
        t_new = np.array([class_to_task[int(yi)] for yi in y], dtype=np.int64)
        self._x = x
        self._y = y
        self._t = t_new
        self._n_tasks = len(self.task_sizes)
        self._task_indices = [np.where(self._t == i)[0] for i in range(self._n_tasks)]
        # Class ids (global) per task for remapping to 0-indexed labels
        self._task_class_ids = []
        start = 0
        for size in self.task_sizes:
            self._task_class_ids.append(list(self.class_order[start : start + size]))
            start += size

    def __len__(self):
        return self._n_tasks

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._n_tasks)
            subsets = [
                _TaskSubset(
                    self._x, self._y, self._t, i,
                    self._task_indices[i], self.transformations,
                    self._task_class_ids[i],
                )
                for i in range(start, stop, step or 1)
            ]
            return ConcatDataset(subsets)
        return _TaskSubset(
            self._x, self._y, self._t, index,
            self._task_indices[index], self.transformations,
            self._task_class_ids[index],
        )
