

import json
import os
import subprocess
import sys
import numpy as np
import torch.nn as nn

from continuum import ClassIncremental, InstanceIncremental
from continuum.datasets import (
    CIFAR100, ImageNet100, TinyImageNet200, ImageFolderDataset, Core50
)
from .utils import get_dataset_class_names
from .uneven_scenario import UnevenClassIncremental


def _ensure_imagenet_subset_assets(workdir: str, subset_size: str) -> None:
    """
    Ensure class order and split files exist for imagenet{subset_size}.
    Auto-generates by invoking scripts/prepare_imagenet_subsets.py when missing.
    """
    cil_dir = os.path.dirname(os.path.dirname(__file__))
    reqs_dir = os.path.join(cil_dir, "dataset_reqs")
    class_order_path = os.path.join(cil_dir, "class_orders", f"imagenet{subset_size}.yaml")
    split_dir = os.path.join(reqs_dir, f"imagenet{subset_size}_splits")
    train_split = os.path.join(split_dir, f"train_{subset_size}.txt")
    val_split = os.path.join(split_dir, f"val_{subset_size}.txt")

    need_class_order = not os.path.exists(class_order_path)
    need_splits = not (os.path.exists(train_split) and os.path.exists(val_split))
    if not (need_class_order or need_splits):
        return

    script_path = os.path.join(cil_dir, "scripts", "prepare_imagenet_subsets.py")
    imagenet_root = os.path.join(workdir, "datasets", "ImageNet")
    train_root = os.path.join(imagenet_root, "train")
    val_root = os.path.join(imagenet_root, "val")
    has_imagenet_data = os.path.isdir(train_root) and os.path.isdir(val_root)

    # Always try to create class_order/classes; create splits only if ImageNet train/val is present.
    cmd = [sys.executable, script_path, "--dataset-root", "../datasets/ImageNet", "--sizes", subset_size]
    if not has_imagenet_data:
        cmd.append("--skip-splits")
    subprocess.run(cmd, cwd=cil_dir, check=False)

    # For runtime training, split files are required.
    if not (os.path.exists(train_split) and os.path.exists(val_split)):
        raise FileNotFoundError(
            f"Missing ImageNet subset split files for imagenet{subset_size}: "
            f"{train_split} / {val_split}. "
            "Expected ImageNet data layout at datasets/ImageNet/train and datasets/ImageNet/val."
        )


class Food101Raw(ImageFolderDataset):
    """Food-101 in raw layout: data_path/images/ and data_path/meta/train.json, test.json.
    Uses meta/classes.txt for class order (index = line number).
    """

    def __init__(self, data_path: str, train: bool = True, download: bool = False):
        self._data_path = data_path
        self.train = train
        # Continuum expects data_path; we set it to images/ so base can find classes, but we override get_data
        images_path = os.path.join(data_path, "images")
        super().__init__(data_path=images_path, train=train, download=download)

    def get_data(self):
        meta_dir = os.path.join(self._data_path, "meta")
        split_file = os.path.join(meta_dir, "train.json" if self.train else "test.json")
        classes_file = os.path.join(meta_dir, "classes.txt")
        with open(classes_file, "r") as f:
            class_names = [line.strip() for line in f if line.strip()]
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        with open(split_file, "r") as f:
            split = json.load(f)
        images_path = os.path.join(self._data_path, "images")
        x_list, y_list = [], []
        for class_name, rel_paths in split.items():
            idx = class_to_idx[class_name]
            for rel in rel_paths:
                full = os.path.join(images_path, rel + ".jpg")
                x_list.append(full)
                y_list.append(idx)
        x = np.array(x_list, dtype=object)
        y = np.array(y_list, dtype=np.int64)
        t = np.full(len(y), -1, dtype=np.int64)
        return x, y, t


class ImageNet1000(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()


class CustomImageFolder(ImageFolderDataset):
    """Generic ImageFolder-style dataset with train/val subfolders.
    Expects: data_path/train/class_name/*.jpg and data_path/val/class_name/*.jpg
    Use this when adding a new dataset (Option B) with folder-per-class layout.
    """

    def __init__(self, data_path: str, train: bool = True, download: bool = False):
        super().__init__(data_path=data_path, train=train, download=download)

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()


def get_dataset(cfg, is_train, transforms=None):
    if cfg.dataset == "cifar100":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = CIFAR100(
            data_path=data_path, 
            download=True, 
            train=is_train, 
            # transforms=transforms
        )
        classes_names = dataset.dataset.classes

    elif cfg.dataset == "tinyimagenet":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = TinyImageNet200(
            data_path, 
            train=is_train,
            download=True
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)
        
    elif cfg.dataset in {"imagenet100", "imagenet200", "imagenet500"}:
        data_path = os.path.join(cfg.dataset_root, "ImageNet")
        # Use relative path from the cil directory
        dataset_reqs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset_reqs")
        subset_size = cfg.dataset.replace("imagenet", "")
        _ensure_imagenet_subset_assets(cfg.workdir, subset_size)
        subset_file = os.path.join(
            dataset_reqs_dir,
            f"imagenet{subset_size}_splits",
            f"{'train' if is_train else 'val'}_{subset_size}.txt",
        )
        dataset = ImageNet100(
            data_path, 
            train=is_train,
            data_subset=subset_file
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.dataset == "imagenet1000":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = ImageNet1000(
            data_path, 
            train=is_train
        )
        classes_names = get_dataset_class_names(cfg.workdir, cfg.dataset)

    elif cfg.dataset == "core50":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)
        dataset = dataset = Core50(
            data_path, 
            scenario="domains", 
            classification="category", 
            train=is_train
        )
        classes_names = [
            "plug adapters", "mobile phones", "scissors", "light bulbs", "cans", 
            "glasses", "balls", "markers", "cups", "remote controls"
        ]

    elif cfg.dataset == "food101":
        # Raw Food-101 layout: dataset_root/food-101 with images/ and meta/train.json, test.json, classes.txt
        data_path = os.path.join(cfg.dataset_root, "food-101")
        dataset = Food101Raw(data_path=data_path, train=is_train, download=False)
        classes_file = os.path.join(data_path, "meta", "classes.txt")
        with open(classes_file, "r") as f:
            classes_names = [line.strip().replace("_", " ") for line in f if line.strip()]

    else:
        raise ValueError(f"'{cfg.dataset}' is an invalid dataset.")

    return dataset, classes_names


def build_cl_scenarios(cfg, is_train, transforms) -> nn.Module:

    dataset, classes_names = get_dataset(cfg, is_train)

    if cfg.scenario == "class":
        if getattr(cfg, "task_sizes", None) is not None:
            scenario = UnevenClassIncremental(
                dataset,
                class_order=cfg.class_order,
                task_sizes=cfg.task_sizes,
                transformations=transforms.transforms,
            )
        else:
            scenario = ClassIncremental(
                dataset,
                initial_increment=cfg.initial_increment,
                increment=cfg.increment,
                transformations=transforms.transforms,  # Convert Compose into list
                class_order=cfg.class_order,
            )

    elif cfg.scenario == "domain":
        scenario = InstanceIncremental(
            dataset,
            transformations=transforms.transforms,
        )

    elif cfg.scenario == "task-agnostic":
        NotImplementedError("Method has not been implemented. Soon be added.")

    else:
        ValueError(f"You have entered `{cfg.scenario}` which is not a defined scenario, " 
                    "please choose from {{'class', 'domain', 'task-agnostic'}}.")

    return scenario, classes_names