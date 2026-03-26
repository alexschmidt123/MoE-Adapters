#!/usr/bin/env python3
"""
Prepare reusable ImageNet subset assets for N in {100, 200, 500}.

Outputs:
- dataset_reqs/imagenet{N}_classes.txt
- class_orders/imagenet{N}.yaml
- dataset_reqs/imagenet{N}_splits/train_{N}.txt
- dataset_reqs/imagenet{N}_splits/val_{N}.txt

``imagenet1000_classes.txt`` may omit the synset column (idx + tab + human name only); then
``dataset_reqs/imagenet1000_synsets.txt`` must list 1000 WordNet IDs in class-index order (0..999).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml


def load_imagenet1000_classes(path: Path, synsets_path: Path) -> list[tuple[int, str, str]]:
    """Load 1000-class table. Supports ``idx\\tsynset\\tname`` or ``idx\\tname`` (synsets from *synsets_path*)."""
    wnids: list[str] | None = None
    if synsets_path.exists():
        wnids = [ln.strip() for ln in synsets_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(wnids) != 1000:
            wnids = None
    rows: list[tuple[int, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 2)
        if len(parts) >= 3:
            idx_str, synset, name = parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            idx_str, name = parts[0], parts[1]
            idx = int(idx_str)
            if wnids is None:
                raise ValueError(
                    f"Two-column format in {path} requires {synsets_path} with exactly 1000 WordNet IDs (one per line)."
                )
            synset = wnids[idx]
        else:
            raise ValueError(f"Unrecognized line in {path}: {line!r}")
        rows.append((int(idx_str), synset, name))
    rows.sort(key=lambda x: x[0])
    return rows


def load_order(path: Path):
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(d["class_order"])


def save_classes(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, (_, synset, name) in enumerate(rows):
            f.write(f"{i}\t{synset}\t{name}\n")


def save_order(order, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump({"class_order": order}, sort_keys=False),
        encoding="utf-8",
    )


def build_split_file(image_root: Path, synset_to_local: dict[str, int], split: str, out_path: Path):
    # Expected layout: ImageNet/train/<synset>/*.JPEG and ImageNet/val/<synset>/*.JPEG
    split_root = image_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Missing split directory: {split_root}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for synset, local_id in synset_to_local.items():
        cls_dir = split_root / synset
        if not cls_dir.exists():
            continue
        for img in sorted(cls_dir.iterdir()):
            if img.is_file():
                rel = f"{split}/{synset}/{img.name}"
                lines.append(f"{rel} {local_id}\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="../datasets/ImageNet",
        help="Path to ImageNet root containing train/ and val/.",
    )
    parser.add_argument(
        "--sizes",
        default="100,200,500",
        help="Comma-separated subset sizes to prepare.",
    )
    parser.add_argument(
        "--skip-splits",
        action="store_true",
        help="Only prepare class files and class_orders; skip train/val split txt generation.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    cil_dir = script_dir.parent
    reqs_dir = cil_dir / "dataset_reqs"
    orders_dir = cil_dir / "class_orders"
    image_root = (cil_dir / args.dataset_root).resolve()

    classes1000 = load_imagenet1000_classes(
        reqs_dir / "imagenet1000_classes.txt",
        reqs_dir / "imagenet1000_synsets.txt",
    )
    order1000 = load_order(orders_dir / "imagenet1000.yaml")

    for n in [int(x) for x in args.sizes.split(",") if x.strip()]:
        if n <= 0 or n > 1000:
            raise ValueError(f"Invalid subset size: {n}")

        subset_rows = classes1000[:n]
        save_classes(subset_rows, reqs_dir / f"imagenet{n}_classes.txt")

        # Keep shuffled style from imagenet1000 order, mapped to local [0, n).
        order_n = [x for x in order1000 if x < n]
        if len(order_n) != n:
            raise ValueError(f"Could not derive order of length {n}")
        save_order(order_n, orders_dir / f"imagenet{n}.yaml")

        synset_to_local = {synset: i for i, (_, synset, _) in enumerate(subset_rows)}
        if args.skip_splits:
            print(f"Prepared ImageNet{n}: classes, class_order (splits skipped).")
            continue

        split_dir = reqs_dir / f"imagenet{n}_splits"
        build_split_file(image_root, synset_to_local, "train", split_dir / f"train_{n}.txt")
        build_split_file(image_root, synset_to_local, "val", split_dir / f"val_{n}.txt")

        print(f"Prepared ImageNet{n}: classes, class_order, train/val splits.")


if __name__ == "__main__":
    main()

