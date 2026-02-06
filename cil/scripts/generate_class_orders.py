#!/usr/bin/env python3
"""
Generate multiple class_order YAML files with different random seeds.
Use these to increase task difficulty variability and evaluate robustness
(mean ± std across orders).

Usage (from cil/):
  python scripts/generate_class_orders.py --dataset food101 --num_classes 101 --seeds 42 123 456 --out_dir class_orders
  python scripts/generate_class_orders.py --dataset cifar100 --num_classes 100 --seeds 0 1 2 3 4 --out_dir class_orders
"""

import argparse
import os
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate class_order YAMLs with different seeds")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. food101, cifar100)")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds for permutations")
    parser.add_argument("--out_dir", type=str, default="class_orders", help="Output directory for YAML files")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    indices = list(range(args.num_classes))
    for seed in args.seeds:
        rng = random.Random(seed)
        order = indices.copy()
        rng.shuffle(order)
        filename = f"{args.dataset}_seed{seed}.yaml"
        filepath = out_path / filename
        with open(filepath, "w") as f:
            f.write(f"# {args.dataset} class order (seed={seed})\n")
            f.write("class_order: " + str(order) + "\n")
        print(f"Wrote {filepath}")

    print(f"Generated {len(args.seeds)} class order(s). Run with: class_order={out_path}/{args.dataset}_seed<SEED>.yaml")


if __name__ == "__main__":
    main()
