#!/usr/bin/env python3
"""Generate summary.csv from a run directory (experiments/RUN_START). Use after run_test.sh."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def read_last_avg(metrics_path: Path):
    try:
        text = metrics_path.read_text(encoding="utf-8", errors="replace")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None, None
        last_line = json.loads(lines[-1])
        if "last" in last_line and "avg" in last_line:
            return float(last_line["last"]), float(last_line["avg"])
        return None, None
    except Exception:
        return None, None


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_run_summary.py <results_dir>")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}")
        sys.exit(1)
    rows = []
    for exp_dir in sorted(run_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        metrics_path = exp_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        # Dir name: 02052026_baseline-run1-123456 → config_name, run_id (flat names)
        name = exp_dir.name
        if "-run" not in name:
            continue
        parts = name.split("-run", 1)
        config_name = parts[0]
        run_id_str = parts[1].split("-")[0]
        try:
            run_id = int(run_id_str)
        except ValueError:
            run_id = run_id_str
        last_acc, avg_acc = read_last_avg(metrics_path)
        rows.append({
            "config_name": config_name,
            "run_id": run_id,
            "last_acc": "" if last_acc is None else f"{last_acc:.2f}",
            "avg_acc": "" if avg_acc is None else f"{avg_acc:.2f}",
        })
    by_config = defaultdict(list)
    for r in rows:
        if r["last_acc"] and r["avg_acc"]:
            by_config[r["config_name"]].append((float(r["last_acc"]), float(r["avg_acc"])))
    for config_name in sorted(by_config.keys()):
        vals = by_config[config_name]
        mean_last = sum(x[0] for x in vals) / len(vals)
        mean_avg = sum(x[1] for x in vals) / len(vals)
        rows.append({
            "config_name": config_name,
            "run_id": "avg",
            "last_acc": f"{mean_last:.2f}",
            "avg_acc": f"{mean_avg:.2f}",
        })
    out_path = run_dir / "summary.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config_name", "run_id", "last_acc", "avg_acc"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
