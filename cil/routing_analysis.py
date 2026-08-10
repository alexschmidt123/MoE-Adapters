"""Aggregate and visualize MoE/GoE routing behavior during evaluation."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


class RoutingAnalyzer:
    """Streaming routing statistics, grouped by task and visual transformer layer."""

    def __init__(self, output_dir, save_sample_limit=2000):
        self.output_dir = Path(output_dir)
        self.save_sample_limit = int(save_sample_limit)
        self.stats = {}
        self.samples = defaultdict(list)

    def collect(self, clip_model, task_id, evaluation_stage=None):
        """Consume transient diagnostics after one homogeneous-task forward."""
        task_id = int(task_id)
        evaluation_stage = task_id if evaluation_stage is None else int(evaluation_stage)
        for name, module in clip_model.named_modules():
            diag = getattr(module, "_last_routing_diagnostics", None)
            if diag is None or getattr(module, "text_or_image", None) != "image":
                continue
            gates = diag["gates"].float().cpu()
            weights = diag["routing_weights"].float().cpu()
            key = (evaluation_stage, task_id, name)
            if key not in self.stats:
                n = gates.shape[1]
                self.stats[key] = {
                    "samples": 0, "batches": 0, "top_k_sum": 0,
                    "selections": torch.zeros(n, dtype=torch.float64),
                    "sparse_weight": torch.zeros(n, dtype=torch.float64),
                    "dense_weight": torch.zeros(n, dtype=torch.float64),
                    "entropy_sum": 0.0, "adjacency": torch.zeros(n, n, dtype=torch.float64),
                    "adjacency_samples": 0,
                }
            s = self.stats[key]
            batch = gates.shape[0]
            s["samples"] += batch
            s["batches"] += 1
            s["top_k_sum"] += int(diag["effective_top_k"]) * batch
            s["selections"] += (gates > 0).sum(0).double()
            s["sparse_weight"] += gates.sum(0).double()
            s["dense_weight"] += weights.sum(0).double()
            s["entropy_sum"] += float((-(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(1)).sum())

            graph = getattr(module, "graph_mixer", None)
            adjacency = getattr(graph, "_last_adjacency", None) if graph is not None else None
            if adjacency is not None:
                adjacency = adjacency.float().cpu()
                s["adjacency"] += adjacency.sum(0).double()
                s["adjacency_samples"] += adjacency.shape[0]
                graph._last_adjacency = None

            remaining = self.save_sample_limit - sum(x.shape[0] for x in self.samples[key])
            if remaining > 0:
                self.samples[key].append(gates[:remaining].clone())
            module._last_routing_diagnostics = None

    @staticmethod
    def _short_layer(name):
        marker = "visual.transformer.resblocks."
        return "L" + name.split(marker, 1)[1].split(".", 1)[0] if marker in name else name

    def export(self):
        """Write CSV/NPZ/JSON data and four routing figures."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows, adjacency_rows = [], []
        arrays = {}
        for (stage, task, layer), s in sorted(self.stats.items()):
            count = max(s["samples"], 1)
            layer_short = self._short_layer(layer)
            for expert in range(len(s["selections"])):
                rows.append({
                    "evaluation_stage": stage, "task": task, "layer": layer_short, "expert": expert,
                    "samples": s["samples"],
                    "selection_count": int(s["selections"][expert]),
                    "selection_rate": float(s["selections"][expert] / count),
                    "mean_sparse_gate": float(s["sparse_weight"][expert] / count),
                    "mean_dense_routing_weight": float(s["dense_weight"][expert] / count),
                    "mean_routing_entropy": s["entropy_sum"] / count,
                    "effective_top_k": s["top_k_sum"] / count,
                })
            if s["adjacency_samples"]:
                adj = s["adjacency"] / s["adjacency_samples"]
                arrays[f"adjacency_stage{stage}_task{task}_{layer_short}"] = adj.numpy()
                for source in range(adj.shape[0]):
                    for target in range(adj.shape[1]):
                        adjacency_rows.append({"evaluation_stage": stage, "task": task, "layer": layer_short,
                                               "source_expert": source, "target_expert": target,
                                               "mean_edge_weight": float(adj[source, target])})
            sample_parts = self.samples.get((stage, task, layer), [])
            if sample_parts:
                arrays[f"sample_gates_stage{stage}_task{task}_{layer_short}"] = torch.cat(sample_parts).numpy()

        self._write_csv(self.output_dir / "expert_routing_summary.csv", rows)
        self._write_csv(self.output_dir / "graph_neighborhoods.csv", adjacency_rows)
        np.savez_compressed(self.output_dir / "routing_tensors.npz", **arrays)
        metadata = {"evaluation_stages": sorted({k[0] for k in self.stats}),
                    "tasks": sorted({k[1] for k in self.stats}),
                    "layers": sorted({self._short_layer(k[2]) for k in self.stats}),
                    "saved_sample_limit_per_task_layer": self.save_sample_limit}
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        if rows:
            self._plot(rows, arrays)
        return self.output_dir

    @staticmethod
    def _write_csv(path, rows):
        if not rows:
            path.write_text("")
            return
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _plot(self, rows, arrays):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Main figures describe the final learned model. Earlier stages remain
        # in CSV/NPZ for routing-drift analysis.
        final_stage = max(r["evaluation_stage"] for r in rows)
        rows = [r for r in rows if r["evaluation_stage"] == final_stage]
        tasks = sorted({r["task"] for r in rows})
        layers = sorted({r["layer"] for r in rows}, key=lambda x: int(x[1:]) if x[1:].isdigit() else 10**9)
        experts = sorted({r["expert"] for r in rows})
        lookup = {(r["task"], r["layer"], r["expert"]): r for r in rows}

        # Heatmaps make layer/task specialization and collapse immediately visible.
        for field, filename, title in [
            ("selection_rate", "expert_selection.png", "Expert selection rate"),
            ("mean_dense_routing_weight", "routing_weights.png", "Mean dense routing weight"),
        ]:
            matrix = np.array([[np.mean([lookup[(t, l, e)][field] for l in layers])
                                for e in experts] for t in tasks])
            fig, ax = plt.subplots(figsize=(max(6, len(experts) * .55), max(3, len(tasks) * .45)))
            im = ax.imshow(matrix, aspect="auto", cmap="viridis")
            ax.set(xlabel="Expert", ylabel="Task", title=title,
                   xticks=range(len(experts)), xticklabels=experts,
                   yticks=range(len(tasks)), yticklabels=tasks)
            fig.colorbar(im, ax=ax)
            fig.tight_layout(); fig.savefig(self.output_dir / filename, dpi=200); plt.close(fig)

        # Utilization aggregated across tasks, one curve per transformer layer.
        fig, ax = plt.subplots(figsize=(max(7, len(experts) * .6), 4.5))
        for layer in layers:
            values = [np.mean([lookup[(t, layer, e)]["selection_rate"] for t in tasks]) for e in experts]
            ax.plot(experts, values, marker="o", linewidth=1, label=layer)
        ax.set(xlabel="Expert", ylabel="Selection rate", title="Expert utilization by layer")
        ax.legend(ncol=min(4, len(layers)), fontsize=7); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(self.output_dir / "expert_utilization.png", dpi=200); plt.close(fig)

        # Neighborhood view: average adjacency over all recorded tasks/layers.
        adjs = [v for k, v in arrays.items() if k.startswith(f"adjacency_stage{final_stage}_")]
        if adjs:
            adj = np.mean(adjs, axis=0); n = adj.shape[0]
            theta = np.linspace(0, 2 * math.pi, n, endpoint=False)
            xy = np.c_[np.cos(theta), np.sin(theta)]
            fig, ax = plt.subplots(figsize=(7, 7))
            threshold = np.quantile(adj[~np.eye(n, dtype=bool)], .75) if n > 1 else 0
            for i in range(n):
                for j in range(n):
                    if i != j and adj[i, j] >= threshold:
                        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                                color="tab:blue", alpha=min(.9, .15 + float(adj[i, j]) * 2),
                                linewidth=.5 + float(adj[i, j]) * 4)
            ax.scatter(xy[:, 0], xy[:, 1], s=500, c=np.diag(adj), cmap="plasma", zorder=3)
            for i, (x, y) in enumerate(xy): ax.text(x, y, str(i), ha="center", va="center", zorder=4)
            ax.set(title="Graph expert neighborhoods (top quartile edges)", aspect="equal")
            ax.axis("off"); fig.tight_layout(); fig.savefig(self.output_dir / "graph_neighborhoods.png", dpi=200); plt.close(fig)
