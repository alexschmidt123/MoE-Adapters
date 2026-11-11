"""Evidence audit for MoCE. Run with --root /path/to/MoE-Adapters --out DIR.

No training, checkpoint loading, or source mutation. Sample metrics require paired
gates [B,N] and adjacency [B,N,N]; averaged adjacency is never substituted.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml


def sample_metrics(gates, adjacency, depth=1):
    gates, adjacency = np.asarray(gates), np.asarray(adjacency)
    if gates.ndim != 2 or adjacency.shape != (len(gates), gates.shape[1], gates.shape[1]):
        raise ValueError('Expected paired gates [B,N] and adjacency [B,N,N]')
    if depth < 1 or not np.isfinite(adjacency).all() or (adjacency < 0).any():
        raise ValueError('Invalid graph depth or adjacency')
    active = gates > 0
    involved = active.copy()
    for _ in range(depth):
        involved |= np.einsum('bi,bij->bj', involved.astype(int), (adjacency > 0).astype(int)) > 0
    p = adjacency / np.maximum(adjacency.sum(-1, keepdims=True), 1e-30)
    entropy = -(p * np.log(np.maximum(p, 1e-30))).sum(-1)
    return dict(executed=active.sum(-1), involved=involved.sum(-1),
                mean_node_effective_degree=np.exp(entropy).mean(-1))


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def audit(root, out):
    out.mkdir(parents=True, exist_ok=True)
    configs = []
    for path in sorted((root / 'cil/configs').rglob('*.yaml')):
        try:
            c = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        if not isinstance(c, dict) or not isinstance(c.get('model'), dict):
            continue
        m = c['model']
        if not m.get('graph_mixer_enabled'):
            continue
        n, budget = m.get('num_experts', 2), m.get('graph_adj_top_m')
        configs.append(dict(config=str(path.relative_to(root)), experts=n,
                            graph_adj_top_m=budget, dense=budget is None or budget >= n,
                            anchor_k=m.get('top_k', 2),
                            identity_alpha=m.get('graph_identity_bias_alpha', 0),
                            residual_alpha=m.get('goe_residual_alpha', 0),
                            graph_layers=m.get('graph_num_layers', 2)))
    write_csv(out / 'config_audit.csv', configs)
    rows = []
    for path in sorted((root / 'cil/experiments').rglob('routing_tensors.npz')):
        with np.load(path, allow_pickle=False) as arrays:
            for key in arrays.files:
                if not key.startswith('sample_gates_'):
                    continue
                gates = arrays[key]
                counts = (gates > 0).sum(0)
                p = counts / max(counts.sum(), 1)
                ent = float(-(p * np.log(np.maximum(p, 1e-30))).sum())
                rows.append(dict(artifact=str(path.relative_to(root)), group=key,
                                 samples=len(gates), experts=gates.shape[-1],
                                 distinct_executed_experts=int((counts > 0).sum()),
                                 mean_executed=float((gates > 0).sum(-1).mean()),
                                 selection_entropy=ent, effective_expert_usage=float(np.exp(ent)),
                                 sample_neighbor_involvement='unavailable: adjacency saved as group mean'))
    write_csv(out / 'archived_routing_metrics.csv', rows)
    report = dict(configs=len(configs), dense_configs=sum(r['dense'] for r in configs),
                  archived_groups=len(rows), checkpoints=[str(p.relative_to(root))
                  for ext in ('*.pth', '*.pt', '*.ckpt') for p in root.rglob(ext)],
                  limitations=['Archived adjacency is averaged; per-sample unions cannot be recovered.',
                               'One-epoch and smoke runs are exploratory, not formal rebuttal evidence.',
                               'CIFAR-100 has no airplane class; use actual fine-label names.',
                               'Top-m includes diagonal candidates; identity mixing may add a self edge.',
                               'Positive dense adjacency gives full structural involvement regardless of edge strength.',
                               'Graph involvement concerns learned node states, not execution of neighboring adapters.'])
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.out)
