"""Optional evaluation analyzer preserving paired sample graphs and gates.

Replace RoutingAnalyzer with SampleGraphAnalyzer in an evaluation driver.
Pass labels and stable dataset indices to collect() when available. The existing
three-argument call also works but cannot identify original images.
"""
from collections import defaultdict
import numpy as np
import torch
from routing_analysis import RoutingAnalyzer


class SampleGraphAnalyzer(RoutingAnalyzer):
    def __init__(self, output_dir, save_sample_limit=2000):
        super().__init__(output_dir, save_sample_limit)
        self.paired = defaultdict(list)
        self.paired_count = defaultdict(int)

    def collect(self, clip_model, task_id, evaluation_stage=None, labels=None, sample_ids=None):
        stage = int(task_id if evaluation_stage is None else evaluation_stage)
        for name, module in clip_model.named_modules():
            diag = getattr(module, '_last_routing_diagnostics', None)
            graph = getattr(module, 'graph_mixer', None)
            adjacency = getattr(graph, '_last_adjacency', None)
            if diag is None or adjacency is None or getattr(module, 'text_or_image', None) != 'image':
                continue
            key = (stage, int(task_id), self._short_layer(name))
            take = min(len(adjacency), self.save_sample_limit - self.paired_count[key])
            if take <= 0:
                continue
            record = dict(gates=diag['gates'][:take].detach().cpu().numpy().copy(),
                          adjacency=adjacency[:take].detach().cpu().numpy().copy())
            for field, value in [('labels', labels), ('sample_ids', sample_ids)]:
                if value is not None:
                    a = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
                    if len(a) != len(adjacency):
                        raise ValueError(field + ' must align with the batch')
                    record[field] = a[:take].copy()
            self.paired[key].append(record)
            self.paired_count[key] += take
        super().collect(clip_model, task_id, evaluation_stage)

    def export(self):
        result = super().export()
        for (stage, task, layer), records in self.paired.items():
            fields = set.intersection(*(set(r) for r in records))
            arrays = {field: np.concatenate([r[field] for r in records]) for field in fields}
            np.savez_compressed(self.output_dir / f'paired_stage{stage}_task{task}_{layer}.npz', **arrays)
        return result
