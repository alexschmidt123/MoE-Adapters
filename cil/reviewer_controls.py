"""Explicit execution budgets and aligned sample-level routing capture."""
import numpy as np
import torch
from torch import nn

def execution_budget(policy,k,n,classes,base=10,step=5,legacy_graph=False):
    if policy not in ['fixed','adaptive','legacy']:raise ValueError(policy)
    adaptive=policy=='adaptive' or (policy=='legacy' and legacy_graph and classes is not None)
    if adaptive:
        if classes is None:raise ValueError('Adaptive routing requires current-stage class count, not a test-image task label')
        if step<=0:raise ValueError('Adaptive step must be positive')
        k=k+(int(classes)-int(base))//int(step)
    return min(int(n),max(1,int(k)))

class IndexedEvaluation(torch.utils.data.Dataset):
    """Stable original test IDs and labels; metadata never enters decisions."""
    def __init__(self,dataset):self.dataset=dataset
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        image,target,task=self.dataset[index];subset=self.dataset;local=index
        if isinstance(subset,torch.utils.data.ConcatDataset):
            import bisect
            part=bisect.bisect_right(subset.cumulative_sizes,index)
            local=index-(subset.cumulative_sizes[part-1] if part else 0);subset=subset.datasets[part]
        if not hasattr(subset,'indices') or not hasattr(subset,'y'):raise TypeError('Stable source IDs are unavailable')
        original=int(subset.indices[local]);fine=int(subset.y[original])
        return image,target,task,original,fine

class SampleCollector:
    """Stage-stratified uniform reservoir with aligned sample graph metadata."""
    def __init__(self,path,seed,limit=256):
        from pathlib import Path
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.seed=seed;self.limit=limit;self.reservoir={};self.stats={}
    def collect(self,model,stage,ids,labels,tasks):
        for name,module in model.named_modules():
            diag=getattr(module,'_last_routing_diagnostics',None)
            if diag is None or getattr(module,'text_or_image',None)!='image':continue
            gates=diag['gates'].detach().float().cpu().numpy();dense=diag['routing_weights'].detach().float().cpu().numpy()
            graph=getattr(module,'graph_mixer',None);adj=getattr(graph,'_last_adjacency',None)
            adj=None if adj is None else adj.detach().float().cpu().numpy()
            key=(int(stage),name);n=gates.shape[1]
            if key not in self.stats:
                self.stats[key]=dict(samples=0,selections=np.zeros(n,dtype=np.int64),class_counts=np.zeros(100,dtype=np.int64),class_selections=np.zeros((100,n),dtype=np.int64),entropy_sum=0.,k_sum=0)
                self.reservoir[key]=dict(seen=0,rows=[],rng=np.random.default_rng(self.seed+stage*1009+sum(map(ord,name))))
            stat=self.stats[key];stat['samples']+=len(gates);stat['selections']+=(gates>0).sum(0);stat['k_sum']+=int(diag['effective_top_k'])*len(gates)
            stat['entropy_sum']+=float(-(dense*np.log(np.maximum(dense,1e-30))).sum())
            for i in range(len(gates)):
                fine=int(labels[i]);stat['class_counts'][fine]+=1;stat['class_selections'][fine]+=(gates[i]>0)
                reservoir=self.reservoir[key];reservoir['seen']+=1
                dest=len(reservoir['rows']) if len(reservoir['rows'])<self.limit else int(reservoir['rng'].integers(reservoir['seen']))
                if dest<self.limit:
                    record=dict(gates=gates[i].copy(),sample_id=int(ids[i]),label=fine,task=int(tasks[i]))
                    if adj is not None:record['adjacency']=adj[i].copy()
                    if dest==len(reservoir['rows']):reservoir['rows'].append(record)
                    else:reservoir['rows'][dest]=record
            module._last_routing_diagnostics=None
            if graph is not None:graph._last_adjacency=None
    def export(self):
        import json
        summary=[]
        for (stage,name),stat in self.stats.items():
            short='L'+name.split('visual.transformer.resblocks.')[-1].split('.')[0]
            rows=self.reservoir[(stage,name)]['rows'];fields=set.intersection(*(set(r) for r in rows)) if rows else set()
            np.savez_compressed(self.path/f'stage{stage}_{short}.npz',**{key:np.stack([r[key] for r in rows]) for key in fields})
            summary.append(dict(stage=stage,layer=name,**{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in stat.items()}))
        (self.path/'routing_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
