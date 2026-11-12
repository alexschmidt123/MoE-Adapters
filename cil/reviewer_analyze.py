"""Produce auditable tables and sample figures from task-agnostic run artifacts."""
import argparse,csv,json,pickle
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--output',required=True);p.add_argument('--cifar-root',required=True);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);table=[];behavior=[];figures=[];common_profile=None
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 data=pickle.load(open(Path(a.cifar_root)/'cifar-100-python/test','rb'),encoding='bytes');images=data[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1);labels=np.array(data[b'fine_labels']);meta=pickle.load(open(Path(a.cifar_root)/'cifar-100-python/meta','rb'),encoding='bytes');names=[x.decode() for x in meta[b'fine_label_names']]
 for r in sorted(Path(a.runs).iterdir()):
  if not r.is_dir() or not (r/'COMPLETE.json').exists():continue
  complete=json.loads((r/'COMPLETE.json').read_text())
  if complete.get('smoke'):continue
  metrics=json.loads((r/'metrics.json').read_text());prov=json.loads((r/'provenance.json').read_text())
  assert prov['evaluation']=='task-agnostic latest-stage router'
  cfg=OmegaConf.load(r/'config.yaml')
  profile={key:OmegaConf.to_container(cfg[key]) if OmegaConf.is_config(cfg[key]) else cfg[key] for key in ['model_name','task_sizes','epochs','batch_size','lr','weight_decay','ls','prompt_template','dataset']}
  profile['native_regularizer_weight']=cfg.model.get('moe_router_z_loss_weight',0)
  profile['load_balance_weight']=cfg.model.get('moe_load_balance_weight',0)
  if common_profile is None:common_profile=profile
  assert common_profile==profile,'Incompatible study configs: analyze schedules/training protocols separately'
  table.append({'run':r.name,'seed':prov['seed'],'stages':len(metrics),'last_accuracy':metrics[-1]['acc'],'average_incremental_accuracy':metrics[-1]['avg_acc'],'forgetting':metrics[-1]['forgetting'],'training_seconds':sum(x['training_seconds'] for x in metrics),'evaluation_seconds':sum(x['evaluation_seconds'] for x in metrics),'native_updates':sum(x['native_updates'] for x in metrics),'initial_adapter_hash':prov['initial_adapter_hash']})
  for f in sorted((r/'sample_routing').glob('stage*_L*.npz')):
   z=np.load(f);g=z['gates'];n=len(g);anchor=g>0;row={'run':r.name,'record':f.stem,'sample_count':n,'mean_selected':float(anchor.sum(1).mean()),'distinct_anchor_sets':len(np.unique(anchor,axis=0))}
   assert np.array_equal(labels[z['sample_id']],z['label']),'Image identity/fine-label mismatch'
   if 'adjacency' in z:
    adj=z['adjacency'];weights=np.einsum('bi,bij->bj',g,adj);weights=weights/weights.sum(1,keepdims=True)
    row['effective_weighted_neighbors']=float(np.exp(-(weights*np.log(np.maximum(weights,1e-30))).sum(1)).mean())
    row['mean_structural_union']=float(((anchor[:,:,None]* (adj>0)).any(1)).sum(1).mean())
    row['sample_adjacency_variance']=float(adj.var(0).mean())
    # Predetermined rule: final stage and last vision layer; maximize neighborhood
    # difference among different-class pairs with partially overlapping anchors.
    if f.stem==f'stage{len(metrics)-1}_L11':
     pairs=[]
     for i in range(n):
      for j in range(i):
       shared=(anchor[i]&anchor[j]).sum();union=(anchor[i]|anchor[j]).sum()
       if z['label'][i]!=z['label'][j] and 0<shared<union:
        pairs.append((float(abs(weights[i]-weights[j]).sum()),i,j))
     if pairs:
      _,i,j=max(pairs);fig,axes=plt.subplots(2,2,figsize=(9,7))
      for rowidx,k in enumerate([i,j]):
       axes[rowidx,0].imshow(images[int(z['sample_id'][k])]);axes[rowidx,0].axis('off');axes[rowidx,0].set_title(names[int(z['label'][k])]+'; ID '+str(z['sample_id'][k]))
       im=axes[rowidx,1].imshow(adj[k],vmin=0,vmax=1,cmap='viridis');axes[rowidx,1].set_title('Anchors '+str(np.flatnonzero(anchor[k]).tolist()));axes[rowidx,1].set_xlabel('Neighbor node');axes[rowidx,1].set_ylabel('Source node')
      fig.suptitle(r.name+'; latest stage, vision layer 11\nPair selected for large weighted-neighborhood difference; illustrative only');fig.tight_layout();name=r.name+'_sample_relationships.png';fig.savefig(out/name,dpi=160);plt.close(fig);figures.append(name)
   behavior.append(row)
  run_behavior=[x for x in behavior if x['run']==r.name and x['record'].endswith('_L11')]
  if run_behavior:
   run_behavior.sort(key=lambda x:int(x['record'].split('_')[0][5:]));stages=[int(x['record'].split('_')[0][5:]) for x in run_behavior]
   fig,ax=plt.subplots(figsize=(9,4));ax.plot(stages,[x['mean_selected'] for x in run_behavior],marker='o',label='Executed anchors')
   if all('effective_weighted_neighbors' in x for x in run_behavior):ax.plot(stages,[x['effective_weighted_neighbors'] for x in run_behavior],marker='s',label='Effective weighted neighbor nodes')
   ax.set_xticks(stages,[str(t)+'\nC='+str(cfg.task_sizes[t]) for t in stages]);ax.set_xlabel('Evaluation stage and arriving class count');ax.set_ylabel('Experts / effective node count');ax.set_title(r.name+'; vision layer 11');ax.legend();fig.tight_layout();name=r.name+'_stage_behavior.png';fig.savefig(out/name,dpi=160);plt.close(fig);figures.append(name)
  routing=r/'sample_routing/routing_summary.json'
  if routing.exists():
   summaries=json.loads(routing.read_text());chosen=[x for x in summaries if x['stage']==len(metrics)-1 and x['layer'].endswith('resblocks.11')]
   if chosen:
    counts=np.array(chosen[0]['class_counts']);selected=np.array(chosen[0]['class_selections']);freq=selected/np.maximum(counts[:,None],1)
    fig,ax=plt.subplots(figsize=(7,14));im=ax.imshow(np.ma.masked_where(np.repeat(counts[:,None]==0,selected.shape[1],axis=1),freq),vmin=0,vmax=1,aspect='auto');ax.set_yticks(range(len(names)),names,fontsize=5);ax.set_xlabel('Expert');ax.set_title(r.name+'; final-stage class-conditioned anchor frequency');fig.colorbar(im,ax=ax);fig.tight_layout();name=r.name+'_class_utilization.png';fig.savefig(out/name,dpi=150);plt.close(fig);figures.append(name)
 def write(name,rows):
  if rows:
   keys=sorted(set().union(*(x.keys() for x in rows)))
   with open(out/name,'w') as h:
    writer=csv.DictWriter(h,keys);writer.writeheader();writer.writerows(rows)
 write('four_way_per_seed.csv',table);write('stage_routing_behavior.csv',behavior)
 grouped={}
 for row in table:
  method=row['run'].rsplit('_seed',1)[0];grouped.setdefault(method,[]).append(row)
 summary={key:{'n_seeds':len(v),'last_mean':float(np.mean([x['last_accuracy'] for x in v])),'last_sd':float(np.std([x['last_accuracy'] for x in v],ddof=1)) if len(v)>1 else None,'avg_mean':float(np.mean([x['average_incremental_accuracy'] for x in v]))} for key,v in grouped.items()}
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
 differences=[]
 by={(x['run'].rsplit('_seed',1)[0],x['seed']):x for x in table}
 for seed in sorted(set(x['seed'] for x in table)):
  get=lambda m:by.get((m,seed))
  if all(get(m) for m in ['moe_fixed','moe_adaptive','moce_fixed','moce_adaptive']):
   value=lambda m:get(m)['average_incremental_accuracy']
   differences.append({'seed':seed,'graph_fixed':value('moce_fixed')-value('moe_fixed'),'graph_adaptive':value('moce_adaptive')-value('moe_adaptive'),'adaptive_moe':value('moe_adaptive')-value('moe_fixed'),'adaptive_moce':value('moce_adaptive')-value('moce_fixed'),'interaction':value('moce_adaptive')-value('moce_fixed')-value('moe_adaptive')+value('moe_fixed')})
 write('paired_seed_differences.csv',differences)
 (out/'ANALYSIS_LIMITATIONS.md').write_text('Only COMPLETE non-smoke runs are included. Tables use training seeds as replicates. Reservoir structural unions are per-image; stage unions across unequal sample counts are not compared. Illustrative pairs maximize weighted-neighbor difference and cannot demonstrate population prevalence. Correlations across training stages are confounded by class identity and progression. No matched even-versus-uneven table, submitted-run settings, U1-U8 mapping, parameter-matched router comparison or causal community claim is inferred from missing artifacts. See checkpoint inventory and manuscript verification checklist.\n')
 print(json.dumps({'completed_runs':len(table),'figures':figures,'output':str(out)}))
if __name__=='__main__':main()
