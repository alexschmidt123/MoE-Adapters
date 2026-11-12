"""Locked four-way task-agnostic CIL study, native optimization unchanged."""
import argparse, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from continuum.metrics import Logger
from continual_clip import utils
from continual_clip.models import ClassIncremental
from continual_clip.datasets import build_cl_scenarios
from reviewer_controls import IndexedEvaluation, SampleCollector, execution_budget

def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--checkpoint-dir');p.add_argument('--dataset-root');a=p.parse_args()
    a.config=str(Path(a.config).resolve())
    if a.checkpoint_dir:a.checkpoint_dir=str(Path(a.checkpoint_dir).resolve())
    cfg=OmegaConf.load(a.config);out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);os.chdir(out)
    seed=int(cfg.seed);utils.seed_all(seed)
    cfg.workdir=str(Path(__file__).resolve().parent)
    if a.dataset_root:cfg.dataset_root=str(Path(a.dataset_root).resolve())
    if isinstance(cfg.class_order,str):
        order=Path(cfg.class_order)
        if not order.exists():order=Path(cfg.workdir)/'class_orders'/order.name
        cfg.class_order=utils.get_class_order(str(order))
    cfg.model.paired_initialization=True
    OmegaConf.save(cfg,'config.yaml');cfg.routing_analysis_enabled=False
    provenance={'seed':seed,'evaluation':'task-agnostic latest-stage router','evaluation_transform':'deterministic resize and center crop','config_sha256':hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),'sources':{str(p.relative_to(Path(__file__).parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('reviewer_controls.py'),Path(__file__).parent/'clip/model.py',Path(__file__).parent/'graph_mixer_proper.py',Path(__file__).parent/'continual_clip/models.py',Path(__file__).parent/'continual_clip/uneven_scenario.py']},'torch':torch.__version__,'gpu':torch.cuda.get_device_name()}
    save(out/'provenance.json',provenance)
    model=ClassIncremental(cfg,torch.device('cuda'))
    original_step=torch.optim.AdamW.step
    gradient_checks=[]
    def checked_step(optimizer,*args,**kwargs):
        grads=[p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), 'Missing or nonfinite training gradients'
        if not gradient_checks:
            gradient_checks.append({'tensors':len(grads),'nonzero':sum(bool(g.count_nonzero()) for g in grads),'nonzero_by_path':{key:sum(p.grad is not None and bool(p.grad.count_nonzero()) for name,p in model.model.named_parameters() if key in name) for key in ['adaptmlp','router','graph_mixer']}})
            save(out/'gradient_check.json',gradient_checks)
        return original_step(optimizer,*args,**kwargs)
    torch.optim.AdamW.step=checked_step
    # Common adapter initialization must match across all arms for a given seed.
    h=hashlib.sha256()
    for name,value in model.model.state_dict().items():
        if 'adaptmlp_list' in name:h.update(name.encode());h.update(value.cpu().numpy().tobytes())
    provenance['initial_adapter_hash']=h.hexdigest();save(out/'provenance.json',provenance)
    from clip.clip import _transform
    evaluation_transform=_transform(model.model.visual.input_resolution,is_train=False)
    test,names=build_cl_scenarios(cfg,False,evaluation_transform);train,train_names=build_cl_scenarios(cfg,True,model.transforms);model.classes_names=names
    collector=SampleCollector(out/'sample_routing',seed);logger=Logger(list_subsets=['test']);records=[]
    if a.smoke:
        original=train.__class__.__getitem__
        def small(self,index):
            ds=original(self,index)
            return Subset(ds,list(range(min(32,len(ds)))))
        train.__class__.__getitem__=small;cfg.epochs=0
    start=time.time();offsets=np.cumsum([0]+list(cfg.task_sizes)[:-1]).tolist()
    for stage in range(1 if a.smoke else len(test)):
        if __import__('shutil').disk_usage(out).free<45*1024**3:raise RuntimeError('Disk guard: less than 45 GiB free')
        torch.cuda.reset_peak_memory_stats();t=time.time()
        if a.checkpoint_dir:
            cp=Path(a.checkpoint_dir)/f'task_{stage:02d}.pt'
            payload=torch.load(cp,map_location='cpu',weights_only=False)
            assert payload.get('stage',payload.get('task_id'))==stage,'Checkpoint stage mismatch'
            original_cfg=OmegaConf.create(payload['config_yaml'])
            for key in ['model_name','task_sizes']:
                assert OmegaConf.to_container(original_cfg[key])==OmegaConf.to_container(cfg[key]) if OmegaConf.is_config(cfg[key]) else original_cfg[key]==cfg[key],key
            original_policy=original_cfg.model.get('execution_budget_policy')
            if original_policy is None:
                original_policy='adaptive' if original_cfg.model.get('graph_mixer_enabled',False) else 'fixed'
            assert original_policy==cfg.model.execution_budget_policy,'Checkpoint training execution policy differs'
            for key in cfg.model:
                if key in ['num_tasks','fair_router_regularization','execution_budget_policy','paired_initialization']:continue
                assert original_cfg.model.get(key)==cfg.model.get(key),'Incompatible checkpoint config: '+key
            if not a.smoke:
                for key in ['epochs','batch_size','lr','weight_decay','ls','prompt_template','dataset']:
                    assert original_cfg.get(key)==cfg.get(key),'Checkpoint training settings differ: '+key
                original_regularized=bool(original_cfg.model.get('fair_router_regularization',False) or original_cfg.model.get('graph_mixer_enabled',False))
                desired_regularized=bool(cfg.model.get('fair_router_regularization',False) or cfg.model.get('graph_mixer_enabled',False))
                if float(cfg.model.get('moe_router_z_loss_weight',0))>0:
                    assert original_regularized==desired_regularized,'Legacy checkpoint trained with different router regularization; cannot relabel as matched control'
            stored_seed=payload.get('provenance',{}).get('seed',original_cfg.get('seed'))
            assert stored_seed==seed,'Checkpoint seed unknown or different; resolve provenance before comparison'
            assert list(payload['class_ids_per_task'])==list(model.class_ids_per_task),'Class order mismatch'
            model.model.load_state_dict(payload['model_state_dict'],strict=True)
            model.current_class_names=payload['current_class_names'];model.text_tokens=payload['text_tokens'].cuda();model.model.eval()
            save(out/f'checkpoint_source_{stage}.json',{'path':str(cp.resolve()),'sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'legacy_evaluation_replaced':'latest stage router'})
        else:model.adaptation(stage,cfg,train,train_names)
        torch.cuda.synchronize();training_seconds=0. if a.checkpoint_dir else time.time()-t
        if a.smoke:train.__class__.__getitem__=original
        if not a.checkpoint_dir:
            state={k:v.detach().cpu().clone() for k,v in model.model.state_dict().items()}
            directory=out/'checkpoints';directory.mkdir(exist_ok=True);target=directory/f'task_{stage:02d}.pt';tmp=target.with_suffix('.tmp')
            payload={'model_state_dict':state,'config_yaml':OmegaConf.to_yaml(cfg),'stage':stage,'class_ids_per_task':model.class_ids_per_task,'classes_names':names,'current_class_names':model.current_class_names,'text_tokens':model.text_tokens.cpu(),'provenance':provenance,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
            torch.save(payload,tmp);check=torch.load(tmp,map_location='cpu',weights_only=False)
            assert all(torch.equal(v,check['model_state_dict'][k]) for k,v in state.items());del check,state,payload;os.replace(tmp,target)
            print('CHECKPOINT_VERIFIED',stage,flush=True)
        ds=IndexedEvaluation(test[:stage+1]);ds=Subset(ds,range(min(64,len(ds)))) if a.smoke else ds
        elapsed=time.time();latency=0.;images=0
        for inputs,targets,tasks,ids,labels in DataLoader(ds,batch_size=64,num_workers=4):
            inputs=inputs.cuda();t=time.time()
            with torch.no_grad():
                logits,_=model.model(inputs,model.text_tokens,stage,is_train=False,num_classes_current_task=len(model.class_ids_per_task[stage]))
            torch.cuda.synchronize();latency+=time.time()-t;images+=len(inputs)
            expected=execution_budget(cfg.model.execution_budget_policy,2,8,len(model.class_ids_per_task[stage]))
            for module in model.model.modules():
                diag=getattr(module,'_last_routing_diagnostics',None)
                if diag is not None:
                    assert diag['effective_top_k']==expected
                    assert torch.all((diag['gates']>0).sum(1)==expected)
            # Metadata is used only after inference, never to choose a router.
            collector.collect(model.model,stage,ids,labels,tasks)
            global_targets=torch.tensor(offsets)[tasks]+targets
            logger.add([logits.cpu().argmax(1),global_targets,tasks],subset='test')
        record={'task':stage,'acc':100*logger.accuracy,'avg_acc':100*logger.average_incremental_accuracy,'forgetting':100*logger.forgetting,'acc_per_task':[100*x for x in logger.accuracy_per_task],'training_seconds':training_seconds,'evaluation_seconds':time.time()-elapsed,'inference_seconds_per_image':latency/images,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'executed_k':expected,'native_updates':0 if a.checkpoint_dir else 1 if a.smoke else int(cfg.epochs)*int(np.ceil(len(train[stage:stage+1])/cfg.batch_size))+1}
        records.append(record);save(out/'metrics.json',records);collector.export();logger.end_task();print(json.dumps(record),flush=True)
    save(out/'COMPLETE.json',{'seconds':time.time()-start,'stages':len(records),'smoke':a.smoke})
if __name__=='__main__':main()
