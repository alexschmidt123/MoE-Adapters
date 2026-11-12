"""Portable checkpoint-first handoff. No scientific runs start without `run`."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from omegaconf import OmegaConf
BASE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 init=sub.add_parser('init');init.add_argument('--output',required=True);init.add_argument('--dataset-root',required=True);init.add_argument('--seeds',default='0,1,2');init.add_argument('--task-sizes');init.add_argument('--neighbor-budget',type=int)
 audit=sub.add_parser('audit');audit.add_argument('--checkpoint-root',required=True);audit.add_argument('--output',required=True)
 run=sub.add_parser('run');run.add_argument('--manifest',required=True);run.add_argument('--train-missing',action='store_true')
 a=p.parse_args()
 if a.command=='init':
  root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True);manifest=root/'jobs.json'
  if manifest.exists():raise FileExistsError('Preserve existing jobs.json; choose another output or edit intentionally')
  ref=BASE/'configs/class/09082026_formal_moe_moce/s10-MoCE-N8-cifar100-ts28-19-13-11-8-6-5-4-3-3.yaml';cfg=OmegaConf.load(ref);del cfg.hydra
  if a.task_sizes:cfg.task_sizes=list(map(int,a.task_sizes.split(',')))
  if a.neighbor_budget is not None:cfg.model.graph_adj_top_m=a.neighbor_budget
  cfg.dataset_root=str(Path(a.dataset_root).resolve());cfg.class_order=str(BASE/'class_orders/cifar100.yaml');cfg.model.fair_router_regularization=True;cfg.model.paired_initialization=True
  jobs=[]
  for seed in map(int,a.seeds.split(',')):
   for graph,budget in [(False,'fixed'),(False,'adaptive'),(True,'fixed'),(True,'adaptive')]:
    x=OmegaConf.create(OmegaConf.to_container(cfg));x.seed=seed;x.model.graph_mixer_enabled=graph;x.model.execution_budget_policy=budget
    name=('moce' if graph else 'moe')+'_'+budget+'_seed'+str(seed);config=root/(name+'.yaml');OmegaConf.save(x,config)
    jobs.append({'name':name,'config':str(config),'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'output':str(root/'runs'/name),'checkpoint_dir':None})
  manifest.write_text(json.dumps(jobs,indent=2));print(manifest);return
 if a.command=='audit':
  # Audit only user-owned, trusted torch files: torch.load may deserialize Python.
  import torch
  rows=[]
  for cp in sorted(p for p in Path(a.checkpoint_root).rglob('*') if p.is_file() and p.suffix in ['.pt','.pth','.ckpt']):
   row={'path':str(cp.resolve()),'bytes':cp.stat().st_size}
   try:
    obj=torch.load(cp,map_location='cpu',weights_only=False)
    row['keys']=list(obj) if isinstance(obj,dict) else [];row['stage']=obj.get('stage',obj.get('task_id')) if isinstance(obj,dict) else None
    required=['model_state_dict','config_yaml','class_ids_per_task','current_class_names','text_tokens'];row['native_stage_format']=isinstance(obj,dict) and all(k in obj for k in required)
    if row['native_stage_format']:
     cfg=OmegaConf.create(obj['config_yaml']);row['configuration']=OmegaConf.to_container(cfg,resolve=False);row['note']='Candidate only: run performs strict architecture/class-order/config validation.'
    else:row['blocker']='Unknown or incomplete format: provide explicit reviewed conversion; do not guess state mappings or initialize missing weights.'
   except Exception as e:row['blocker']=repr(e)
   rows.append(row)
  target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(rows,indent=2));print(target,len(rows));return
 jobs=json.loads(Path(a.manifest).read_text());root=Path(a.manifest).resolve().parent
 lock=open(root/'campaign.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=dict(os.environ,TQDM_DISABLE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
 for job in jobs:
  config=Path(job['config']);assert hashlib.sha256(config.read_bytes()).hexdigest()==job['config_sha256'],'Configuration lock changed; review and explicitly reseal the manifest before runs'
  out=Path(job['output'])
  if out.exists():
   if (out/'COMPLETE.json').exists() and json.loads((out/'provenance.json').read_text())['config_sha256']==job['config_sha256']:continue
   raise RuntimeError('Existing incomplete/incompatible output preserved: '+str(out))
  cp=job.get('checkpoint_dir')
  if not cp and not a.train_missing:raise RuntimeError('No checkpoint mapping for '+job['name']+'. Supply compatible checkpoints or explicitly use --train-missing.')
  if cp:
   missing=[str(Path(cp)/f'task_{stage:02d}.pt') for stage in range(10) if not (Path(cp)/f'task_{stage:02d}.pt').exists()]
   if missing:raise RuntimeError('Incomplete checkpoint mapping; preserved, no automatic retraining: '+str(missing))
  active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
  if active:raise RuntimeError('GPU already has compute jobs; defer launch without interrupting: '+active)
  args=[sys.executable,str(BASE/'reviewer_run.py'),'--config',str(config),'--output',str(out)]
  if cp:args+=['--checkpoint-dir',cp]
  record=root/(job['name']+'.log');monitor=root/(job['name']+'_resources.jsonl')
  with open(record,'x') as log,open(monitor,'x') as samples:
   proc=subprocess.Popen(args,cwd=BASE,env=env,stdout=log,stderr=subprocess.STDOUT)
   while proc.poll() is None:
    sample={'time':time.time(),'pid':proc.pid}
    try:
     sample['gpu_csv']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],text=True).strip()
     sample['process_ps']=subprocess.check_output(['ps','-p',str(proc.pid),'-o','%cpu=,rss=,etime='],text=True).strip()
     sample['memory_kb']={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))}
    except Exception as e:sample['monitor_error']=repr(e)
    samples.write(json.dumps(sample)+'\n');samples.flush();time.sleep(5)
   if proc.returncode:raise RuntimeError('Run failed, campaign stopped: '+str(record))
 print('All mapped runs complete. Run reviewer_analyze.py as documented; manuscript-only verification remains explicit.')
if __name__=='__main__':main()
