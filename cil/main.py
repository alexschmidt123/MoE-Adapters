
import os
import json
import datetime
import hydra
import logging
from omegaconf import DictConfig, OmegaConf, open_dict

from tqdm import tqdm

import torch
import statistics
from torch.utils.data import DataLoader
from continuum.metrics import Logger

from continual_clip import utils
from continual_clip.models import load_model
from continual_clip.datasets import build_cl_scenarios

# Register custom resolver to get config name without extension
# This will be called when Hydra resolves the directory path
def get_config_name_no_ext():
    try:
        # Try to get from HydraConfig if available
        from hydra.core.hydra_config import HydraConfig
        cfg = HydraConfig.get()
        config_name = cfg.job.config_name
        # Remove .yaml extension if present
        if config_name.endswith('.yaml'):
            return config_name[:-5]  # Remove .yaml
        return config_name
    except:
        # Fallback: try to get from environment or use default
        import os
        config_name = os.environ.get('HYDRA_CONFIG_NAME', 'experiment')
        if config_name.endswith('.yaml'):
            return config_name[:-5]
        return config_name

OmegaConf.register_new_resolver("config_name_no_ext", get_config_name_no_ext)


def _get_in_cfg(cfg, key, default=None):
    """Get key from root or from any nested level (e.g. config group)."""
    try:
        v = OmegaConf.select(cfg, key, default=OmegaConf.MISSING)
        if v is not OmegaConf.MISSING:
            return v
    except Exception:
        pass
    try:
        d = OmegaConf.to_container(cfg, resolve=False)
    except Exception:
        return default
    if not isinstance(d, dict):
        return default

    def find(d, k):
        if k in d:
            return d[k]
        for kk, v in d.items():
            if kk == "hydra":
                continue
            if isinstance(v, dict):
                found = find(v, k)
                if found is not None:
                    return found
        return None

    return find(d, key) or default


def _normalize_cfg(cfg: DictConfig) -> DictConfig:
    """Return a new unstructured config with all non-hydra content merged at root (fixes config-group layout)."""
    try:
        d = OmegaConf.to_container(cfg, resolve=False)
    except Exception:
        return cfg
    if not isinstance(d, dict):
        return cfg

    merged = {}
    for key, val in d.items():
        if key.startswith("_") or key == "hydra":
            continue
        if isinstance(val, dict):
            for k, v in val.items():
                if k.startswith("_"):
                    continue
                if k not in merged:
                    merged[k] = v
        else:
            if key not in merged:
                merged[key] = val

    out = OmegaConf.create(merged)
    OmegaConf.set_struct(out, False)
    if "hydra" in d:
        out["hydra"] = d["hydra"]
    return out


@hydra.main(config_path=None, config_name=None, version_base="1.1") 
def continual_clip(cfg: DictConfig) -> None:

    cfg = _normalize_cfg(cfg)

    # Ensure all results go under experiments/ (never create an "output" or "outputs" folder)
    cwd = os.getcwd()
    workdir = utils.get_workdir(path=cwd)
    experiments_root = os.path.join(workdir, "experiments")
    if "experiments" not in os.path.normpath(os.path.abspath(cwd)):
        redirect_dir = os.path.join(
            experiments_root,
            "run-" + datetime.datetime.now().strftime("%m%d%Y-%H%M%S")  # mmddyyyy-HHMMSS
        )
        os.makedirs(redirect_dir, exist_ok=True)
        os.chdir(redirect_dir)

    workdir = utils.get_workdir(path=os.getcwd())
    with open_dict(cfg):
        cfg.workdir = workdir
        cfg.dataset_root = os.path.join(workdir, cfg.dataset_root)
        cfg.class_order = utils.get_class_order(os.path.join(workdir, cfg.class_order))

    utils.save_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(cfg, device)

    eval_dataset, classes_names = build_cl_scenarios(
        cfg, is_train=False, transforms=model.transforms
    )
    print(eval_dataset, eval_dataset)
    # print('eval_classname', classes_names)
    train_dataset, train_classes_names = build_cl_scenarios(
        cfg, is_train=True, transforms=model.transforms
    )
    # print('train_classes_names', train_classes_names)
    model.classes_names = classes_names

    with open(cfg.log_path, 'w+') as f: 
        pass

    acc_list = []
    metric_logger = Logger(list_subsets=["test"])

    # test
    for task_id, _ in enumerate(eval_dataset):
        # breakpoint()
        logging.info(f"Evaluation for task {task_id} has started.")
        # breakpoint()
        model.adaptation(task_id, cfg, train_dataset, train_classes_names)  # task id 已经传入model

        eval_loader = DataLoader(eval_dataset[:task_id + 1], batch_size=64)
        if getattr(model, "routing_analyzer", None) is not None:
            model.routing_evaluation_stage = task_id
        # Uneven scenario: eval dataset returns task-local labels per subset, but model
        # outputs logits over global class indices. Remap targets to global for correct accuracy.
        use_uneven = getattr(cfg, "task_sizes", None) is not None
        if use_uneven:
            # offsets[t] = number of classes in tasks 0..t-1 (global index of first class in task t)
            class_ids_per_task = list(utils.get_class_ids_per_task(cfg))
            offsets = [0]
            for t in range(1, task_id + 1):
                offsets.append(offsets[-1] + len(class_ids_per_task[t - 1]))
        # Disable tqdm progress bar if TQDM_DISABLE environment variable is set
        disable_tqdm = os.environ.get("TQDM_DISABLE", "0") == "1"
        for inputs, targets, task_ids in tqdm(eval_loader, disable=disable_tqdm):
            inputs, targets = inputs.to(device), targets.to(device)
            if use_uneven:
                # task_ids: [B], targets: [B] task-local; convert to global class indices
                offsets_t = torch.tensor(offsets, dtype=torch.long)
                global_targets = offsets_t[task_ids] + targets.cpu()
            else:
                global_targets = targets.cpu()
            outputs = model(inputs, task_ids)
            metric_logger.add([outputs.cpu().argmax(dim=1), global_targets, task_ids], subset="test")

        acc_list.append(100 * metric_logger.accuracy)
        log_entry = {
            'task': task_id,
            'acc': round(100 * metric_logger.accuracy, 2),
            'avg_acc': round(100 * metric_logger.average_incremental_accuracy, 2),
            'forgetting': round(100 * metric_logger.forgetting, 6),
            'acc_per_task': [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task],
            'bwt': round(100 * metric_logger.backward_transfer, 2),
            'fwt': round(100 * metric_logger.forward_transfer, 2),
        }
        # Uneven scenario: show class counts so acc_per_task is read as (task_id -> classes in task)
        if use_uneven:
            task_sizes = list(getattr(cfg, 'task_sizes', []))
            log_entry['task_sizes'] = task_sizes
            log_entry['classes_per_task'] = task_sizes[: task_id + 1]
        with open(cfg.log_path, 'a+') as f:
            f.write(json.dumps(log_entry) + '\n')
            metric_logger.end_task()
        # Persist partial analysis after every stage, not only at run end.
        if getattr(model, "routing_analyzer", None) is not None:
            model.routing_analyzer.export()
        # assert 1 == 2
    with open(cfg.log_path, 'a+') as f:
        summary = {
            'last': round(acc_list[-1], 2),
            'avg': round(statistics.mean(acc_list), 2)
        }
        if getattr(cfg, 'task_sizes', None) is not None:
            summary['task_sizes'] = list(cfg.task_sizes)
        f.write(json.dumps(summary) + '\n')

    if getattr(model, "routing_analyzer", None) is not None:
        analysis_dir = model.routing_analyzer.export()
        print(f"Routing analysis saved to: {analysis_dir}")

        



if __name__ == "__main__":
    continual_clip()
