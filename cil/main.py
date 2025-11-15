
import os
import json
import hydra
import logging
from omegaconf import DictConfig, OmegaConf

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


@hydra.main(config_path=None, config_name=None, version_base="1.1") 
def continual_clip(cfg: DictConfig) -> None:

    cfg.workdir = utils.get_workdir(path=os.getcwd())
    cfg.dataset_root = os.path.join(cfg.workdir, cfg.dataset_root)

    utils.save_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, cfg.class_order))
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
        # breakpoint()
        for inputs, targets, task_ids in tqdm(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, task_ids)
            metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")

        acc_list.append(100 * metric_logger.accuracy)
        with open(cfg.log_path, 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'acc': round(100 * metric_logger.accuracy, 2),
                'avg_acc': round(100 * metric_logger.average_incremental_accuracy, 2),
                'forgetting': round(100 * metric_logger.forgetting, 6),
                'acc_per_task': [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task],
                'bwt': round(100 * metric_logger.backward_transfer, 2),
                'fwt': round(100 * metric_logger.forward_transfer, 2),
            }) + '\n')
            metric_logger.end_task()
        # assert 1 == 2
    with open(cfg.log_path, 'a+') as f:
        f.write(json.dumps({
            'last': round(acc_list[-1], 2), 
            'avg': round(statistics.mean(acc_list), 2)
        }) + '\n')

        



if __name__ == "__main__":
    continual_clip()