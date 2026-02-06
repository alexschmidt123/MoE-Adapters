from omegaconf import DictConfig, open_dict
from tqdm import tqdm
import torch.nn.functional as F

import clip.clip as clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import get_class_ids_per_task, get_class_names, get_num_tasks, batch, merge_we_router, wise_we, moving_avg, l2_loss, \
    virtual_vocab, distillation
import copy

from .cc import conceptual_captions

from . import utils
import os
import random

from .dynamic_dataset import DynamicDataset


class ClassIncremental(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        # So that CLIP MoE allocates one router per task (fixes uneven / multi-task)
        num_tasks = get_num_tasks(cfg)
        with open_dict(cfg):
            if not hasattr(cfg, "model") or cfg.model is None:
                from omegaconf import OmegaConf
                cfg.model = OmegaConf.create({})
        with open_dict(cfg.model):
            cfg.model.num_tasks = num_tasks
        self.model, self.transforms, _ = clip.load(cfg.model_name, device=device, jit=jit, cfg=cfg)
        self.ref_model = None
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.dynamic_dataset = DynamicDataset(cfg)

    def forward(self, image, taskid):
        with torch.no_grad():
            logits_per_image, _ = self.model(image, self.text_tokens, 0, is_train=False)
            probs = logits_per_image.softmax(dim=-1)
        return probs

    def adaptation(self, task_id, cfg, train_dataset, train_classes_names):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)

        if cfg.method != "zeroshot":
            self.train(task_id, cfg, train_dataset, train_classes_names)

    def train(self, task_id, cfg, train_dataset, train_classes_names):
        ### laoding dataset
        train_loader = DataLoader(train_dataset[task_id:task_id + 1],
                                  batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=8)

        train_iter = iter(train_loader)  # 获取每个step的数据集
        # print('cfg.batch_size',cfg.batch_size)


        EPOCH = getattr(cfg, 'epochs', 1)  # Configurable epochs, default to 1 for backward compatibility
        num_batches = len(train_loader)
        total_iterations = EPOCH * num_batches
        print(f"Training task {task_id}: {EPOCH} epoch(s), {num_batches} batches per epoch, {total_iterations} total iterations")

        ### whole-model
        exclude_params_name = ["logit_scale"]

        # 冻结参数
        for k, v in self.model.named_parameters():  # 冻结其他参数
            if "adaptmlp" not in k and "router" not in k and "noise" not in k and "graph_mixer" not in k and "alpha_graph" not in k:
                v.requires_grad = False


        params = [
            v for k, v in self.model.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k or "graph_mixer" in k or "alpha_graph" in k
        ]
        params_name = [
            k for k, v in self.model.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k or "graph_mixer" in k or "alpha_graph" in k
        ]
        # print('========trainable params============', params_name)

        logit_scale = self.model.logit_scale

        # optimizer
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = utils.cosine_lr(
            optimizer, cfg.lr, 30, total_iterations
        )

        # move model to device
        self.model = self.model.cuda()
        devices = list(range(torch.cuda.device_count()))
        # print("Using devices", devices)

        # text
        # For training, we use only the current task's classes
        # The model output will have shape [batch_size, num_classes_in_current_task]
        # So labels should remain 0-indexed for the current task (no shift needed)
        classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        print(classnames)
        texts = [self.prompt_template.format(c) for c in classnames]

        texts = clip.tokenize(texts).to(self.device)

        # method

        # start training
        self.model.train()
        # Disable tqdm progress bar if TQDM_DISABLE environment variable is set
        disable_tqdm = os.environ.get("TQDM_DISABLE", "0") == "1"
        for iteration in tqdm(range(total_iterations + 1), disable=disable_tqdm):
            scheduler(iteration)
            try:
                inputs, targets, task_ids = next(train_iter)
            except:
                train_iter = iter(train_loader)
                inputs, targets, task_ids = next(train_iter)

            # Continuum library remaps labels to be 0-indexed per task
            # However, for TinyImageNet, continuum may use cumulative labels instead
            # We need to remap them to 0-indexed for the current task
            # Check label range and remap if necessary
            num_classes_current_task = len(texts)
            if iteration == 0:  # Debug: print label info on first iteration
                print(f"Task {task_id}: Label range before remap: min={targets.min().item()}, max={targets.max().item()}, num_classes={num_classes_current_task}")
            
            # Remap labels to 0-indexed for current task if needed
            # Continuum should do this, but for TinyImageNet it might use cumulative indices
            if targets.max().item() >= num_classes_current_task:
                # Labels are cumulative, need to remap to task-local
                # Find the minimum label value for this task (should be the first class index)
                min_label = targets.min().item()
                # Remap: subtract the minimum to get 0-indexed
                targets = targets - min_label
                if iteration == 0:
                    print(f"Task {task_id}: Remapped labels - new range: min={targets.min().item()}, max={targets.max().item()}")
            
            # Ensure labels are in valid range [0, num_classes_current_task - 1]
            assert targets.min().item() >= 0 and targets.max().item() < num_classes_current_task, \
                f"Task {task_id}: Invalid label range! min={targets.min().item()}, max={targets.max().item()}, num_classes={num_classes_current_task}"

            inputs, targets = inputs.cuda(), targets.cuda()

            num_classes_current_task = len(self.class_ids_per_task[task_id])
            logits_per_image, _ = self.model(
                inputs, texts, task_id, is_train=True,
                num_classes_current_task=num_classes_current_task
            )
            # -- cross entropy loss --
            loss = F.cross_entropy(logits_per_image, targets, label_smoothing=cfg.ls)
            
            # Add extra losses from graph mixer (if any)
            # Collect extra losses from all ResidualAttentionBlocks
            extra_loss = 0.0
            for module in self.model.modules():
                if hasattr(module, 'extra_losses') and module.extra_losses is not None:
                    extra_loss = extra_loss + module.extra_losses
                    # Reset for next iteration
                    module.extra_losses = None
            
            if extra_loss != 0.0:
                loss = loss + extra_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        self.model.eval()


class DomainIncremental(nn.Module):
    pass


class TaskAgnostic(nn.Module):
    pass


def load_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    r"""Load a CLIP model in different continual scenarios.

    Arguments:
        cfg (DictConfig): Experiment configurations.
        device (torch.device): Device to train (or) evaluate the model on.

    Returns:
        nn.Module: Return scenario specific CLIP model.
    """
    if cfg.scenario == "class":
        return ClassIncremental(cfg, device)
    elif cfg.scenario == "domain":
        return DomainIncremental(cfg, device)
    elif cfg.scenario == "task-aganostic":
        return TaskAgnostic(cfg, device)
    else:
        raise ValueError(f"""
            `{cfg.scenarios}` is not a valid scenario, 
            Please choose from ['class', "domain', 'task-agnostic']
        """)

