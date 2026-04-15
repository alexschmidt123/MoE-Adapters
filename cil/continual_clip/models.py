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
import csv

from .dynamic_dataset import DynamicDataset


def _freeze_gnn_path(model):
    """Freeze all graph_mixer parameters in the vision transformer (GoE drift control)."""
    if not hasattr(model, "visual") or model.visual is None:
        return
    vis = model.visual
    if not hasattr(vis, "transformer") or vis.transformer is None:
        return
    any_frozen = False
    for block in vis.transformer.resblocks:
        if hasattr(block, "graph_mixer") and block.graph_mixer is not None:
            for p in block.graph_mixer.parameters():
                if p.requires_grad:
                    any_frozen = True
                p.requires_grad = False
    if any_frozen:
        print("GoE: GNN path frozen (goe_freeze_gnn_after_task).")


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
            # MoE/GoE use router_list[taskid] per task; eval batches mix tasks 0..T → must use correct router per sample.
            # (Using router 0 for all caused bad acc for task 1, task 8, and every other task in the reported metrics.)
            num_tasks = len(self.class_ids_per_task)
            if isinstance(taskid, torch.Tensor) and taskid.dim() >= 1 and taskid.numel() > 1:
                # Per-sample task ids [B]: split by task, run with correct taskid each, then reorder
                taskid_flat = taskid.view(-1).to(image.device)
                unique_tasks = taskid_flat.unique(sorted=True)
                logits_list = []
                indices_list = []
                for t in unique_tasks:
                    t_int = min(int(t.item()), num_tasks - 1)  # clamp to valid router index
                    mask = (taskid_flat == t)
                    indices_list.append(mask.nonzero(as_tuple=True)[0])
                    img_t = image[mask]
                    logits_t, _ = self.model(img_t, self.text_tokens, t_int, is_train=False)
                    logits_list.append(logits_t)
                # Reorder to original batch order (same order as taskid)
                logits_per_image = torch.zeros(
                    image.shape[0], logits_list[0].shape[1], device=image.device, dtype=logits_list[0].dtype
                )
                for inds, logits_t in zip(indices_list, logits_list):
                    logits_per_image[inds] = logits_t
            else:
                tid = int(taskid.item()) if isinstance(taskid, torch.Tensor) else int(taskid)
                tid = min(tid, num_tasks - 1)
                logits_per_image, _ = self.model(image, self.text_tokens, tid, is_train=False)
            probs = logits_per_image.softmax(dim=-1)
        return probs

    def adaptation(self, task_id, cfg, train_dataset, train_classes_names):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)

        if cfg.method != "zeroshot":
            self.train(task_id, cfg, train_dataset, train_classes_names)
            # Optional: freeze GNN path after a given task to reduce router-input drift (GoE)
            freeze_after = getattr(getattr(cfg, "model", None), "goe_freeze_gnn_after_task", -1)
            if isinstance(freeze_after, (int, float)) and task_id >= int(freeze_after) and freeze_after >= 0:
                _freeze_gnn_path(self.model)

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

        # Change D: optional separate LRs for experts (plastic), GNN (stable), router (medium)
        lr_experts = getattr(cfg, 'lr_experts', None)
        lr_gnn = getattr(cfg, 'lr_gnn', None)
        lr_router = getattr(cfg, 'lr_router', None)
        use_split_lr = (lr_experts is not None and lr_gnn is not None and lr_router is not None)

        if use_split_lr:
            params_experts = [v for k, v in self.model.named_parameters() if v.requires_grad and "adaptmlp" in k]
            params_gnn = [v for k, v in self.model.named_parameters() if v.requires_grad and "graph_mixer" in k]
            params_router = [v for k, v in self.model.named_parameters() if v.requires_grad and ("router" in k or "noise" in k)]
            param_groups = [{"params": params_experts, "lr": lr_experts}, {"params": params_router, "lr": lr_router}]
            base_lrs = [lr_experts, lr_router]
            if params_gnn:
                param_groups.insert(1, {"params": params_gnn, "lr": lr_gnn})
                base_lrs.insert(1, lr_gnn)
            optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
            scheduler = utils.cosine_lr(optimizer, base_lrs, 30, total_iterations)
        else:
            params = [
                v for k, v in self.model.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k or "graph_mixer" in k or "alpha_graph" in k
            ]
            optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
            scheduler = utils.cosine_lr(optimizer, cfg.lr, 30, total_iterations)

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
        loss_csv_file = None
        loss_csv_writer = None
        if getattr(cfg, "save_training_loss_csv", True):
            loss_csv_path = getattr(cfg, "training_loss_csv", "training_loss.csv")
            if not os.path.isabs(loss_csv_path):
                loss_csv_path = os.path.join(os.getcwd(), loss_csv_path)
            os.makedirs(os.path.dirname(loss_csv_path) or ".", exist_ok=True)
            file_has_data = os.path.isfile(loss_csv_path) and os.path.getsize(loss_csv_path) > 0
            loss_csv_file = open(loss_csv_path, "a", newline="")
            loss_csv_writer = csv.writer(loss_csv_file)
            if not file_has_data:
                loss_csv_writer.writerow(["task_id", "iteration", "loss"])
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

            if loss_csv_writer is not None:
                loss_csv_writer.writerow([task_id, iteration, float(loss.detach().item())])
                loss_csv_file.flush()

        if loss_csv_file is not None:
            loss_csv_file.close()

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

