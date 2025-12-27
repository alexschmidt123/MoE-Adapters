from collections import OrderedDict
from typing import Tuple, Union, List, Optional, List, Optional

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .adapter import Adapter
from torch.distributions.normal import Normal
from collections import Counter

global_taskid = 0
global_is_train=True


def generate_geometric_capacities(num_experts: int, base_capacity: int = 16, ratio: float = 2.0) -> List[int]:
    """
    Generate expert capacities using geometric progression strategy.
    
    Args:
        num_experts: Number of experts
        base_capacity: Starting capacity for the smallest expert (default: 16)
        ratio: Geometric ratio between consecutive experts (default: 2.0)
               For ratio=2.0: [16, 32, 64, 128, ...]
    
    Returns:
        List of capacities following geometric progression
    """
    capacities = [int(base_capacity * (ratio ** i)) for i in range(num_experts)]
    return capacities


def generate_arithmetic_capacities(num_experts: int, base_capacity: int = 32, step: int = 16) -> List[int]:
    """
    Generate expert capacities using arithmetic progression strategy.
    More balanced than geometric, with smaller gaps between experts.
    
    Args:
        num_experts: Number of experts
        base_capacity: Starting capacity for the smallest expert (default: 32)
        step: Step size between consecutive experts (default: 16)
              For step=16: [32, 48, 64, 80, ...]
    
    Returns:
        List of capacities following arithmetic progression
    """
    capacities = [int(base_capacity + step * i) for i in range(num_experts)]
    return capacities


def generate_hybrid_capacities(num_experts: int, base_capacity: int = 32) -> List[int]:
    """
    Generate expert capacities using hybrid strategy.
    Combines homogeneous and heterogeneous elements for balanced distribution.
    
    Args:
        num_experts: Number of experts
        base_capacity: Base capacity for smaller experts (default: 32)
                      Creates pattern like [32, 32, 48, 64] for 4 experts
    
    Returns:
        List of capacities following hybrid progression
    """
    if num_experts <= 2:
        return [base_capacity] * num_experts
    
    # Create hybrid: some small (homogeneous), then increasing (heterogeneous)
    num_small = max(1, num_experts // 2)
    capacities = [base_capacity] * num_small
    
    # Add increasing capacities for remaining experts
    for i in range(num_experts - num_small):
        capacities.append(int(base_capacity * (1.5 ** (i + 1))))
    
    return capacities
class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts

        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)

        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space

        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)  # 加权

        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), device=stitched.device)
        # combine samples that have been processed by the same k experts

        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        # back to log space
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, text_or_image=None, cfg=None):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.is_train = global_is_train
        self.step = 1
        
        # Read MoE config (backward compatible with defaults)
        if cfg is not None and hasattr(cfg, 'model'):
            self.experts_num = getattr(cfg.model, 'num_experts', 2)
            self.top_k = getattr(cfg.model, 'top_k', 2)
            # Get expert hidden layers for deeper adapters (None = standard 2-layer)
            expert_hidden_layers = getattr(cfg.model, 'expert_hidden_layers', None)
            if expert_hidden_layers is not None and isinstance(expert_hidden_layers, str):
                if expert_hidden_layers.lower() == 'none':
                    expert_hidden_layers = None
            if expert_hidden_layers is not None and hasattr(expert_hidden_layers, '__iter__') and not isinstance(expert_hidden_layers, str):
                self.expert_hidden_layers = list(expert_hidden_layers)
            else:
                self.expert_hidden_layers = None
            
            # HMoE: Heterogeneous Mixture of Experts configuration
            # If enabled, experts can have different capacities (bottleneck sizes)
            self.hmoe_enabled = getattr(cfg.model, 'hmoe_enabled', False)
            if self.hmoe_enabled:
                # Get capacity generation strategy
                hmoe_strategy = getattr(cfg.model, 'hmoe_strategy', None)
                if isinstance(hmoe_strategy, str):
                    hmoe_strategy = hmoe_strategy.lower()
                
                # Generate or get expert capacities
                if hmoe_strategy == 'geometric':
                    # Geometric strategy: capacities follow geometric progression
                    base_capacity = int(getattr(cfg.model, 'hmoe_base_capacity', 16))
                    geometric_ratio = float(getattr(cfg.model, 'hmoe_geometric_ratio', 2.0))
                    self.expert_capacities = generate_geometric_capacities(
                        self.experts_num, base_capacity, geometric_ratio
                    )
                    print(f"HMoE: Using geometric strategy with capacities: {self.expert_capacities} (base={base_capacity}, ratio={geometric_ratio})")
                elif hmoe_strategy == 'arithmetic':
                    # Arithmetic strategy: capacities follow arithmetic progression (more balanced)
                    base_capacity = int(getattr(cfg.model, 'hmoe_base_capacity', 32))
                    step = int(getattr(cfg.model, 'hmoe_arithmetic_step', 16))
                    self.expert_capacities = generate_arithmetic_capacities(
                        self.experts_num, base_capacity, step
                    )
                    print(f"HMoE: Using arithmetic strategy with capacities: {self.expert_capacities} (base={base_capacity}, step={step})")
                elif hmoe_strategy == 'hybrid':
                    # Hybrid strategy: combines homogeneous and heterogeneous elements
                    base_capacity = int(getattr(cfg.model, 'hmoe_base_capacity', 32))
                    self.expert_capacities = generate_hybrid_capacities(
                        self.experts_num, base_capacity
                    )
                    print(f"HMoE: Using hybrid strategy with capacities: {self.expert_capacities} (base={base_capacity})")
                else:
                    # Manual specification or None (defaults to homogeneous)
                    expert_capacities = getattr(cfg.model, 'hmoe_expert_capacities', None)
                    if expert_capacities is not None:
                        if isinstance(expert_capacities, str):
                            if expert_capacities.lower() == 'none':
                                expert_capacities = None
                        if expert_capacities is not None and hasattr(expert_capacities, '__iter__') and not isinstance(expert_capacities, str):
                            expert_capacities = list(expert_capacities)
                            # Ensure we have the right number of capacities
                            if len(expert_capacities) != self.experts_num:
                                raise ValueError(f"hmoe_expert_capacities must have length {self.experts_num}, got {len(expert_capacities)}")
                            self.expert_capacities = expert_capacities
                        else:
                            self.expert_capacities = None
                    else:
                        self.expert_capacities = None
                
                # P-Penalty loss weight (from HMoE paper)
                self.hmoe_balanced_activation_weight = float(getattr(cfg.model, 'hmoe_balanced_activation_weight', 0.0))
                # Entropy regularization on gating probabilities (encourages diverse expert usage)
                self.hmoe_gate_entropy_weight = float(getattr(cfg.model, 'hmoe_gate_entropy_weight', 0.0))
                # Load balancing loss weight (standard MoE load balancing)
                # Encourages uniform distribution of examples across experts
                self.hmoe_load_balance_weight = float(getattr(cfg.model, 'hmoe_load_balance_weight', 0.0))
            else:
                self.expert_capacities = None
                self.hmoe_balanced_activation_weight = 0.0
                self.hmoe_gate_entropy_weight = 0.0
                self.hmoe_load_balance_weight = 0.0
        else:
            self.experts_num = 2
            self.top_k = 2
            self.expert_hidden_layers = None
            self.hmoe_enabled = False
            self.expert_capacities = None
            self.hmoe_balanced_activation_weight = 0.0
            self.hmoe_gate_entropy_weight = 0.0
            self.hmoe_load_balance_weight = 0.0
            
        # Default bottleneck size (used for homogeneous experts or as fallback)
        self.ffn_num = 64  # Bottleneck size fixed at 64 for homogeneous case
        self.softmax = nn.Softmax(1)
        self.softplus = nn.Softplus()
        self.noisy_gating = True
        self.adaptmlp_list = nn.ModuleList()
        self.text_or_image = text_or_image
        if text_or_image == 'text':
            print('text transformer')
            self.choose_map_text = torch.zeros([self.experts_num])
        else:
            print('image transformer')
            self.choose_map_image = torch.zeros([self.experts_num])
        self.router_list = nn.ParameterList()
        self.w_noise_list = nn.ParameterList()
        for i in range(self.step):
            self.router_list.append(nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True))
            self.w_noise_list.append(nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True))
        
        # Create experts with potentially heterogeneous capacities
        for i in range(self.experts_num):
            # Determine bottleneck size for this expert
            if self.hmoe_enabled and self.expert_capacities is not None:
                expert_bottleneck = self.expert_capacities[i]
            else:
                expert_bottleneck = self.ffn_num  # Use default homogeneous size
            
            self.adaptmlp = Adapter(d_model=d_model, dropout=0.1, bottleneck=expert_bottleneck,
                                    init_option='lora',
                                    adapter_scalar=0.1,
                                    adapter_layernorm_option='none',
                                    hidden_layers=self.expert_hidden_layers,
                                    )
            self.adaptmlp_list.append(self.adaptmlp)

        # Graph-over-Experts (GoE) mixer configuration
        if cfg is not None and hasattr(cfg, 'model'):
            self.graph_enabled = getattr(cfg.model, 'graph_mixer_enabled', False)
            if self.graph_enabled:
                # Import here to avoid circular dependency issues
                # Use absolute import since script runs from cil/ directory
                import graph_mixer
                # Helper to get list from config (None if not present)
                def get_list_attr(cfg_obj, attr_name, default=None):
                    val = getattr(cfg_obj, attr_name, default)
                    if val is None or val == "None":
                        return None
                    # Convert OmegaConf list to Python list if needed
                    if hasattr(val, '__iter__') and not isinstance(val, str):
                        return list(val)
                    return val
                
                self.graph_mixer = graph_mixer.GraphExpertMixer(
                    d_model=d_model,
                    num_experts=self.experts_num,
                    symmetrize=getattr(cfg.model, 'graph_symmetrize', True),
                    add_self_loop=getattr(cfg.model, 'graph_add_self_loop', True),
                    use_noisy_adjacency=getattr(cfg.model, 'graph_use_noisy_adjacency', True),
                    noise_epsilon=float(getattr(cfg.model, 'graph_noise_epsilon', 0.01)),
                    # Header layer configurations
                    graph_head_layers=get_list_attr(cfg.model, 'graph_head_layers', None),
                    graph_head_activation=str(getattr(cfg.model, 'graph_head_activation', 'none')),
                    graph_head_use_norm=bool(getattr(cfg.model, 'graph_head_use_norm', False)),
                    graph_head_dropout=float(getattr(cfg.model, 'graph_head_dropout', 0.0)),
                    graph_noise_head_layers=get_list_attr(cfg.model, 'graph_noise_head_layers', None),
                    graph_noise_head_activation=str(getattr(cfg.model, 'graph_noise_head_activation', 'none')),
                    graph_noise_head_use_norm=bool(getattr(cfg.model, 'graph_noise_head_use_norm', False)),
                    graph_proto_layers=get_list_attr(cfg.model, 'graph_proto_layers', None),
                    graph_proto_activation=str(getattr(cfg.model, 'graph_proto_activation', 'none')),
                    graph_proto_use_norm=bool(getattr(cfg.model, 'graph_proto_use_norm', False)),
                    graph_proto_dropout=float(getattr(cfg.model, 'graph_proto_dropout', 0.0)),
                    graph_proj_layers=get_list_attr(cfg.model, 'graph_proj_layers', None),
                    graph_proj_activation=str(getattr(cfg.model, 'graph_proj_activation', 'none')),
                    graph_proj_use_norm=bool(getattr(cfg.model, 'graph_proj_use_norm', False)),
                )
                self.alpha_graph = nn.Parameter(
                    torch.tensor(float(getattr(cfg.model, 'graph_alpha_init', 0.0)))
                )
                self.graph_entropy_weight = float(getattr(cfg.model, 'graph_entropy_weight', 0.0))
            else:
                self.graph_mixer = None
                self.alpha_graph = None
                self.graph_entropy_weight = 0.0
        else:
            self.graph_enabled = False
            self.graph_mixer = None
            self.alpha_graph = None
            self.graph_entropy_weight = 0.0
        
        # Attribute to accumulate extra losses (e.g., entropy regularization)
        self.extra_losses = None
        
        # self.taskid = None
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)
    
    def compute_hmoe_balanced_activation_loss(self, gates, expert_capacities):
        """
        Compute P-Penalty loss for HMoE (from HMoE paper).
        Penalizes activation of larger experts proportionally to their size.
        
        Formula: L_P-Penalty = N * Σ(M_i * P̂_i)
        where:
            N = number of experts
            M_i = capacity/size of expert i (bottleneck dimension)
            P̂_i = average gating probability for expert i (mean over batch)
        
        This encourages smaller experts to be activated more frequently.
        
        Args:
            gates: Gate values [B, num_experts] indicating expert selection weights
            expert_capacities: List of expert capacities (bottleneck sizes) [num_experts]
            
        Returns:
            loss: Scalar loss value (P-Penalty)
        """
        if not self.hmoe_enabled or self.hmoe_balanced_activation_weight <= 0:
            return torch.tensor(0.0, device=gates.device, dtype=gates.dtype)
        
        N = len(expert_capacities)  # Number of experts
        M = torch.tensor(expert_capacities, device=gates.device, dtype=gates.dtype)  # [num_experts] - expert capacities
        
        # Compute average gating probability per expert (P̂_i)
        # P̂_i = mean of gates over batch dimension
        P_hat = gates.mean(dim=0)  # [num_experts] - average probability for each expert
        
        # P-Penalty loss: L = N * Σ(M_i * P̂_i)
        # This penalizes larger experts (higher M_i) when they have high activation (high P̂_i)
        p_penalty = N * (M * P_hat).sum()
        
        return p_penalty

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        # print('1231',clean_values)  # 全nan
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        #

        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, w_gate, w_noise, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        clean_logits = x @ w_gate.to(x)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ w_noise.to(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.experts_num), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.noisy_gating and self.top_k < self.experts_num and train:  # 目前未用上
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        if global_taskid is not None:
            # x shape: [L, B, D] where L is sequence length, B is batch size, D is model dim
            x_re = x.permute(1, 0, 2)[:, 0, :]  # Use CLS token: [B, D]
            gates, load = self.noisy_top_k_gating(x_re, self.is_train, self.router_list[global_taskid],
                                                  self.w_noise_list[global_taskid])
            importance = gates.sum(0)

            nonzero_indices = torch.nonzero(gates)
            counter = Counter(nonzero_indices[:, 1].tolist())
            for number, count in counter.items():
                if self.text_or_image == 'text':
                    self.choose_map_text[number] = self.choose_map_text[number] + count
                else:
                    self.choose_map_image[number] = self.choose_map_image[number] + count
            dispatcher = SparseDispatcher(self.experts_num, gates)
            expert_inputs = dispatcher.dispatch(x.permute(1, 0, 2).view(x.shape[1], -1))
            expert_outputs = [self.adaptmlp_list[i](expert_inputs[i].view(expert_inputs[i].shape[0],
                                                                          x.shape[0], x.shape[2]).to(x), add_residual=False)
                              for i in range(self.experts_num)]

            i = 0
            while i < len(expert_outputs):
                if expert_outputs[i].shape[0] == 0:
                    expert_outputs.pop(i)
                else:
                    expert_outputs[i] = expert_outputs[i].view(expert_outputs[i].shape[0], -1)
                    i += 1

            # Standard MoE combine
            y_moe = dispatcher.combine(expert_outputs)
            y_moe = y_moe.view(x.shape[1], x.shape[0], x.shape[2])  # [B, L, D]
            
            # HMoE: Compute balanced activation loss (if enabled)
            if self.hmoe_enabled and self.is_train:
                # P-Penalty loss
                if self.hmoe_balanced_activation_weight > 0:
                    hmoe_loss = self.compute_hmoe_balanced_activation_loss(gates, self.expert_capacities)
                    # Accumulate extra loss (will be added in training loop)
                    if self.extra_losses is None:
                        self.extra_losses = self.hmoe_balanced_activation_weight * hmoe_loss
                    else:
                        self.extra_losses = self.extra_losses + self.hmoe_balanced_activation_weight * hmoe_loss
                
                # Entropy regularization on gating probabilities (encourages diverse expert usage)
                if self.hmoe_gate_entropy_weight > 0:
                    # Compute entropy: -sum(p * log(p)) per sample, averaged
                    # Higher entropy = more uniform distribution = more diverse expert usage
                    p = gates.clamp_min(1e-9)  # Avoid log(0)
                    gate_entropy = -(p * p.log()).sum(dim=-1).mean()  # [B] -> scalar
                    # We want to maximize entropy (diversity), so minimize negative entropy
                    entropy_loss = -gate_entropy
                    # Accumulate extra loss
                    if self.extra_losses is None:
                        self.extra_losses = self.hmoe_gate_entropy_weight * entropy_loss
                    else:
                        self.extra_losses = self.extra_losses + self.hmoe_gate_entropy_weight * entropy_loss
                
                # Load balancing loss (standard MoE load balancing)
                # Encourages uniform distribution of examples across experts
                if self.hmoe_load_balance_weight > 0:
                    # importance = sum of gate values per expert (measures total expert usage)
                    # load = number of examples routed to each expert
                    importance = gates.sum(0)  # [num_experts]
                    load_balance_loss = self.cv_squared(importance) + self.cv_squared(load)
                    # Accumulate extra loss
                    if self.extra_losses is None:
                        self.extra_losses = self.hmoe_load_balance_weight * load_balance_loss
                    else:
                        self.extra_losses = self.extra_losses + self.hmoe_load_balance_weight * load_balance_loss
            
            # Graph-over-Experts path (if enabled)
            if self.graph_enabled and self.graph_mixer is not None:
                # Compute graph mixing using the same pooled representation
                # Pass is_train flag for noisy adjacency (noise only during training)
                A, X_all, Y_all = self.graph_mixer(x_re, is_train=self.is_train)  # A: [B,N,N], X_all: [B,N,D], Y_all: [B,N,D]
                
                # Fuse graph messages using the same gates from router
                # y_graph[b] = sum_e gates[b,e] * Y_all[b,e]
                y_graph = torch.einsum("bn,bnd->bd", gates, Y_all)  # [B, D]
                
                # Broadcast over sequence length
                y_graph = y_graph.unsqueeze(1).expand(-1, x.shape[0], -1)  # [B, L, D]
                
                # Add graph contribution with learnable weight
                y_fused = y_moe + self.alpha_graph * y_graph
                
                # Optional entropy regularization on adjacency matrix
                if self.graph_entropy_weight > 0 and self.is_train:
                    # Row entropy: -sum(p * log(p)) per row, averaged over batch and rows
                    p = A.clamp_min(1e-9)
                    row_entropy = -(p * p.log()).sum(dim=-1).mean()
                    # Accumulate extra loss (will be added in training loop)
                    if self.extra_losses is None:
                        self.extra_losses = self.graph_entropy_weight * row_entropy
                    else:
                        self.extra_losses = self.extra_losses + self.graph_entropy_weight * row_entropy
            else:
                y_fused = y_moe
            
            x = x + self.mlp(self.ln_2(x)) + y_fused.permute(1, 0, 2)
        else:
            x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, text_or_image=None, cfg=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, text_or_image, cfg) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, text_or_image=None, cfg=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        # Added so this info is available. should not change anything.
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, text_or_image=text_or_image, cfg=cfg)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 baseline = False,
                 cfg = None
                 ):
        super().__init__()
        self.baseline = baseline
        self.cfg = cfg

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                text_or_image='image',
                cfg=cfg
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            text_or_image='text',
            cfg=cfg
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text, taskid, is_train):
        global global_taskid, global_is_train
        global_taskid = taskid
        global_is_train = is_train
        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # if self.baseline:
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text
        


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, cfg=None):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        cfg=cfg
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    for p in model.parameters():
        p.data = p.data.float()
    return model.eval()