"""
Proper Graph Neural Network for MoE/HMoE Systems

Implements the correct structure: Input → GNN → Router → Experts → Output
Always uses all N experts in the graph (same workflow for any N).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Optional[List[int]] = None,
    activation: str = "gelu",
    use_norm: bool = False,
    dropout: float = 0.0,
    bias: bool = True
) -> nn.Module:
    """Build a multi-layer perceptron (MLP)."""
    layers = []
    dims = [input_dim]
    if hidden_dims:
        dims.extend(hidden_dims)
    dims.append(output_dim)
    
    if activation.lower() == "gelu":
        act_fn = nn.GELU()
    elif activation.lower() == "relu":
        act_fn = nn.ReLU()
    elif activation.lower() == "tanh":
        act_fn = nn.Tanh()
    elif activation.lower() == "none":
        act_fn = nn.Identity()
    else:
        raise ValueError(f"Unknown activation: {activation}")
    
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
        if i < len(dims) - 2:
            if use_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    
    return nn.Sequential(*layers)


class ProperGraphExpertMixer(nn.Module):
    """
    Proper GNN structure: Processes input first, outputs GNN-enhanced representation.
    
    Structure: Input → GNN (all N experts) → (output used by Router → Experts)
    Same workflow for any N; no coarse router.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_layers: int = 2,
        hidden_dim: Optional[int] = None,
        symmetrize: bool = True,
        add_self_loop: bool = True,
        activation: str = "gelu",
        dropout: float = 0.0,
        layer_norm: bool = True,
        residual: bool = True,
        graph_head_layers: Optional[List[int]] = None,
        graph_proto_layers: Optional[List[int]] = None,
        graph_proj_layers: Optional[List[int]] = None,
        identity_bias_alpha: float = 0.0,
        adj_top_m: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim or d_model
        self.symmetrize = symmetrize
        self.add_self_loop = add_self_loop
        self.residual = residual
        # Change B: identity-biased adjacency A = alpha*I + (1-alpha)*softmax(...); alpha 1 = no mixing
        self.identity_bias_alpha = float(identity_bias_alpha)
        # Change J: sparsify adjacency to top-m edges per node (None = dense)
        self.adj_top_m = adj_top_m
        
        # Node feature embeddings (one per expert)
        # These represent expert-specific features
        self.expert_embeddings = nn.Parameter(
            torch.randn(num_experts, self.hidden_dim) * 0.02
        )
        
        # Input projection to hidden dimension
        self.input_proj = nn.Linear(d_model, self.hidden_dim)
        
        # Adjacency predictor: learns expert-to-expert relationships
        self.adjacency_head = build_mlp(
            input_dim=d_model,
            output_dim=num_experts * num_experts,
            hidden_dims=graph_head_layers,
            activation=activation,
            use_norm=layer_norm,
            dropout=dropout,
            bias=True
        )
        
        # GNN layers (Graph Convolution)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(
                GraphConvLayer(
                    in_dim=self.hidden_dim if i == 0 else self.hidden_dim,
                    out_dim=self.hidden_dim,
                    activation=activation,
                    dropout=dropout if i < num_layers - 1 else 0.0,
                    layer_norm=layer_norm,
                    residual=residual and i > 0
                )
            )
        
        # Output projection (back to d_model)
        self.output_proj = nn.Linear(self.hidden_dim, d_model)
        # Zero-init so at start x_gnn ≈ 0 → router gets uniform input → GoE starts close to MoE baseline
        nn.init.zeros_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
        
        # Activation
        if activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
    
    def forward(
        self,
        x_sample: torch.Tensor,
        is_train: bool = True,
        return_per_expert: bool = False
    ):
        """
        Forward pass: Processes input and returns GNN-enhanced representation.
        
        Args:
            x_sample: Pooled input representation [B, D]
            is_train: Whether in training mode
            return_per_expert: If True, return (x_gnn, Y) with Y [B, N, H] for per-expert routing.
            
        Returns:
            If return_per_expert False: x_gnn [B, D]
            If return_per_expert True: (x_gnn [B, D], Y [B, N, hidden_dim])
        """
        B = x_sample.shape[0]
        N = self.num_experts
        
        # 1. Build node features (all N experts)
        x_proj = self.input_proj(x_sample)  # [B, hidden_dim]
        node_features = x_proj.unsqueeze(1).expand(-1, N, -1) + self.expert_embeddings.unsqueeze(0)  # [B, N, hidden_dim]
        
        # 2. Build adjacency matrix [B, N, N]
        A_logits = self.adjacency_head(x_sample).view(B, N, N)
        if self.symmetrize:
            A_logits = (A_logits + A_logits.transpose(-2, -1)) / 2.0
        if self.add_self_loop:
            eye = torch.eye(N, device=A_logits.device, dtype=A_logits.dtype)
            A_logits = A_logits + eye.unsqueeze(0)
        A = F.softmax(A_logits, dim=-1)  # row-stochastic

        # Change J: keep only top-m edges per node, renormalize (less oversmoothing)
        if self.adj_top_m is not None and self.adj_top_m < N:
            m = min(self.adj_top_m, N)
            top_vals, top_idx = A.topk(m, dim=-1)
            A_sparse = torch.zeros_like(A)
            A_sparse.scatter_(-1, top_idx, top_vals)
            A = A_sparse / (A_sparse.sum(dim=-1, keepdim=True).clamp_min(1e-9))

        # Change B: identity-biased A = alpha*I + (1-alpha)*A (avoid harmful mixing early)
        if self.identity_bias_alpha > 0:
            eye = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
            A = self.identity_bias_alpha * eye + (1.0 - self.identity_bias_alpha) * A

        # 3. Multi-layer message passing
        Y = node_features  # [B, N, hidden_dim]
        for i, gnn_layer in enumerate(self.gnn_layers):
            Y_new = gnn_layer(Y, A)
            if self.residual and i > 0:
                Y = Y + Y_new
            else:
                Y = Y_new
        
        # 4. Aggregate node features (mean over experts) for backward-compat x_gnn
        Y_agg = Y.mean(dim=1)  # [B, hidden_dim]
        
        # 5. Project back to d_model
        x_gnn = self.output_proj(Y_agg)  # [B, D]
        
        if return_per_expert:
            return (x_gnn, Y)
        return x_gnn


class GraphConvLayer(nn.Module):
    """
    Graph Convolution Layer for message passing.
    
    Implements: Y = σ(A @ X @ W + b)
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        layer_norm: bool = True,
        residual: bool = False
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = residual and (in_dim == out_dim)
        
        # Linear transformation
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        
        # Normalization
        self.norm = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
        
        # Activation
        if activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "relu":
            self.act = nn.ReLU()
        elif activation.lower() == "tanh":
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of graph convolution.
        
        Args:
            X: Node features [B, N, in_dim]
            A: Adjacency matrix [B, N, N]
            
        Returns:
            Y: Updated node features [B, N, out_dim]
        """
        # Message passing: A @ X
        messages = torch.bmm(A, X)  # [B, N, in_dim]
        
        # Linear transformation
        Y = self.linear(messages)  # [B, N, out_dim]
        
        # Normalization
        Y = self.norm(Y)
        
        # Residual connection (if dimensions match)
        if self.residual:
            Y = Y + X
        
        # Activation
        Y = self.act(Y)
        
        # Dropout
        Y = self.dropout(Y)
        
        return Y
