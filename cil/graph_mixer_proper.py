"""
Proper Graph Neural Network for MoE/HMoE Systems

Implements the correct structure: Input → GNN → Router → Experts → Output
With hybrid approach: Router before GNN only for large N (>= 8)
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
    
    Structure: Input → GNN → (output used by Router → Experts)
    
    For large N (>= 8), can optionally use a coarse router to select experts before GNN.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_layers: int = 2,
        hidden_dim: Optional[int] = None,
        use_coarse_router: bool = False,
        coarse_router_k: Optional[int] = None,
        symmetrize: bool = True,
        add_self_loop: bool = True,
        activation: str = "gelu",
        dropout: float = 0.0,
        layer_norm: bool = True,
        residual: bool = True,
        graph_head_layers: Optional[List[int]] = None,
        graph_proto_layers: Optional[List[int]] = None,
        graph_proj_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim or d_model
        self.use_coarse_router = use_coarse_router
        self.coarse_router_k = coarse_router_k or (num_experts // 2)  # Default: half of experts
        self.symmetrize = symmetrize
        self.add_self_loop = add_self_loop
        self.residual = residual
        
        # Coarse router (only used if use_coarse_router=True and N >= 8)
        if self.use_coarse_router and num_experts >= 8:
            self.coarse_router = nn.Linear(d_model, num_experts)
            self.coarse_softmax = nn.Softmax(dim=-1)
        else:
            self.coarse_router = None
            self.use_coarse_router = False  # Disable if N < 8
        
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
        is_train: bool = True
    ) -> torch.Tensor:
        """
        Forward pass: Processes input and returns GNN-enhanced representation.
        
        Args:
            x_sample: Pooled input representation [B, D]
            is_train: Whether in training mode
            
        Returns:
            x_gnn: GNN-processed representation [B, D] (to be used by router and experts)
        """
        B = x_sample.shape[0]
        N = self.num_experts
        D = self.d_model
        
        # Optional: Coarse router to select subset of experts (for large N)
        if self.use_coarse_router and self.coarse_router is not None:
            # Coarse selection: select top-k experts
            coarse_logits = self.coarse_router(x_sample)  # [B, N]
            coarse_gates = self.coarse_softmax(coarse_logits)  # [B, N]
            _, selected_indices = coarse_gates.topk(self.coarse_router_k, dim=1)  # [B, k]
            # Get unique selected expert indices across batch
            selected_experts_unique = selected_indices.unique().sort()[0]  # [k_unique]
            N_active = len(selected_experts_unique)
            selected_experts = selected_experts_unique.cpu().tolist()  # Convert to list for indexing
            selected_experts_tensor = selected_experts_unique  # Keep tensor version
        else:
            # Use all experts
            selected_experts = list(range(N))
            selected_experts_tensor = torch.arange(N, device=x_sample.device, dtype=torch.long)
            N_active = N
        
        # 1. Build node features
        # Combine input projection with expert embeddings
        x_proj = self.input_proj(x_sample)  # [B, hidden_dim]
        
        # Create node features: [B, N_active, hidden_dim]
        # Each node gets input projection + expert-specific embedding
        node_features = x_proj.unsqueeze(1).expand(-1, N_active, -1)  # [B, N_active, hidden_dim]
        expert_emb = self.expert_embeddings[selected_experts_tensor]  # [N_active, hidden_dim]
        node_features = node_features + expert_emb.unsqueeze(0)  # [B, N_active, hidden_dim]
        
        # 2. Build adjacency matrix (for selected experts)
        A_logits = self.adjacency_head(x_sample).view(B, N, N)  # [B, N, N]
        
        # Extract adjacency for selected experts
        A_logits = A_logits[:, selected_experts_tensor][:, :, selected_experts_tensor]  # [B, N_active, N_active]
        
        # Optional symmetrization
        if self.symmetrize:
            A_logits = (A_logits + A_logits.transpose(-2, -1)) / 2.0
        
        # Optional self-loops
        if self.add_self_loop:
            eye = torch.eye(N_active, device=A_logits.device, dtype=A_logits.dtype)
            A_logits = A_logits + eye.unsqueeze(0)  # [B, N_active, N_active]
        
        # Row-wise softmax to get row-stochastic adjacency
        A = F.softmax(A_logits, dim=-1)  # [B, N_active, N_active]
        
        # 3. Multi-layer message passing
        Y = node_features  # [B, N_active, hidden_dim]
        for i, gnn_layer in enumerate(self.gnn_layers):
            Y_new = gnn_layer(Y, A)  # [B, N_active, hidden_dim]
            if self.residual and i > 0:
                Y = Y + Y_new  # Residual connection
            else:
                Y = Y_new
        
        # 4. Aggregate node features (mean pooling over experts)
        Y_agg = Y.mean(dim=1)  # [B, hidden_dim]
        
        # 5. Project back to d_model
        x_gnn = self.output_proj(Y_agg)  # [B, D]
        
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
