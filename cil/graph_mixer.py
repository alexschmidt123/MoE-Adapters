"""
Graph-over-Experts (GoE) Mixer Module

This module implements a lightweight graph-based mixing mechanism for MoE systems.
It allows inactive experts to influence the final output via learned adjacency matrix
and message passing over expert proto-features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GraphExpertMixer(nn.Module):
    """
    Graph-over-Experts Mixer that learns per-sample expert adjacency and performs
    message passing to let inactive experts influence the output.
    
    Args:
        d_model (int): Dimension of the model (feature dimension)
        num_experts (int): Number of experts in the MoE system
        symmetrize (bool): If True, symmetrize the adjacency matrix A = (A + A^T) / 2
        add_self_loop (bool): If True, add identity to adjacency before normalization
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        symmetrize: bool = True,
        add_self_loop: bool = True,
        use_noisy_adjacency: bool = True,
        noise_epsilon: float = 0.01
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.symmetrize = symmetrize
        self.add_self_loop = add_self_loop
        self.use_noisy_adjacency = use_noisy_adjacency
        self.noise_epsilon = noise_epsilon
        
        # Adjacency matrix predictor: maps pooled token to N*N logits
        self.A_head = nn.Linear(d_model, num_experts * num_experts)
        
        # Noise head for adjacency: learns sample-specific noise variance
        if self.use_noisy_adjacency:
            self.adj_noise_head = nn.Linear(d_model, num_experts * num_experts)
            self.softplus = nn.Softplus()
        
        # Per-expert proto-feature generators (lightweight, no heavy adapters)
        self.proto = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) 
            for _ in range(num_experts)
        ])
        
        # Graph message projection and activation
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.act = nn.GELU()
        
    def forward(
        self, 
        x_sample: torch.Tensor,
        is_train: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Graph Expert Mixer.
        
        Args:
            x_sample: Pooled sample representation [B, D]
            is_train: Whether in training mode (affects noise injection)
            
        Returns:
            A: Adjacency matrix [B, N, N] (row-stochastic after softmax)
            X_all: Proto-features for all experts [B, N, D]
            Y_all: Graph-mixed expert features [B, N, D]
        """
        B = x_sample.shape[0]
        N = self.num_experts
        D = self.d_model
        
        # 1. Predict adjacency logits [B, N*N] -> [B, N, N]
        A_logits = self.A_head(x_sample).view(B, N, N)
        
        # 1.5. Add noise to adjacency logits for exploration (only during training)
        if self.use_noisy_adjacency and is_train:
            # Predict sample-specific noise variance
            noise_logits = self.adj_noise_head(x_sample).view(B, N, N)  # [B, N, N]
            # Apply softplus to ensure positive noise variance, add epsilon for stability
            noise_stddev = self.softplus(noise_logits) + self.noise_epsilon  # [B, N, N]
            # Add Gaussian noise to adjacency logits
            noise = torch.randn_like(A_logits) * noise_stddev
            A_logits = A_logits + noise
        
        # 2. Optional symmetrization
        if self.symmetrize:
            A_logits = (A_logits + A_logits.transpose(-2, -1)) / 2.0
        
        # 3. Optional self-loop addition (before softmax)
        if self.add_self_loop:
            # Add identity matrix to logits (encourage self-connection)
            eye = torch.eye(N, device=A_logits.device, dtype=A_logits.dtype)
            A_logits = A_logits + eye.unsqueeze(0)  # [B, N, N]
        
        # 4. Row-wise softmax to get row-stochastic adjacency
        A = F.softmax(A_logits, dim=-1)  # [B, N, N]
        
        # 5. Create proto-features X_all [B, N, D]
        # Stack outputs from lightweight per-expert linears
        X_all = torch.stack(
            [self.proto[i](x_sample) for i in range(N)],
            dim=1
        )  # [B, N, D]
        
        # 6. Message passing: Y_all = proj(act(A @ X_all))
        # A @ X_all: [B, N, N] @ [B, N, D] -> [B, N, D]
        messages = torch.bmm(A, X_all)  # [B, N, D]
        messages = self.act(messages)
        Y_all = self.proj(messages)  # [B, N, D]
        
        return A, X_all, Y_all

