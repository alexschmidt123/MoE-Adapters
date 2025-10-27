"""
Graph-over-Experts (GoE) Mixer Module

This module implements a graph-based mixer that allows inactive experts
to influence the final output through learned adjacency matrices.
"""

import torch
import torch.nn as nn
from typing import Tuple


class GraphExpertMixer(nn.Module):
    """
    Per-sample expert graph mixer.
    
    Learns adjacency A (N x N) and propagates messages from all experts
    (including inactive ones) to produce Y_all: [B, N, D].
    
    This allows experts not selected by top-k routing to still contribute
    to the final output through graph message passing.
    
    Args:
        d_model: Model dimension
        num_experts: Number of experts N
        symmetrize: Whether to symmetrize adjacency matrix
        add_self_loop: Whether to add self-loops to adjacency
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        symmetrize: bool = True,
        add_self_loop: bool = True
    ):
        super().__init__()
        self.N = num_experts
        self.d_model = d_model
        self.symm = symmetrize
        self.add_self = add_self_loop
        
        # Learnable parameters for graph structure and transformations
        self.A_head = nn.Linear(d_model, self.N * self.N)  # logits for adjacency A
        
        # Lightweight proto heads - one per expert (cheaper than running full experts)
        self.proto = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(self.N)
        ])
        
        self.proj = nn.Linear(d_model, d_model, bias=False)  # Wg projection
        self.act = nn.GELU()
    
    def forward(self, x_re: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through graph mixer.
        
        Args:
            x_re: [B, D] pooled token features used by router
            
        Returns:
            A:     [B, N, N] row-stochastic adjacency matrix
            X_all: [B, N, D] proto expert outputs
            Y_all: [B, N, D] graph-mixed expert outputs
        """
        B, D = x_re.shape
        
        # Generate proto expert outputs (lightweight, all N experts)
        X_all = torch.stack([h(x_re) for h in self.proto], dim=1)  # [B, N, D]
        
        # Compute adjacency matrix
        A_logits = self.A_head(x_re).view(B, self.N, self.N)
        
        # Optional: symmetrize adjacency
        if self.symm:
            A_logits = 0.5 * (A_logits + A_logits.transpose(1, 2))
        
        # Optional: add self-loops
        if self.add_self:
            eye = torch.eye(self.N, device=x_re.device, dtype=A_logits.dtype).unsqueeze(0)
            A_logits = A_logits + eye
        
        # Row-stochastic adjacency (each row sums to 1)
        A = torch.softmax(A_logits, dim=-1)
        
        # Graph message passing: propagate through adjacency
        Y_all = torch.matmul(A, X_all)  # [B, N, D]
        Y_all = self.act(Y_all)
        Y_all = self.proj(Y_all)
        
        return A, X_all, Y_all
    
    def compute_entropy_loss(self, A: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy regularization loss on adjacency matrix.
        
        Encourages diverse connectivity patterns by penalizing
        concentrated distributions.
        
        Args:
            A: [B, N, N] adjacency matrix
            
        Returns:
            Scalar entropy loss (negative mean row entropy)
        """
        A_clamped = A.clamp_min(1e-9)
        row_entropy = -(A_clamped * A_clamped.log()).sum(dim=-1).mean()
        return -row_entropy  # Negative because we want to maximize entropy
    
    def get_adjacency_stats(self, A: torch.Tensor) -> dict:
        """
        Compute diagnostic statistics for adjacency matrix.
        
        Args:
            A: [B, N, N] adjacency matrix
            
        Returns:
            Dictionary with mean diagonal, off-diagonal, and entropy
        """
        B, N, _ = A.shape
        
        # Extract diagonal and off-diagonal elements
        diag_mask = torch.eye(N, device=A.device, dtype=torch.bool).unsqueeze(0)
        diag_elements = A[diag_mask].view(B, N)
        off_diag_elements = A[~diag_mask].view(B, N * (N - 1))
        
        # Compute row entropy
        A_clamped = A.clamp_min(1e-9)
        row_entropy = -(A_clamped * A_clamped.log()).sum(dim=-1)
        
        return {
            'mean_diag': diag_elements.mean().item(),
            'mean_off_diag': off_diag_elements.mean().item(),
            'mean_entropy': row_entropy.mean().item(),
        }

