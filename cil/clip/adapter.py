# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in",
                 hidden_layers=None):
        """
        Args:
            d_model: Input/output dimension
            bottleneck: Bottleneck dimension (default 64, kept fixed)
            dropout: Dropout probability
            init_option: Initialization method
            adapter_scalar: Scaling factor for adapter output
            adapter_layernorm_option: LayerNorm placement ('in', 'out', 'none')
            hidden_layers: List of hidden layer dimensions for deeper adapters.
                          If None, uses standard 2-layer adapter (down -> up).
                          All hidden layers use the same size as bottleneck (64).
                          Example: [64] creates 3-layer adapter: down -> hidden -> up
        """
        super().__init__()
        self.n_embd = d_model if d_model is None else d_model
        self.down_size = bottleneck if bottleneck is not None else 64

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.dropout = dropout
        self.non_linear_func = nn.ReLU()
        
        # Build adapter layers
        if hidden_layers is None or len(hidden_layers) == 0:
            # Standard 2-layer adapter: down -> up
            self.down_proj = nn.Linear(self.n_embd, self.down_size)
            self.up_proj = nn.Linear(self.down_size, self.n_embd)
            self.hidden_layers = None
        else:
            # Deeper adapter: down -> hidden1 -> hidden2 -> ... -> up
            # All hidden layers use bottleneck size (64)
            self.down_proj = nn.Linear(self.n_embd, self.down_size)
            self.hidden_layers = nn.ModuleList()
            for hidden_dim in hidden_layers:
                # Use bottleneck size for all hidden layers
                self.hidden_layers.append(nn.Linear(self.down_size, self.down_size))
            self.up_proj = nn.Linear(self.down_size, self.n_embd)
        
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
                if self.hidden_layers is not None:
                    for hidden_layer in self.hidden_layers:
                        nn.init.kaiming_uniform_(hidden_layer.weight, a=math.sqrt(5))
                        nn.init.zeros_(hidden_layer.bias)

    def forward(self, x, add_residual=True, residual=None):

        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in': #  none
            x = self.adapter_layer_norm_before(x)

        # Down projection
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        
        # Hidden layers (if any)
        if self.hidden_layers is not None:
            for hidden_layer in self.hidden_layers:
                down = hidden_layer(down)
                down = self.non_linear_func(down)
                down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        
        # Up projection
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out': #  none
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up
        return output