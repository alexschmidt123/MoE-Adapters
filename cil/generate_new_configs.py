#!/usr/bin/env python3
"""Generate 64 config files for new GNN structure optimization"""

from pathlib import Path

# Define the 4 options for each parameter
num_layers_options = [1, 2, 3, 4]
hidden_dim_options = [256, 512, 768, 1024]
head_layers_options = [
    None,           # Single layer
    [256],          # 1 hidden layer
    [512, 256],     # 2 hidden layers
    [256, 128]      # 2 hidden layers (smaller)
]

# Create config directory
config_dir = Path('configs/class/new_cifar_configs')
config_dir.mkdir(parents=True, exist_ok=True)

# Generate all 64 configs
configs_created = []
for num_layers in num_layers_options:
    for hidden_dim in hidden_dim_options:
        for head_layers in head_layers_options:
            # Create method name and filename
            if head_layers is None:
                head_str = 'None'
                head_yaml = 'null'
            else:
                head_str = '_'.join(map(str, head_layers))
                head_yaml = '[' + ', '.join(map(str, head_layers)) + ']'
            
            method_name = f"MoE-Adapters-N4-GoE-L{num_layers}-H{hidden_dim}-Head{head_str}"
            filename = f"cifar100_2-2-MoE-Adapters-N4-GoE-L{num_layers}-H{hidden_dim}-Head{head_str}.yaml"
            
            # Create config content
            config_content = f"""# Configuration for CIFAR-100: 2-2 scenario, N=4, New GNN Structure
# graph_num_layers={num_layers}, graph_hidden_dim={hidden_dim}, graph_head_layers={head_layers}

defaults:
  - ../cifar100_2-2-MoE-Adapters-N2
  - _self_

# Override method name to create distinct output directory
method: "{method_name}"

# Dataset configuration
dataset_root: ""
class_order: ""
workdir: ""
scenario: "class"
prompt_template: "a bad photo of a {{}}."
log_path: "metrics.json"
model_name: "ViT-B/16"
dataset: "cifar100"
batch_size: 64
increment: ${{initial_increment}}
initial_increment: 2
weight_decay: 0.0
l2: 0
ce_method: 0
lr: 1e-3
ls: 0.0

# Model configuration
model:
  # MoE parameters
  num_experts: 4           # Number of experts (N=4)
  top_k: 2                 # Keep k=2 (standard sparse MoE)
  
  # Standard MoE (no HMoE)
  hmoe_enabled: false
  
  # Graph mixer settings
  graph_mixer_enabled: true       # Enable Graph-over-Experts mixer
  graph_symmetrize: true          # Symmetrize adjacency matrix: A = (A + A^T) / 2
  graph_add_self_loop: true       # Add self-loops before row normalization
  graph_alpha_init: 0.0           # Initial value for graph fusion weight
  graph_entropy_weight: 0.0       # Entropy regularization weight (0 = disabled)
  
  # New GNN structure parameters
  graph_num_layers: {num_layers}          # GNN message passing depth
  graph_hidden_dim: {hidden_dim}         # Hidden dimension for GNN
  graph_head_layers: {head_yaml}  # Adjacency predictor depth
  graph_layer_norm: true                  # Use LayerNorm
  graph_residual: true                     # Use residual connections
  graph_activation: "gelu"                # Activation function
  graph_dropout: 0.0                      # Dropout rate
"""
            
            # Save config
            config_path = config_dir / filename
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            configs_created.append(filename)

print(f"Created {len(configs_created)} config files in {config_dir}")
