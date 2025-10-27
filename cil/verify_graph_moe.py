#!/usr/bin/env python
"""
Verification script for Graph-over-Experts MoE refactoring.

This script checks that all components are properly installed and configured.
Run this before starting experiments to ensure everything is working correctly.

Usage:
    python verify_graph_moe.py
"""

import sys
import os
from pathlib import Path


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_file_exists(filepath, description):
    """Check if a file exists and print result."""
    exists = Path(filepath).exists()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists


def check_import(module_name, description):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {description}: {module_name} - {e}")
        return False


def verify_files():
    """Verify all required files exist."""
    print_section("File Verification")
    
    files_to_check = [
        ("continual_clip/graph_mixer.py", "Graph mixer module"),
        ("configs/experts.yaml", "Expert config"),
        ("run_cifar100-2-2-MoE-GNN.sh", "Example run script"),
        ("GRAPH_MOE_README.md", "Documentation"),
        ("REFACTORING_SUMMARY.md", "Refactoring summary"),
        ("clip/model.py", "Modified model"),
        ("clip/clip.py", "Modified CLIP loader"),
        ("continual_clip/models.py", "Modified training"),
        ("main.py", "Modified main script"),
        ("configs/class/cifar100_2-2-MoE-Adapters.yaml", "Updated config"),
    ]
    
    results = [check_file_exists(f, desc) for f, desc in files_to_check]
    return all(results)


def verify_imports():
    """Verify required Python packages."""
    print_section("Import Verification")
    
    imports_to_check = [
        ("torch", "PyTorch"),
        ("omegaconf", "OmegaConf"),
        ("hydra", "Hydra"),
    ]
    
    results = [check_import(module, desc) for module, desc in imports_to_check]
    return all(results)


def verify_graph_mixer():
    """Verify graph mixer module structure."""
    print_section("Graph Mixer Module Check")
    
    try:
        from continual_clip.graph_mixer import GraphExpertMixer
        print("✓ GraphExpertMixer class imported successfully")
        
        # Check required methods
        required_methods = ['forward', 'compute_entropy_loss', 'get_adjacency_stats']
        for method in required_methods:
            if hasattr(GraphExpertMixer, method):
                print(f"✓ Method '{method}' exists")
            else:
                print(f"✗ Method '{method}' missing")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Error loading GraphExpertMixer: {e}")
        return False


def verify_config():
    """Verify expert configuration."""
    print_section("Configuration Check")
    
    try:
        from omegaconf import OmegaConf
        
        # Load experts config
        config_path = Path("configs/experts.yaml")
        if not config_path.exists():
            print("✗ experts.yaml not found")
            return False
        
        cfg = OmegaConf.load(config_path)
        print("✓ experts.yaml loaded successfully")
        
        # Check required fields
        required_fields = [
            'num_experts', 'top_k', 'graph_mixer_enabled',
            'graph_symmetrize', 'graph_add_self_loop',
            'graph_alpha_init', 'graph_entropy_weight'
        ]
        
        for field in required_fields:
            if field in cfg:
                print(f"✓ Config field '{field}' = {cfg[field]}")
            else:
                print(f"✗ Config field '{field}' missing")
                return False
        
        # Verify sensible defaults
        if cfg.num_experts < 1:
            print(f"✗ Invalid num_experts: {cfg.num_experts} (should be >= 1)")
            return False
        
        if cfg.top_k < 1 or cfg.top_k > cfg.num_experts:
            print(f"✗ Invalid top_k: {cfg.top_k} (should be 1 <= k <= N)")
            return False
        
        print("✓ Configuration values are valid")
        return True
        
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False


def verify_model_integration():
    """Verify model integration (shallow check without loading weights)."""
    print_section("Model Integration Check")
    
    try:
        # Check if cfg parameter was added to key classes
        import inspect
        from clip.model import ResidualAttentionBlock, Transformer, VisualTransformer, CLIP
        
        classes_to_check = [
            (ResidualAttentionBlock, "ResidualAttentionBlock"),
            (Transformer, "Transformer"),
            (VisualTransformer, "VisualTransformer"),
            (CLIP, "CLIP"),
        ]
        
        for cls, name in classes_to_check:
            sig = inspect.signature(cls.__init__)
            if 'cfg' in sig.parameters:
                print(f"✓ {name}.__init__ has 'cfg' parameter")
            else:
                print(f"✗ {name}.__init__ missing 'cfg' parameter")
                return False
        
        # Check if ResidualAttentionBlock has graph mixer attributes
        init_source = inspect.getsource(ResidualAttentionBlock.__init__)
        if 'graph_mixer' in init_source and 'alpha_graph' in init_source:
            print("✓ ResidualAttentionBlock has graph mixer integration")
        else:
            print("✗ ResidualAttentionBlock missing graph mixer integration")
            return False
        
        # Check forward method for graph mixer logic
        forward_source = inspect.getsource(ResidualAttentionBlock.forward)
        if 'graph_mixer' in forward_source and 'y_graph' in forward_source:
            print("✓ ResidualAttentionBlock.forward has graph mixer logic")
        else:
            print("✗ ResidualAttentionBlock.forward missing graph mixer logic")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking model integration: {e}")
        return False


def verify_training_integration():
    """Verify training loop integration."""
    print_section("Training Integration Check")
    
    try:
        import inspect
        from continual_clip.models import ClassIncremental
        
        # Check if cfg is passed to clip.load
        init_source = inspect.getsource(ClassIncremental.__init__)
        if 'cfg=cfg' in init_source or 'cfg = cfg' in init_source:
            print("✓ ClassIncremental passes cfg to clip.load")
        else:
            print("✗ ClassIncremental doesn't pass cfg to clip.load")
            return False
        
        # Check if training loop handles extra losses
        train_source = inspect.getsource(ClassIncremental.train)
        if '_extra_losses' in train_source and 'loss_total' in train_source:
            print("✓ Training loop handles extra losses")
        else:
            print("✗ Training loop missing extra loss handling")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking training integration: {e}")
        return False


def run_all_checks():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("  Graph-over-Experts MoE Verification")
    print("="*60)
    
    checks = [
        ("Files", verify_files),
        ("Imports", verify_imports),
        ("Graph Mixer", verify_graph_mixer),
        ("Configuration", verify_config),
        ("Model Integration", verify_model_integration),
        ("Training Integration", verify_training_integration),
    ]
    
    results = {}
    for name, check_fn in checks:
        results[name] = check_fn()
    
    # Summary
    print_section("Verification Summary")
    
    all_passed = all(results.values())
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")
    
    if all_passed:
        print("\n" + "="*60)
        print("  ✓ All checks passed! System is ready.")
        print("="*60)
        print("\nYou can now run:")
        print("  bash run_cifar100-2-2-MoE-GNN.sh")
        print("\nOr test with different expert counts:")
        print("  EXPERTS_N=8 bash run_cifar100-2-2-MoE-GNN.sh")
        return 0
    else:
        print("\n" + "="*60)
        print("  ✗ Some checks failed. Please review errors above.")
        print("="*60)
        print("\nRefer to GRAPH_MOE_README.md for setup instructions.")
        return 1


if __name__ == "__main__":
    # Change to cil directory if not already there
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    exit_code = run_all_checks()
    sys.exit(exit_code)

