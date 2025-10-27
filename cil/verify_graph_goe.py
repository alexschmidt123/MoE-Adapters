"""
Verification script for Graph-over-Experts (GoE) implementation.
This script tests that the graph mixer can be imported and instantiated correctly.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_graph_mixer_import():
    """Test that GraphExpertMixer can be imported."""
    try:
        from graph_mixer import GraphExpertMixer
        print("✓ GraphExpertMixer imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import GraphExpertMixer: {e}")
        return False

def test_graph_mixer_creation():
    """Test that GraphExpertMixer can be instantiated."""
    try:
        from graph_mixer import GraphExpertMixer
        
        d_model = 512
        num_experts = 4
        
        mixer = GraphExpertMixer(
            d_model=d_model,
            num_experts=num_experts,
            symmetrize=True,
            add_self_loop=True
        )
        
        print(f"✓ GraphExpertMixer created: d_model={d_model}, num_experts={num_experts}")
        return True
    except Exception as e:
        print(f"✗ Failed to create GraphExpertMixer: {e}")
        return False

def test_graph_mixer_forward():
    """Test that GraphExpertMixer forward pass works."""
    try:
        from graph_mixer import GraphExpertMixer
        
        batch_size = 8
        d_model = 512
        num_experts = 4
        
        mixer = GraphExpertMixer(
            d_model=d_model,
            num_experts=num_experts,
            symmetrize=True,
            add_self_loop=True
        )
        
        # Create dummy input
        x_sample = torch.randn(batch_size, d_model)
        
        # Forward pass
        A, X_all, Y_all = mixer(x_sample)
        
        # Check shapes
        assert A.shape == (batch_size, num_experts, num_experts), f"A shape mismatch: {A.shape}"
        assert X_all.shape == (batch_size, num_experts, d_model), f"X_all shape mismatch: {X_all.shape}"
        assert Y_all.shape == (batch_size, num_experts, d_model), f"Y_all shape mismatch: {Y_all.shape}"
        
        # Check row-stochastic property
        row_sums = A.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "A is not row-stochastic"
        
        print(f"✓ GraphExpertMixer forward pass successful")
        print(f"  - A shape: {A.shape} (row-stochastic: ✓)")
        print(f"  - X_all shape: {X_all.shape}")
        print(f"  - Y_all shape: {Y_all.shape}")
        return True
    except Exception as e:
        print(f"✗ GraphExpertMixer forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test that old config structure still works (cfg=None)."""
    try:
        from clip.model import ResidualAttentionBlock
        
        # Create ResidualAttentionBlock with cfg=None (old behavior)
        block = ResidualAttentionBlock(
            d_model=512,
            n_head=8,
            attn_mask=None,
            text_or_image='image',
            cfg=None
        )
        
        # Check that defaults are preserved
        assert block.experts_num == 2, "Default experts_num should be 2"
        assert block.top_k == 2, "Default top_k should be 2"
        assert block.graph_enabled == False, "Graph should be disabled by default"
        assert block.graph_mixer is None, "Graph mixer should be None by default"
        
        print("✓ Backward compatibility verified:")
        print(f"  - experts_num = {block.experts_num} (default)")
        print(f"  - top_k = {block.top_k} (default)")
        print(f"  - graph_enabled = {block.graph_enabled} (disabled)")
        return True
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Graph-over-Experts (GoE) Verification Tests")
    print("=" * 60)
    print()
    
    tests = [
        ("Import Test", test_graph_mixer_import),
        ("Creation Test", test_graph_mixer_creation),
        ("Forward Pass Test", test_graph_mixer_forward),
        ("Backward Compatibility Test", test_backward_compatibility),
    ]
    
    results = []
    for name, test_fn in tests:
        print(f"\n{name}:")
        print("-" * 40)
        success = test_fn()
        results.append((name, success))
        print()
    
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Graph-over-Experts implementation verified.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    exit(main())

