"""
Quick test script to verify the setup is working correctly.
"""

import torch
import sys
from pathlib import Path

print("="*70)
print("EZKL CNN Benchmark - Setup Test")
print("="*70)

# Test 1: PyTorch
print("\n1. Testing PyTorch...")
try:
    import torch
    print(f"   ✓ PyTorch version: {torch.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"   ✗ PyTorch test failed: {e}")
    sys.exit(1)

# Test 2: EZKL
print("\n2. Testing EZKL...")
try:
    import ezkl
    print(f"   ✓ EZKL imported successfully")
except Exception as e:
    print(f"   ✗ EZKL test failed: {e}")
    print("   Please install EZKL: https://docs.ezkl.xyz/")
    sys.exit(1)

# Test 3: Model creation
print("\n3. Testing model creation...")
try:
    from src.models import create_single_layer_model, create_composite_model
    from config.experiment_config import LAYER_CONFIGS, COMPOSITE_ARCHITECTURES
    
    # Test single layer
    model = create_single_layer_model('ReLU', LAYER_CONFIGS['ReLU'])
    dummy_input = torch.randn(*LAYER_CONFIGS['ReLU']['input_shape'])
    output = model(dummy_input)
    print(f"   ✓ Single layer model (ReLU): {dummy_input.shape} -> {output.shape}")
    
    # Test composite
    model = create_composite_model('CNN_ReLU', COMPOSITE_ARCHITECTURES['CNN_ReLU'])
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"   ✓ Composite model (CNN_ReLU): {dummy_input.shape} -> {output.shape}")
    
except Exception as e:
    print(f"   ✗ Model creation test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: ONNX export
print("\n4. Testing ONNX export...")
try:
    temp_dir = Path('./temp/test')
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    model = create_single_layer_model('ReLU', LAYER_CONFIGS['ReLU'])
    dummy_input = torch.randn(*LAYER_CONFIGS['ReLU']['input_shape'])
    onnx_path = temp_dir / 'test_model.onnx'
    
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=14
    )
    
    print(f"   ✓ ONNX export successful: {onnx_path}")
    
except Exception as e:
    print(f"   ✗ ONNX export test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Configuration
print("\n5. Testing experiment configuration...")
try:
    from config.experiment_config import (
        EXPERIMENT_COUNTS,
        LAYER_CONFIGS,
        SCALING_CONFIGS,
        COMPOSITE_ARCHITECTURES
    )
    
    print(f"   ✓ Core layers: {len(LAYER_CONFIGS)}")
    print(f"   ✓ Scaling configs: {len(SCALING_CONFIGS)}")
    print(f"   ✓ Composite architectures: {len(COMPOSITE_ARCHITECTURES)}")
    print(f"   ✓ Total experiments: {EXPERIMENT_COUNTS['total']}")
    
    assert EXPERIMENT_COUNTS['total'] == 40, "Expected 40 total experiments"
    print(f"   ✓ Experiment count verified: 40")
    
except Exception as e:
    print(f"   ✗ Configuration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ All tests passed! Setup is ready.")
print("="*70)
print("\nNext steps:")
print("  1. Run experiments: python src/run_experiments.py")
print("  2. Or use: ./run_benchmark.sh all")
print("  3. Analyze results: python src/analyze_results.py")
print("="*70 + "\n")
