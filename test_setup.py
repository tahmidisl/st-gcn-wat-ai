#!/usr/bin/env python
"""
Test script to verify ST-GCN setup and dependencies
"""
import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import torchvision  
        print(f"✓ TorchVision {torchvision.__version__}")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        
        import yaml
        print("✓ PyYAML")
        
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        
        import h5py
        print(f"✓ h5py {h5py.__version__}")
        
        print("\n✓ All dependencies imported successfully!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
            print(f"✓ CUDA devices: {torch.cuda.device_count()}")
        else:
            print("⚠ CUDA not available (CPU only)")
        return True
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False

def test_model_structure():
    """Test if model can be loaded"""
    print("\nTesting model structure...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(__file__))
        
        from net.st_gcn import Model
        
        # Test model creation
        model = Model(
            in_channels=2,
            num_class=26,
            graph_args={'layout': 'openpose', 'strategy': 'spatial'}
        )
        
        print("✓ ST-GCN model loaded successfully")
        
        # Test forward pass with dummy data
        import torch
        x = torch.randn(1, 2, 64, 18, 1)  # Batch, channels, time, joints, persons
        output = model(x)
        print(f"✓ Forward pass successful: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_config():
    """Test if config file exists and is valid"""
    print("\nTesting configuration...")
    
    config_path = "data/BOLD/train_categorical.yaml"
    if os.path.exists(config_path):
        print(f"✓ Config file found: {config_path}")
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            print("✓ Config file is valid YAML")
            return True
        except Exception as e:
            print(f"✗ Config file error: {e}")
            return False
    else:
        print(f"⚠ Config file not found: {config_path}")
        print("  Run data preprocessing first!")
        return False

def main():
    print("ST-GCN Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_cuda, 
        test_model_structure,
        test_config
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✓ Setup is ready for training!")
    else:
        print("⚠ Some tests failed. Check dependencies and setup.")
        
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 
