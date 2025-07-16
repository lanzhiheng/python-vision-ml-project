#!/usr/bin/env python3
"""
Simple Test Script

This script performs basic tests without GUI components to verify
the core ML/CV functionality is working.
"""

import sys
import traceback


def test_basic_imports():
    """Test basic package imports."""
    print("Testing basic package imports...")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except Exception as e:
        print(f"✗ NumPy failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except Exception as e:
        print(f"✗ Pandas failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except Exception as e:
        print(f"✗ Matplotlib failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
    except Exception as e:
        print(f"✗ scikit-learn failed: {e}")
        return False
    
    return True


def test_opencv():
    """Test OpenCV functionality."""
    print("\\nTesting OpenCV...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[30:70, 30:70] = [255, 0, 0]  # Red square
        
        # Test basic operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        print(f"✓ OpenCV {cv2.__version__}")
        print(f"  - Original image shape: {test_image.shape}")
        print(f"  - Grayscale image shape: {gray.shape}")
        print(f"  - Blurred image shape: {blurred.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ OpenCV failed: {e}")
        return False


def test_pytorch():
    """Test PyTorch functionality."""
    print("\\nTesting PyTorch...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Test tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        
        # Test neural network
        model = nn.Linear(3, 2)
        test_input = torch.randn(1, 3)
        output = model(test_input)
        
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  - Tensor multiplication: {x.shape} x {y.shape} = {z.shape}")
        print(f"  - Neural network: {test_input.shape} -> {output.shape}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        
        return True
        
    except Exception as e:
        print(f"✗ PyTorch failed: {e}")
        return False


def test_computer_vision_module():
    """Test our computer vision module."""
    print("\\nTesting custom computer vision module...")
    
    try:
        sys.path.insert(0, '../src')
        from computer_vision.image_processor import ImageProcessor
        
        processor = ImageProcessor()
        
        # Create a test image
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [0, 255, 0]  # Green square
        
        # Test processing functions
        gray = processor.convert_to_grayscale(test_image)
        resized = processor.resize_image(test_image, (50, 50))
        info = processor.get_image_info(test_image)
        
        print("✓ Computer Vision module working")
        print(f"  - Image info: {info['width']}x{info['height']}, {info['channels']} channels")
        print(f"  - Grayscale conversion: {test_image.shape} -> {gray.shape}")
        print(f"  - Resize operation: {test_image.shape} -> {resized.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Computer Vision module failed: {e}")
        traceback.print_exc()
        return False


def test_deep_learning_module():
    """Test our deep learning module."""
    print("\\nTesting custom deep learning module...")
    
    try:
        sys.path.insert(0, '../src')
        from deep_learning.neural_network import SimpleNN, create_sample_data
        
        # Create a simple model
        model = SimpleNN(input_size=10, hidden_size=20, num_classes=3)
        
        # Create sample data
        X, y = create_sample_data(num_samples=50, input_size=10, num_classes=3)
        
        # Test forward pass
        import torch
        with torch.no_grad():
            output = model(X[:5])
        
        print("✓ Deep Learning module working")
        print(f"  - Model created with {sum(p.numel() for p in model.parameters())} parameters")
        print(f"  - Sample data: {X.shape}, labels: {y.shape}")
        print(f"  - Forward pass: {X[:5].shape} -> {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Deep Learning module failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=== Simple ML/CV Environment Test ===\\n")
    
    tests = [
        test_basic_imports,
        test_opencv,
        test_pytorch,
        test_computer_vision_module,
        test_deep_learning_module,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed with exception: {e}")
    
    print(f"\\n=== Test Results ===")
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 All tests passed! Your ML/CV environment is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())