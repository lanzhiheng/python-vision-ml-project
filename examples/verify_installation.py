#!/usr/bin/env python3
"""
Installation Verification Script

This script verifies that all required dependencies are properly installed
and working in the ML/CV project environment.
"""

import sys
import importlib
import traceback
from typing import Dict, List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version meets requirements."""
    version = sys.version_info
    required_major, required_minor = 3, 10
    
    if version.major >= required_major and version.minor >= required_minor:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (requires >= {required_major}.{required_minor})"


def check_package_import(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package can be imported."""
    try:
        if import_name is None:
            import_name = package_name
        
        module = importlib.import_module(import_name)
        
        # Get version if available
        version = "unknown"
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'VERSION'):
            version = module.VERSION
        elif hasattr(module, 'version'):
            version = module.version
        
        return True, f"{package_name} v{version}"
    except ImportError as e:
        return False, f"{package_name}: {str(e)}"
    except Exception as e:
        return False, f"{package_name}: Unexpected error - {str(e)}"


def check_opencv_functionality() -> Tuple[bool, str]:
    """Check OpenCV specific functionality."""
    try:
        import cv2
        import numpy as np
        
        # Test basic image operations
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        return True, f"OpenCV functionality verified (shape: {test_image.shape} -> {edges.shape})"
    except Exception as e:
        return False, f"OpenCV functionality test failed: {str(e)}"


def check_pytorch_functionality() -> Tuple[bool, str]:
    """Check PyTorch specific functionality."""
    try:
        import torch
        import torch.nn as nn
        
        # Test basic tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        
        # Test CUDA availability
        cuda_info = ""
        if torch.cuda.is_available():
            cuda_info = f", CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}"
        else:
            cuda_info = ", CUDA: Not available"
        
        # Test simple neural network
        model = nn.Linear(3, 2)
        test_input = torch.randn(1, 3)
        output = model(test_input)
        
        return True, f"PyTorch functionality verified{cuda_info} (tensor ops: {z.shape}, nn: {output.shape})"
    except Exception as e:
        return False, f"PyTorch functionality test failed: {str(e)}"


def check_pyqt5_functionality() -> Tuple[bool, str]:
    """Check PyQt5 specific functionality."""
    try:
        from PyQt5.QtWidgets import QApplication, QWidget
        from PyQt5.QtCore import Qt
        
        # Test if QApplication can be created (in headless mode)
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        # Test widget creation
        widget = QWidget()
        
        return True, "PyQt5 functionality verified (widgets can be created)"
    except Exception as e:
        return False, f"PyQt5 functionality test failed: {str(e)}"


def check_scikit_learn_functionality() -> Tuple[bool, str]:
    """Check scikit-learn specific functionality."""
    try:
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        # Test basic ML workflow
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return True, f"scikit-learn functionality verified (accuracy: {accuracy:.2f})"
    except Exception as e:
        return False, f"scikit-learn functionality test failed: {str(e)}"


def run_verification() -> Dict[str, Tuple[bool, str]]:
    """Run all verification checks."""
    results = {}
    
    print("=== ML/CV Project Installation Verification ===\\n")
    
    # Check Python version
    print("1. Checking Python version...")
    results['python'] = check_python_version()
    
    # Check core packages
    core_packages = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('PIL/Pillow', 'PIL'),
        ('PyYAML', 'yaml'),
    ]
    
    print("\\n2. Checking core packages...")
    for package_name, import_name in core_packages:
        results[package_name] = check_package_import(package_name, import_name)
    
    # Check ML/DL packages
    ml_packages = [
        ('PyTorch', 'torch'),
        ('torchvision', 'torchvision'),
        ('torchaudio', 'torchaudio'),
        ('scikit-learn', 'sklearn'),
    ]
    
    print("\\n3. Checking ML/DL packages...")
    for package_name, import_name in ml_packages:
        results[package_name] = check_package_import(package_name, import_name)
    
    # Check CV packages
    cv_packages = [
        ('OpenCV', 'cv2'),
    ]
    
    print("\\n4. Checking Computer Vision packages...")
    for package_name, import_name in cv_packages:
        results[package_name] = check_package_import(package_name, import_name)
    
    # Check GUI packages
    gui_packages = [
        ('PyQt5', 'PyQt5.QtWidgets'),
    ]
    
    print("\\n5. Checking GUI packages...")
    for package_name, import_name in gui_packages:
        results[package_name] = check_package_import(package_name, import_name)
    
    # Check functionality
    print("\\n6. Checking functionality...")
    
    print("   Testing OpenCV...")
    results['opencv_functionality'] = check_opencv_functionality()
    
    print("   Testing PyTorch...")
    results['pytorch_functionality'] = check_pytorch_functionality()
    
    print("   Testing PyQt5...")
    results['pyqt5_functionality'] = check_pyqt5_functionality()
    
    print("   Testing scikit-learn...")
    results['sklearn_functionality'] = check_scikit_learn_functionality()
    
    return results


def print_results(results: Dict[str, Tuple[bool, str]]):
    """Print verification results."""
    print("\\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for component, (success, message) in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:<8} {component}: {message}")
        if success:
            passed += 1
    
    print("="*60)
    print(f"SUMMARY: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Your ML/CV environment is ready to use.")
        return True
    else:
        print("⚠️  Some checks failed. Please install missing packages or fix issues.")
        return False


def main():
    """Main verification function."""
    try:
        results = run_verification()
        success = print_results(results)
        
        if not success:
            print("\\nTo install missing packages, run:")
            print("pip install -r requirements.txt")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\\nVerification script error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())