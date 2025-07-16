#!/usr/bin/env python3
"""
Deep Learning Example Script

This script demonstrates basic deep learning functionality using PyTorch.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from deep_learning.neural_network import demo_neural_network, demo_cnn


def main():
    """Run deep learning examples."""
    print("Running Deep Learning Example...")
    print("-" * 50)
    
    # Run the neural network demo
    print("1. Neural Network Demo:")
    success1 = demo_neural_network()
    
    print("\\n2. CNN Demo:")
    success2 = demo_cnn()
    
    if success1 and success2:
        print("\\n✓ Deep Learning example completed successfully!")
    else:
        print("\\n✗ Deep Learning example failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())