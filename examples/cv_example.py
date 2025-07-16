#!/usr/bin/env python3
"""
Computer Vision Example Script

This script demonstrates basic computer vision functionality using OpenCV.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from computer_vision.image_processor import ImageProcessor, demo_image_processing


def main():
    """Run computer vision examples."""
    print("Running Computer Vision Example...")
    print("-" * 50)
    
    # Run the image processing demo
    success = demo_image_processing()
    
    if success:
        print("\\n✓ Computer Vision example completed successfully!")
    else:
        print("\\n✗ Computer Vision example failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())