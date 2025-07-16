#!/usr/bin/env python3
"""
Computer Vision Image Processor

This module provides basic image processing functionality using OpenCV.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import os


class ImageProcessor:
    """
    A class for basic image processing operations using OpenCV.
    """
    
    def __init__(self):
        """Initialize the ImageProcessor."""
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load an image from file.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Loaded image or None if failed
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found.")
            return None
            
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image {image_path}")
                return None
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def save_image(self, image: np.ndarray, output_path: str) -> bool:
        """
        Save an image to file.
        
        Args:
            image (np.ndarray): Image to save
            output_path (str): Output file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            success = cv2.imwrite(output_path, image)
            if success:
                print(f"Image saved to {output_path}")
                return True
            else:
                print(f"Failed to save image to {output_path}")
                return False
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
    def resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Resize an image.
        
        Args:
            image (np.ndarray): Input image
            size (Tuple[int, int]): Target size (width, height)
            
        Returns:
            np.ndarray: Resized image
        """
        return cv2.resize(image, size)
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.
        
        Args:
            image (np.ndarray): Input color image
            
        Returns:
            np.ndarray: Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian blur to image.
        
        Args:
            image (np.ndarray): Input image
            kernel_size (int): Size of the Gaussian kernel (must be odd)
            
        Returns:
            np.ndarray: Blurred image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def detect_edges(self, image: np.ndarray, low_threshold: int = 50, 
                    high_threshold: int = 150) -> np.ndarray:
        """
        Detect edges using Canny edge detector.
        
        Args:
            image (np.ndarray): Input image
            low_threshold (int): Lower threshold for edge detection
            high_threshold (int): Upper threshold for edge detection
            
        Returns:
            np.ndarray: Edge detected image
        """
        gray = self.convert_to_grayscale(image)
        return cv2.Canny(gray, low_threshold, high_threshold)
    
    def detect_contours(self, image: np.ndarray) -> Tuple[list, np.ndarray]:
        """
        Detect contours in image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Tuple[list, np.ndarray]: Contours and hierarchy
        """
        gray = self.convert_to_grayscale(image)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy
    
    def draw_contours(self, image: np.ndarray, contours: list, 
                     color: Tuple[int, int, int] = (0, 255, 0), 
                     thickness: int = 2) -> np.ndarray:
        """
        Draw contours on image.
        
        Args:
            image (np.ndarray): Input image
            contours (list): List of contours
            color (Tuple[int, int, int]): Color for drawing contours
            thickness (int): Thickness of contour lines
            
        Returns:
            np.ndarray: Image with drawn contours
        """
        result = image.copy()
        cv2.drawContours(result, contours, -1, color, thickness)
        return result
    
    def get_image_info(self, image: np.ndarray) -> dict:
        """
        Get basic information about an image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            dict: Image information
        """
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'shape': image.shape,
            'dtype': str(image.dtype),
            'size': image.size
        }


def demo_image_processing():
    """
    Demonstrate basic image processing functionality.
    """
    print("=== Computer Vision Image Processing Demo ===")
    
    # Create a sample image for demonstration
    sample_image = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(sample_image, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(sample_image, (300, 100), 50, (0, 255, 0), -1)  # Green circle
    cv2.line(sample_image, (200, 200), (350, 250), (0, 0, 255), 3)  # Red line
    
    processor = ImageProcessor()
    
    # Display image information
    info = processor.get_image_info(sample_image)
    print("Sample Image Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Apply various processing operations
    print("\\nApplying image processing operations...")
    
    # Convert to grayscale
    gray = processor.convert_to_grayscale(sample_image)
    print(f"Converted to grayscale: {gray.shape}")
    
    # Apply blur
    blurred = processor.apply_gaussian_blur(sample_image, 15)
    print("Applied Gaussian blur")
    
    # Detect edges
    edges = processor.detect_edges(sample_image)
    print(f"Detected edges: {edges.shape}")
    
    # Resize image
    resized = processor.resize_image(sample_image, (200, 150))
    print(f"Resized image: {resized.shape}")
    
    print("\\nDemo completed successfully!")
    return True


if __name__ == "__main__":
    demo_image_processing()