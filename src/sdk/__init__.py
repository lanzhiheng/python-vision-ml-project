"""
Image Vectorization SDK

This module provides a comprehensive SDK for image vectorization and storage
using CLIP models and Milvus vector database.
"""

from .image_vector_sdk import ImageVectorSDK
from .config_manager import ConfigManager

__all__ = ['ImageVectorSDK', 'ConfigManager']

# Version information
__version__ = "1.0.0"
__author__ = "Image Vector SDK Team"
__email__ = "support@example.com"