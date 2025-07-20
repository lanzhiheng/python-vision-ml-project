"""
Image Vectorization Module

This module provides functionality to convert images into vector representations
using various deep learning models, primarily CLIP.
"""

from .image_vectorizer import ImageVectorizer
from .clip_encoder import CLIPEncoder

__all__ = ['ImageVectorizer', 'CLIPEncoder']