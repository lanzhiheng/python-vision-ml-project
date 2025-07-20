#!/usr/bin/env python3
"""
Image Vectorizer Module

This module provides high-level image vectorization functionality with
support for various models and preprocessing options.
"""

import os
import logging
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple, Any
import numpy as np
from PIL import Image
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .clip_encoder import CLIPEncoder

logger = logging.getLogger(__name__)


class ImageVectorizer:
    """
    High-level image vectorization class with caching and batch processing.
    """
    
    def __init__(self, 
                 model_name: str = "ViT-B/32",
                 pretrained: str = "openai",
                 device: str = "auto",
                 cache_enabled: bool = True,
                 cache_size: int = 1000,
                 max_workers: int = 4):
        """
        Initialize the Image Vectorizer.
        
        Args:
            model_name: CLIP model architecture name
            pretrained: Pretrained weights to use
            device: Device to run the model on
            cache_enabled: Whether to enable embedding caching
            cache_size: Maximum number of embeddings to cache
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.cache_enabled = cache_enabled
        self.max_workers = max_workers
        
        # Initialize encoder
        self.encoder = CLIPEncoder(model_name, pretrained, device)
        
        # Initialize cache
        self.embedding_cache = {}
        self.cache_size = cache_size
        self.cache_stats = {'hits': 0, 'misses': 0}
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        logger.info(f"ImageVectorizer initialized with {model_name}")
    
    def vectorize_image(self, 
                       image_path: Union[str, Path],
                       normalize: bool = True,
                       use_cache: bool = True) -> Dict[str, Any]:
        """
        Vectorize a single image.
        
        Args:
            image_path: Path to the image file
            normalize: Whether to normalize the vector
            use_cache: Whether to use cached embeddings
            
        Returns:
            Dict containing embedding and metadata
        """
        image_path = Path(image_path)
        
        # Validate image
        if not self._is_valid_image(image_path):
            raise ValueError(f"Invalid image file: {image_path}")
        
        # Check cache first
        if use_cache and self.cache_enabled:
            cache_key = self._get_cache_key(image_path)
            if cache_key in self.embedding_cache:
                self.cache_stats['hits'] += 1
                logger.debug(f"Cache hit for {image_path}")
                return self.embedding_cache[cache_key]
        
        # Extract embedding
        start_time = time.time()
        try:
            embedding = self.encoder.encode_image(str(image_path), normalize=normalize)
            processing_time = time.time() - start_time
            
            # Create result
            result = {
                'image_path': str(image_path),
                'embedding': embedding,
                'embedding_dim': len(embedding),
                'model_name': self.model_name,
                'normalized': normalize,
                'processing_time': processing_time,
                'file_size': image_path.stat().st_size,
                'image_hash': self._compute_file_hash(image_path)
            }
            
            # Add to cache
            if use_cache and self.cache_enabled:
                self._add_to_cache(cache_key, result)
                self.cache_stats['misses'] += 1
            
            logger.debug(f"Vectorized {image_path} in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to vectorize {image_path}: {e}")
            raise
    
    def vectorize_images_batch(self,
                              image_paths: List[Union[str, Path]],
                              batch_size: int = 32,
                              normalize: bool = True,
                              use_cache: bool = True,
                              parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Vectorize multiple images in batches.
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing
            normalize: Whether to normalize vectors
            use_cache: Whether to use cached embeddings
            parallel: Whether to use parallel processing
            
        Returns:
            List of dictionaries containing embeddings and metadata
        """
        image_paths = [Path(p) for p in image_paths]
        
        # Filter valid images
        valid_paths = [p for p in image_paths if self._is_valid_image(p)]
        if len(valid_paths) != len(image_paths):
            logger.warning(f"Filtered out {len(image_paths) - len(valid_paths)} invalid images")
        
        if not valid_paths:
            return []
        
        results = []
        
        if parallel and len(valid_paths) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(self.vectorize_image, path, normalize, use_cache): path
                    for path in valid_paths
                }
                
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to process {path}: {e}")
        else:
            # Sequential processing
            for path in valid_paths:
                try:
                    result = self.vectorize_image(path, normalize, use_cache)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
        
        # Sort results by original order
        path_to_result = {r['image_path']: r for r in results}
        ordered_results = []
        for path in valid_paths:
            if str(path) in path_to_result:
                ordered_results.append(path_to_result[str(path)])
        
        logger.info(f"Vectorized {len(ordered_results)}/{len(image_paths)} images")
        return ordered_results
    
    def vectorize_directory(self,
                           directory: Union[str, Path],
                           recursive: bool = True,
                           **kwargs) -> List[Dict[str, Any]]:
        """
        Vectorize all images in a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to search recursively
            **kwargs: Additional arguments for vectorize_images_batch
            
        Returns:
            List of vectorization results
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        # Find all image files
        image_paths = []
        if recursive:
            for ext in self.supported_formats:
                image_paths.extend(directory.rglob(f"*{ext}"))
                image_paths.extend(directory.rglob(f"*{ext.upper()}"))
        else:
            for ext in self.supported_formats:
                image_paths.extend(directory.glob(f"*{ext}"))
                image_paths.extend(directory.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_paths)} images in {directory}")
        
        if not image_paths:
            return []
        
        return self.vectorize_images_batch(image_paths, **kwargs)
    
    def search_similar_images(self,
                             query_embedding: np.ndarray,
                             image_embeddings: List[Dict[str, Any]],
                             top_k: int = 10,
                             threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Find similar images based on embedding similarity.
        
        Args:
            query_embedding: Query image embedding
            image_embeddings: List of image embeddings to search
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar images with similarity scores
        """
        similarities = []
        
        for img_data in image_embeddings:
            embedding = img_data['embedding']
            similarity = self._cosine_similarity(query_embedding, embedding)
            
            if similarity >= threshold:
                result = img_data.copy()
                result['similarity'] = similarity
                similarities.append(result)
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def get_embedding_stats(self, embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about a collection of embeddings."""
        if not embeddings:
            return {}
        
        embedding_vectors = np.array([e['embedding'] for e in embeddings])
        
        return {
            'count': len(embeddings),
            'dimension': embedding_vectors.shape[1],
            'mean_norm': float(np.mean(np.linalg.norm(embedding_vectors, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embedding_vectors, axis=1))),
            'total_file_size': sum(e.get('file_size', 0) for e in embeddings),
            'avg_processing_time': np.mean([e.get('processing_time', 0) for e in embeddings]),
            'model_name': embeddings[0].get('model_name', 'unknown')
        }
    
    def save_embeddings(self, 
                       embeddings: List[Dict[str, Any]], 
                       output_file: Union[str, Path]):
        """Save embeddings to a file."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_embeddings = []
        for emb in embeddings:
            emb_copy = emb.copy()
            emb_copy['embedding'] = emb_copy['embedding'].tolist()
            serializable_embeddings.append(emb_copy)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_embeddings, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(embeddings)} embeddings to {output_file}")
    
    def load_embeddings(self, input_file: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load embeddings from a file."""
        input_file = Path(input_file)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            embeddings = json.load(f)
        
        # Convert lists back to numpy arrays
        for emb in embeddings:
            emb['embedding'] = np.array(emb['embedding'])
        
        logger.info(f"Loaded {len(embeddings)} embeddings from {input_file}")
        return embeddings
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        self.cache_stats = {'hits': 0, 'misses': 0}
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'cache_enabled': self.cache_enabled,
            'cache_size': len(self.embedding_cache),
            'max_cache_size': self.cache_size,
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': hit_rate
        }
    
    def _is_valid_image(self, image_path: Path) -> bool:
        """Check if the image file is valid."""
        if not image_path.exists():
            return False
        
        if image_path.suffix.lower() not in self.supported_formats:
            return False
        
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    def _get_cache_key(self, image_path: Path) -> str:
        """Generate a cache key for an image."""
        stat = image_path.stat()
        return f"{image_path}:{stat.st_mtime}:{stat.st_size}:{self.model_name}"
    
    def _add_to_cache(self, key: str, value: Dict[str, Any]):
        """Add an embedding to cache with size management."""
        if len(self.embedding_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        
        self.embedding_cache[key] = value
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))