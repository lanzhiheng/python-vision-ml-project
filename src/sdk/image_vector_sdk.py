#!/usr/bin/env python3
"""
Image Vector SDK Main Module

This module provides the main SDK class that integrates image vectorization
and vector storage capabilities with a simple, high-level API.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import json

from .config_manager import ConfigManager
from ..vectorization import ImageVectorizer
from ..database import VectorStore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageVectorSDK:
    """
    Main SDK class for image vectorization and storage.
    
    This class provides a high-level interface for:
    - Loading and processing images
    - Converting images to vector embeddings
    - Storing vectors in Milvus database
    - Searching for similar images
    - Managing the vector database
    """
    
    def __init__(self,
                 config_file: Optional[Union[str, Path]] = None,
                 env_file: Optional[Union[str, Path]] = None,
                 auto_connect: bool = True):
        """
        Initialize the Image Vector SDK.
        
        Args:
            config_file: Path to YAML configuration file
            env_file: Path to .env file
            auto_connect: Whether to automatically connect to database
        """
        # Load configuration
        if config_file is None:
            config_file = Path(__file__).parent.parent.parent / "configs" / "sdk_config.yaml"
        if env_file is None:
            env_file = Path(__file__).parent.parent.parent / ".env"
            
        self.config_manager = ConfigManager(config_file, env_file)
        
        # Validate configuration
        config_errors = self.config_manager.validate_config()
        if config_errors:
            logger.warning(f"Configuration validation errors: {config_errors}")
        
        # Initialize components
        self._setup_logging()
        self._initialize_vectorizer()
        self._initialize_vector_store()
        
        # Connection status
        self.connected = False
        
        # Statistics
        self.stats = {
            'images_processed': 0,
            'vectors_stored': 0,
            'searches_performed': 0,
            'total_processing_time': 0.0,
            'session_start_time': time.time()
        }
        
        # Auto-connect if requested
        if auto_connect:
            self.connect()
        
        logger.info("ImageVectorSDK initialized successfully")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_config = self.config_manager.get_logging_config()
        
        # Create logs directory if it doesn't exist
        log_file = Path(log_config['file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config['level'].upper()),
            format=log_config['format'],
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_vectorizer(self):
        """Initialize the image vectorizer."""
        clip_config = self.config_manager.get_clip_config()
        perf_config = self.config_manager.get_performance_config()
        
        self.vectorizer = ImageVectorizer(
            model_name=clip_config['model_name'],
            pretrained=clip_config['pretrained'],
            device=clip_config['device'],
            cache_enabled=perf_config['enable_cache'],
            cache_size=perf_config['cache_size'],
            max_workers=perf_config['max_workers']
        )
        
        logger.info("Image vectorizer initialized")
    
    def _initialize_vector_store(self):
        """Initialize the vector store."""
        milvus_config = self.config_manager.get_milvus_config()
        vector_config = self.config_manager.get_vector_config()
        
        collection_name = milvus_config.pop('collection_name', 'image_vectors')
        
        self.vector_store = VectorStore(
            milvus_config=milvus_config,
            collection_name=collection_name,
            dimension=vector_config['dimension'],
            metric_type=vector_config['metric_type'],
            index_type=vector_config['index_type']
        )
        
        logger.info("Vector store initialized")
    
    def connect(self) -> bool:
        """
        Connect to the vector database.
        
        Returns:
            bool: True if connection successful
        """
        if self.vector_store.connect():
            self.connected = True
            logger.info("SDK connected to database successfully")
            return True
        else:
            logger.error("Failed to connect SDK to database")
            return False
    
    def disconnect(self):
        """Disconnect from the vector database."""
        if self.connected:
            self.vector_store.disconnect()
            self.connected = False
            logger.info("SDK disconnected from database")
    
    def process_and_store_image(self,
                               image_path: Union[str, Path],
                               metadata: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        Process a single image and store its vector in the database.
        
        Args:
            image_path: Path to the image file
            metadata: Additional metadata to store
            
        Returns:
            Vector ID if successful, None otherwise
        """
        if not self.connected:
            raise RuntimeError("SDK not connected to database")
        
        try:
            # Vectorize image
            start_time = time.time()
            result = self.vectorizer.vectorize_image(str(image_path))
            
            # Add custom metadata
            if metadata:
                result.update(metadata)
            
            # Store vector
            ids = self.vector_store.store_image_vectors([result])
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['images_processed'] += 1
            self.stats['vectors_stored'] += len(ids) if ids else 0
            self.stats['total_processing_time'] += processing_time
            
            if ids:
                logger.info(f"Processed and stored image {image_path} with ID {ids[0]}")
                return ids[0]
            else:
                logger.error(f"Failed to store vector for {image_path}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return None
    
    def process_and_store_images(self,
                                image_paths: List[Union[str, Path]],
                                metadata_list: Optional[List[Dict[str, Any]]] = None,
                                batch_size: Optional[int] = None,
                                parallel: bool = True) -> List[int]:
        """
        Process multiple images and store their vectors in the database.
        
        Args:
            image_paths: List of image file paths
            metadata_list: List of metadata dictionaries (one per image)
            batch_size: Batch size for processing
            parallel: Whether to use parallel processing
            
        Returns:
            List of vector IDs
        """
        if not self.connected:
            raise RuntimeError("SDK not connected to database")
        
        if not image_paths:
            return []
        
        # Use default batch size if not provided
        if batch_size is None:
            batch_size = self.config_manager.get_image_processing_config()['batch_size']
        
        try:
            start_time = time.time()
            
            # Vectorize images
            results = self.vectorizer.vectorize_images_batch(
                image_paths,
                batch_size=batch_size,
                parallel=parallel
            )
            
            # Add custom metadata if provided
            if metadata_list:
                if len(metadata_list) != len(results):
                    logger.warning("Metadata list length doesn't match results length")
                else:
                    for i, metadata in enumerate(metadata_list):
                        if i < len(results):
                            results[i].update(metadata)
            
            # Store vectors in batches
            all_ids = []
            for i in range(0, len(results), batch_size):
                batch = results[i:i + batch_size]
                ids = self.vector_store.store_image_vectors(batch)
                if ids:
                    all_ids.extend(ids)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['images_processed'] += len(results)
            self.stats['vectors_stored'] += len(all_ids)
            self.stats['total_processing_time'] += processing_time
            
            logger.info(f"Processed and stored {len(all_ids)}/{len(image_paths)} images")
            return all_ids
            
        except Exception as e:
            logger.error(f"Failed to process images: {e}")
            return []
    
    def process_directory(self,
                         directory: Union[str, Path],
                         recursive: bool = True,
                         **kwargs) -> List[int]:
        """
        Process all images in a directory and store their vectors.
        
        Args:
            directory: Directory path
            recursive: Whether to search recursively
            **kwargs: Additional arguments for processing
            
        Returns:
            List of vector IDs
        """
        if not self.connected:
            raise RuntimeError("SDK not connected to database")
        
        directory = Path(directory)
        
        # Find all image files
        supported_formats = self.config_manager.get_image_processing_config()['supported_formats']
        image_paths = []
        
        if recursive:
            for ext in supported_formats:
                image_paths.extend(directory.rglob(f"*{ext}"))
                image_paths.extend(directory.rglob(f"*{ext.upper()}"))
        else:
            for ext in supported_formats:
                image_paths.extend(directory.glob(f"*{ext}"))
                image_paths.extend(directory.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_paths)} images in {directory}")
        
        if not image_paths:
            return []
        
        return self.process_and_store_images(image_paths, **kwargs)
    
    def search_similar_images(self,
                             query_image: Union[str, Path, np.ndarray],
                             top_k: int = 10,
                             threshold: float = 0.0,
                             filters: Optional[Dict[str, Any]] = None,
                             include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar images in the database.
        
        Args:
            query_image: Query image (path or numpy array)
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            filters: Additional filters to apply
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of similar image results
        """
        if not self.connected:
            raise RuntimeError("SDK not connected to database")
        
        try:
            # Get query vector
            if isinstance(query_image, (str, Path)):
                result = self.vectorizer.vectorize_image(str(query_image))
                query_vector = result['embedding']
            elif isinstance(query_image, np.ndarray):
                query_vector = query_image
            else:
                raise ValueError("Query image must be a file path or numpy array")
            
            # Search for similar images
            results = self.vector_store.search_similar_images(
                query_vector=query_vector,
                top_k=top_k,
                threshold=threshold,
                filters=filters,
                include_metadata=include_metadata
            )
            
            # Update statistics
            self.stats['searches_performed'] += 1
            
            logger.info(f"Found {len(results)} similar images")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar images: {e}")
            return []
    
    def get_image_vector(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Get the vector representation of an image without storing it.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image vector or None if failed
        """
        try:
            result = self.vectorizer.vectorize_image(str(image_path))
            return result['embedding']
        except Exception as e:
            logger.error(f"Failed to get vector for {image_path}: {e}")
            return None
    
    def delete_vectors(self,
                      ids: Optional[List[int]] = None,
                      image_paths: Optional[List[str]] = None) -> bool:
        """
        Delete vectors from the database.
        
        Args:
            ids: List of vector IDs to delete
            image_paths: List of image paths to delete
            
        Returns:
            bool: True if deletion successful
        """
        if not self.connected:
            raise RuntimeError("SDK not connected to database")
        
        return self.vector_store.delete_image_vectors(ids=ids, image_paths=image_paths)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.connected:
            return {}
        
        return self.vector_store.get_stats()
    
    def get_sdk_stats(self) -> Dict[str, Any]:
        """Get SDK usage statistics."""
        current_time = time.time()
        session_duration = current_time - self.stats['session_start_time']
        
        stats = self.stats.copy()
        stats.update({
            'session_duration_seconds': session_duration,
            'avg_processing_time_per_image': (
                self.stats['total_processing_time'] / max(self.stats['images_processed'], 1)
            ),
            'connected': self.connected,
            'vectorizer_cache_stats': self.vectorizer.get_cache_stats()
        })
        
        return stats
    
    def export_database(self,
                       output_file: Union[str, Path],
                       include_embeddings: bool = False) -> bool:
        """
        Export database contents to a file.
        
        Args:
            output_file: Output file path
            include_embeddings: Whether to include embedding vectors
            
        Returns:
            bool: True if export successful
        """
        if not self.connected:
            raise RuntimeError("SDK not connected to database")
        
        return self.vector_store.export_vectors(output_file, include_embeddings=include_embeddings)
    
    def import_database(self,
                       input_file: Union[str, Path],
                       batch_size: int = 100) -> int:
        """
        Import vectors from a file into the database.
        
        Args:
            input_file: Input file path
            batch_size: Batch size for import
            
        Returns:
            int: Number of vectors imported
        """
        if not self.connected:
            raise RuntimeError("SDK not connected to database")
        
        return self.vector_store.import_vectors(input_file, batch_size)
    
    def rebuild_index(self,
                     index_type: Optional[str] = None,
                     metric_type: Optional[str] = None) -> bool:
        """
        Rebuild the vector index.
        
        Args:
            index_type: Type of index to create
            metric_type: Distance metric type
            
        Returns:
            bool: True if successful
        """
        if not self.connected:
            raise RuntimeError("SDK not connected to database")
        
        return self.vector_store.create_index(index_type, metric_type)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return self.config_manager.get_summary()
    
    def save_config(self, output_file: Union[str, Path]):
        """Save current configuration to a file."""
        self.config_manager.save_config(output_file)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of all components.
        
        Returns:
            Dictionary with health status of each component
        """
        health = {
            'sdk': {'status': 'healthy', 'details': 'SDK initialized'},
            'config': {'status': 'healthy', 'details': 'Configuration loaded'},
            'vectorizer': {'status': 'healthy', 'details': 'Vectorizer ready'},
            'database': {'status': 'unknown', 'details': 'Not tested'},
            'overall': {'status': 'unknown', 'details': ''}
        }
        
        # Check configuration
        config_errors = self.config_manager.validate_config()
        if config_errors:
            health['config'] = {
                'status': 'warning',
                'details': f'Configuration errors: {config_errors}'
            }
        
        # Check database connection
        if self.connected:
            try:
                db_stats = self.get_database_stats()
                health['database'] = {
                    'status': 'healthy',
                    'details': f'Connected, {db_stats.get(\"total_vectors\", 0)} vectors stored'
                }
            except Exception as e:
                health['database'] = {
                    'status': 'error',
                    'details': f'Database error: {e}'
                }
        else:
            health['database'] = {
                'status': 'warning',
                'details': 'Not connected to database'
            }
        
        # Determine overall status
        statuses = [comp['status'] for comp in health.values() if comp != health['overall']]
        if 'error' in statuses:
            health['overall'] = {'status': 'error', 'details': 'Some components have errors'}
        elif 'warning' in statuses:
            health['overall'] = {'status': 'warning', 'details': 'Some components have warnings'}
        else:
            health['overall'] = {'status': 'healthy', 'details': 'All components healthy'}
        
        return health
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def __repr__(self):
        """String representation of the SDK."""
        return f"ImageVectorSDK(connected={self.connected}, images_processed={self.stats['images_processed']}, vectors_stored={self.stats['vectors_stored']})"