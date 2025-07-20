#!/usr/bin/env python3
"""
Vector Store Module

This module provides a high-level interface for storing and retrieving
image vectors with metadata management and search capabilities.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import json
import time

from .milvus_client import MilvusClient

logger = logging.getLogger(__name__)


class VectorStore:
    """
    High-level vector storage interface with advanced search and management features.
    """
    
    def __init__(self,
                 milvus_config: Dict[str, Any],
                 collection_name: str = "image_vectors",
                 dimension: int = 512,
                 metric_type: str = "L2",
                 index_type: str = "IVF_FLAT",
                 auto_create: bool = True):
        """
        Initialize Vector Store.
        
        Args:
            milvus_config: Milvus connection configuration
            collection_name: Name of the collection to use
            dimension: Vector dimension
            metric_type: Distance metric type
            index_type: Index type for vectors
            auto_create: Whether to auto-create collection if not exists
        """
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric_type = metric_type
        self.index_type = index_type
        self.auto_create = auto_create
        
        # Initialize Milvus client
        self.client = MilvusClient(**milvus_config)
        self.connected = False
        
        # Statistics
        self.stats = {
            'total_inserted': 0,
            'total_searched': 0,
            'total_deleted': 0,
            'last_operation_time': None
        }
        
        logger.info(f"VectorStore initialized for collection: {collection_name}")
    
    def connect(self) -> bool:
        """
        Connect to the vector database.
        
        Returns:
            bool: True if connection successful
        """
        if self.client.connect():
            self.connected = True
            
            # Auto-create collection if needed
            if self.auto_create:
                self._ensure_collection_exists()
            
            logger.info("VectorStore connected successfully")
            return True
        else:
            logger.error("Failed to connect VectorStore")
            return False
    
    def disconnect(self):
        """Disconnect from the vector database."""
        self.client.disconnect()
        self.connected = False
        logger.info("VectorStore disconnected")
    
    def store_image_vectors(self,
                           image_data: List[Dict[str, Any]],
                           batch_size: int = 100) -> List[int]:
        """
        Store image vectors with metadata.
        
        Args:
            image_data: List of dictionaries containing image data with embeddings
            batch_size: Batch size for insertion
            
        Returns:
            List of inserted record IDs
        """
        if not self.connected:
            raise RuntimeError("VectorStore not connected")
        
        if not image_data:
            return []
        
        all_ids = []
        
        # Process in batches
        for i in range(0, len(image_data), batch_size):
            batch = image_data[i:i + batch_size]
            
            # Extract vectors and metadata
            vectors = [item['embedding'] for item in batch]
            metadata = [self._prepare_metadata(item) for item in batch]
            
            # Insert batch
            ids = self.client.insert_vectors(
                collection_name=self.collection_name,
                vectors=vectors,
                metadata=metadata
            )
            
            if ids:
                all_ids.extend(ids)
                self.stats['total_inserted'] += len(ids)
            
            logger.debug(f"Inserted batch {i//batch_size + 1}, size: {len(batch)}")
        
        self.stats['last_operation_time'] = time.time()
        logger.info(f"Stored {len(all_ids)} image vectors")
        return all_ids
    
    def search_similar_images(self,
                             query_vector: np.ndarray,
                             top_k: int = 10,
                             threshold: float = 0.0,
                             filters: Optional[Dict[str, Any]] = None,
                             include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar images based on vector similarity.
        
        Args:
            query_vector: Query image vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            filters: Additional filters to apply
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of similar image results
        """
        if not self.connected:
            raise RuntimeError("VectorStore not connected")
        
        # Build filter expression
        filter_expr = self._build_filter_expression(filters) if filters else ""
        
        # Determine output fields
        output_fields = ["image_path", "image_hash", "model_name", "created_at"]
        if include_metadata:
            output_fields.append("metadata")
        
        # Perform search
        results = self.client.search_vectors(
            collection_name=self.collection_name,
            query_vectors=[query_vector],
            top_k=top_k,
            filter_expr=filter_expr,
            output_fields=output_fields
        )
        
        if not results or not results[0]:
            return []
        
        # Format and filter results
        formatted_results = []
        for hit in results[0]:
            similarity = hit['score']
            
            # Apply threshold filter
            if similarity < threshold:
                continue
            
            result = {
                'id': hit['id'],
                'similarity': similarity,
                'distance': hit['distance'],
                'image_path': hit.get('image_path', ''),
                'image_hash': hit.get('image_hash', ''),
                'model_name': hit.get('model_name', ''),
                'created_at': hit.get('created_at', 0)
            }
            
            # Add metadata if requested
            if include_metadata and 'metadata' in hit:
                result['metadata'] = hit['metadata']
            
            formatted_results.append(result)
        
        self.stats['total_searched'] += 1
        self.stats['last_operation_time'] = time.time()
        
        logger.info(f"Found {len(formatted_results)} similar images")
        return formatted_results
    
    def search_by_image_path(self,
                            image_path: Union[str, Path],
                            top_k: int = 10,
                            exclude_self: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar images by image path.
        
        Args:
            image_path: Path to the query image
            top_k: Number of results to return
            exclude_self: Whether to exclude the query image from results
            
        Returns:
            List of similar image results
        """
        image_path = str(image_path)
        
        # First, find the vector for this image
        filter_expr = f'image_path == "{image_path}"'
        
        results = self.client.search_vectors(
            collection_name=self.collection_name,
            query_vectors=[np.zeros(self.dimension)],  # Dummy vector
            top_k=1,
            filter_expr=filter_expr,
            output_fields=["embedding"]
        )
        
        if not results or not results[0]:
            logger.warning(f"Image not found in database: {image_path}")
            return []
        
        # Get the embedding
        vector_id = results[0][0]['id']
        
        # Now search for similar images
        # Note: We need to retrieve the actual embedding first
        # This is a limitation that would require a separate query method
        logger.warning("search_by_image_path requires additional implementation for embedding retrieval")
        return []
    
    def delete_image_vectors(self,
                            ids: Optional[List[int]] = None,
                            image_paths: Optional[List[str]] = None,
                            filters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Delete image vectors from the store.
        
        Args:
            ids: List of vector IDs to delete
            image_paths: List of image paths to delete
            filters: Additional filters for deletion
            
        Returns:
            bool: True if deletion successful
        """
        if not self.connected:
            raise RuntimeError("VectorStore not connected")
        
        if ids:
            success = self.client.delete_vectors(self.collection_name, ids)
            if success:
                self.stats['total_deleted'] += len(ids)
                logger.info(f"Deleted {len(ids)} vectors by ID")
            return success
        
        elif image_paths:
            # Delete by image paths (requires custom implementation)
            logger.warning("Delete by image paths not yet implemented")
            return False
        
        elif filters:
            # Delete by filters (requires custom implementation)
            logger.warning("Delete by filters not yet implemented")
            return False
        
        else:
            raise ValueError("Must provide ids, image_paths, or filters for deletion")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        if not self.connected:
            return {}
        
        stats = self.client.get_collection_stats(self.collection_name)
        if stats:
            stats.update(self.stats)
        return stats or {}
    
    def export_vectors(self,
                      output_file: Union[str, Path],
                      filters: Optional[Dict[str, Any]] = None,
                      include_embeddings: bool = False) -> bool:
        """
        Export vectors and metadata to a file.
        
        Args:
            output_file: Output file path
            filters: Filters to apply for export
            include_embeddings: Whether to include embedding vectors
            
        Returns:
            bool: True if export successful
        """
        logger.warning("Export functionality requires additional implementation")
        return False
    
    def import_vectors(self,
                      input_file: Union[str, Path],
                      batch_size: int = 100) -> int:
        """
        Import vectors from a file.
        
        Args:
            input_file: Input file path
            batch_size: Batch size for import
            
        Returns:
            int: Number of vectors imported
        """
        logger.warning("Import functionality requires additional implementation")
        return 0
    
    def create_index(self,
                    index_type: Optional[str] = None,
                    metric_type: Optional[str] = None,
                    index_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create or rebuild index on the collection.
        
        Args:
            index_type: Type of index to create
            metric_type: Distance metric type
            index_params: Additional index parameters
            
        Returns:
            bool: True if index creation successful
        """
        if not self.connected:
            raise RuntimeError("VectorStore not connected")
        
        # Use provided parameters or defaults
        idx_type = index_type or self.index_type
        met_type = metric_type or self.metric_type
        
        collection = self.client.get_collection(self.collection_name)
        if not collection:
            return False
        
        try:
            # Drop existing index
            collection.drop_index()
            
            # Create new index
            index_params = index_params or {"nlist": 1024}
            index_config = {
                "metric_type": met_type,
                "index_type": idx_type,
                "params": index_params
            }
            
            collection.create_index("embedding", index_config)
            collection.load()
            
            logger.info(f"Created index {idx_type} with metric {met_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists, create if not."""
        collections = self.client.list_collections()
        
        if self.collection_name not in collections:
            success = self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                description=f"Image vectors collection with {self.dimension} dimensions",
                metric_type=self.metric_type,
                index_type=self.index_type
            )
            
            if success:
                logger.info(f"Created collection: {self.collection_name}")
            else:
                raise RuntimeError(f"Failed to create collection: {self.collection_name}")
    
    def _prepare_metadata(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for storage."""
        metadata = {
            'image_path': str(image_data.get('image_path', '')),
            'image_hash': image_data.get('image_hash', ''),
            'file_size': int(image_data.get('file_size', 0)),
            'model_name': image_data.get('model_name', ''),
            'extra_metadata': {
                'processing_time': image_data.get('processing_time', 0),
                'normalized': image_data.get('normalized', False),
                'embedding_dim': image_data.get('embedding_dim', self.dimension)
            }
        }
        
        # Add any additional metadata
        for key, value in image_data.items():
            if key not in ['embedding', 'image_path', 'image_hash', 'file_size', 'model_name']:
                if isinstance(value, (str, int, float, bool, list, dict)):
                    metadata['extra_metadata'][key] = value
        
        return metadata
    
    def _build_filter_expression(self, filters: Dict[str, Any]) -> str:
        """Build Milvus filter expression from filters dict."""
        expressions = []
        
        for key, value in filters.items():
            if key == 'model_name':
                expressions.append(f'model_name == "{value}"')
            elif key == 'file_size_min':
                expressions.append(f'file_size >= {value}')
            elif key == 'file_size_max':
                expressions.append(f'file_size <= {value}')
            elif key == 'created_after':
                expressions.append(f'created_at >= {value}')
            elif key == 'created_before':
                expressions.append(f'created_at <= {value}')
        
        return " and ".join(expressions)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        collection_info = self.get_collection_info()
        
        return {
            'connection_status': 'connected' if self.connected else 'disconnected',
            'collection_name': self.collection_name,
            'dimension': self.dimension,
            'metric_type': self.metric_type,
            'index_type': self.index_type,
            'total_vectors': collection_info.get('num_entities', 0),
            'operations': self.stats.copy()
        }