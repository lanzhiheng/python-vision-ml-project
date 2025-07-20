#!/usr/bin/env python3
"""
Milvus Client Module

This module provides a client for connecting to and interacting with Milvus
vector database for storing and retrieving image embeddings.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import time
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, Index
)
from pymilvus.exceptions import MilvusException

logger = logging.getLogger(__name__)


class MilvusClient:
    """
    A client for interacting with Milvus vector database.
    """
    
    def __init__(self,
                 host: str = "localhost",
                 port: int = 19530,
                 user: str = "",
                 password: str = "",
                 database: str = "default",
                 timeout: int = 30,
                 alias: str = "default"):
        """
        Initialize Milvus client.
        
        Args:
            host: Milvus server host
            port: Milvus server port
            user: Username for authentication
            password: Password for authentication
            database: Database name
            timeout: Connection timeout
            alias: Connection alias
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.timeout = timeout
        self.alias = alias
        
        self.connection = None
        self.collections = {}
        
        logger.info(f"Initializing Milvus client for {host}:{port}")
    
    def connect(self) -> bool:
        """
        Connect to Milvus server.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Connection parameters
            conn_params = {
                "host": self.host,
                "port": self.port,
                "alias": self.alias
            }
            
            # Add authentication if provided
            if self.user:
                conn_params["user"] = self.user
            if self.password:
                conn_params["password"] = self.password
            if self.database != "default":
                conn_params["db_name"] = self.database
            
            # Connect
            connections.connect(**conn_params)
            
            # Test connection
            if self._test_connection():
                self.connection = True
                logger.info(f"Successfully connected to Milvus at {self.host}:{self.port}")
                return True
            else:
                logger.error("Connection test failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Milvus server."""
        try:
            connections.disconnect(alias=self.alias)
            self.connection = None
            self.collections.clear()
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}")
    
    def create_collection(self,
                         collection_name: str,
                         dimension: int,
                         description: str = "",
                         metric_type: str = "L2",
                         index_type: str = "IVF_FLAT",
                         nlist: int = 1024) -> bool:
        """
        Create a new collection for storing image vectors.
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            description: Collection description
            metric_type: Distance metric (L2, IP, COSINE)
            index_type: Index type (IVF_FLAT, IVF_SQ8, HNSW, etc.)
            nlist: Number of cluster units (for IVF indexes)
            
        Returns:
            bool: True if creation successful
        """
        try:
            # Check if collection already exists
            if utility.has_collection(collection_name, using=self.alias):
                logger.warning(f"Collection {collection_name} already exists")
                return True
            
            # Define fields
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="image_hash", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="file_size", dtype=DataType.INT64),
                FieldSchema(name="model_name", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            # Create schema
            schema = CollectionSchema(fields, description=description)
            
            # Create collection
            collection = Collection(collection_name, schema, using=self.alias)
            
            # Create index on vector field
            index_params = {
                "metric_type": metric_type,
                "index_type": index_type,
                "params": {"nlist": nlist}
            }
            
            collection.create_index("embedding", index_params)
            
            # Load collection
            collection.load()
            
            # Store collection reference
            self.collections[collection_name] = collection
            
            logger.info(f"Created collection {collection_name} with dimension {dimension}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False
    
    def get_collection(self, collection_name: str) -> Optional[Collection]:
        """
        Get a collection by name.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection object or None if not found
        """
        try:
            if collection_name in self.collections:
                return self.collections[collection_name]
            
            if utility.has_collection(collection_name, using=self.alias):
                collection = Collection(collection_name, using=self.alias)
                self.collections[collection_name] = collection
                return collection
            
            logger.warning(f"Collection {collection_name} not found")
            return None
            
        except Exception as e:
            logger.error(f"Error getting collection {collection_name}: {e}")
            return None
    
    def insert_vectors(self,
                      collection_name: str,
                      vectors: List[np.ndarray],
                      metadata: List[Dict[str, Any]]) -> Optional[List[int]]:
        """
        Insert vectors and metadata into a collection.
        
        Args:
            collection_name: Name of the collection
            vectors: List of embedding vectors
            metadata: List of metadata dictionaries
            
        Returns:
            List of inserted IDs or None if failed
        """
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return None
            
            if len(vectors) != len(metadata):
                raise ValueError("Number of vectors must match number of metadata entries")
            
            # Prepare data
            current_time = int(time.time() * 1000)  # milliseconds
            
            data = [
                [meta.get('image_path', '') for meta in metadata],  # image_path
                [meta.get('image_hash', '') for meta in metadata],  # image_hash
                [meta.get('file_size', 0) for meta in metadata],    # file_size
                [meta.get('model_name', '') for meta in metadata],  # model_name
                vectors,                                            # embedding
                [current_time] * len(vectors),                      # created_at
                [meta.get('extra_metadata') if meta.get('extra_metadata') is not None else {} for meta in metadata]  # metadata
            ]
            
            # Insert data
            result = collection.insert(data)
            
            # Flush to ensure data is persisted
            collection.flush()
            
            logger.info(f"Inserted {len(vectors)} vectors into {collection_name}")
            return result.primary_keys
            
        except Exception as e:
            logger.error(f"Failed to insert vectors into {collection_name}: {e}")
            return None
    
    def search_vectors(self,
                      collection_name: str,
                      query_vectors: List[np.ndarray],
                      top_k: int = 10,
                      search_params: Optional[Dict] = None,
                      filter_expr: str = "",
                      output_fields: Optional[List[str]] = None) -> Optional[List[List[Dict]]]:
        """
        Search for similar vectors in a collection.
        
        Args:
            collection_name: Name of the collection
            query_vectors: Query vectors
            top_k: Number of top results to return
            search_params: Search parameters
            filter_expr: Filter expression
            output_fields: Fields to return in results
            
        Returns:
            Search results or None if failed
        """
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return None
            
            # Default search parameters
            if search_params is None:
                search_params = {"nprobe": 10}
            
            # Default output fields
            if output_fields is None:
                output_fields = ["image_path", "image_hash", "model_name", "created_at"]
            
            # Perform search
            search_results = collection.search(
                data=query_vectors,
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=output_fields
            )
            
            # Format results
            formatted_results = []
            for result in search_results:
                result_list = []
                for hit in result:
                    hit_dict = {
                        'id': hit.id,
                        'distance': hit.distance,
                        'score': 1.0 / (1.0 + hit.distance)  # Convert distance to similarity score
                    }
                    # Add output fields
                    for field in output_fields:
                        if hasattr(hit.entity, field):
                            hit_dict[field] = getattr(hit.entity, field)
                    result_list.append(hit_dict)
                formatted_results.append(result_list)
            
            logger.info(f"Search completed in {collection_name}, found {len(formatted_results)} result sets")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search vectors in {collection_name}: {e}")
            return None
    
    def delete_vectors(self,
                      collection_name: str,
                      ids: List[int]) -> bool:
        """
        Delete vectors by IDs.
        
        Args:
            collection_name: Name of the collection
            ids: List of vector IDs to delete
            
        Returns:
            bool: True if deletion successful
        """
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return False
            
            # Delete by IDs
            expr = f"id in {ids}"
            collection.delete(expr)
            
            # Flush to ensure deletion is persisted
            collection.flush()
            
            logger.info(f"Deleted {len(ids)} vectors from {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors from {collection_name}: {e}")
            return False
    
    def get_collection_stats(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary containing collection statistics
        """
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return None
            
            # Get collection info
            stats = {
                'name': collection_name,
                'description': collection.description,
                'num_entities': collection.num_entities,
                'schema': {
                    'fields': [
                        {
                            'name': field.name,
                            'type': field.dtype.name,
                            'is_primary': field.is_primary,
                            'auto_id': getattr(field, 'auto_id', False)
                        }
                        for field in collection.schema.fields
                    ]
                }
            }
            
            # Get index info
            try:
                indexes = collection.indexes
                stats['indexes'] = [
                    {
                        'field_name': idx.field_name,
                        'index_type': idx.params.get('index_type', 'unknown'),
                        'metric_type': idx.params.get('metric_type', 'unknown')
                    }
                    for idx in indexes
                ]
            except:
                stats['indexes'] = []
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats for {collection_name}: {e}")
            return None
    
    def list_collections(self) -> List[str]:
        """
        List all collections in the database.
        
        Returns:
            List of collection names
        """
        try:
            return utility.list_collections(using=self.alias)
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def drop_collection(self, collection_name: str) -> bool:
        """
        Drop a collection.
        
        Args:
            collection_name: Name of the collection to drop
            
        Returns:
            bool: True if successful
        """
        try:
            if collection_name in self.collections:
                del self.collections[collection_name]
            
            utility.drop_collection(collection_name, using=self.alias)
            logger.info(f"Dropped collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to drop collection {collection_name}: {e}")
            return False
    
    def _test_connection(self) -> bool:
        """Test the connection to Milvus."""
        try:
            # Try to list collections as a connection test
            utility.list_collections(using=self.alias)
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        try:
            return {
                'host': self.host,
                'port': self.port,
                'database': self.database,
                'alias': self.alias,
                'connected': self.connection is not None,
                'collections': list(self.collections.keys())
            }
        except Exception as e:
            logger.error(f"Failed to get server info: {e}")
            return {}