#!/usr/bin/env python3
"""
Test Suite for Image Vector SDK

This module contains comprehensive tests for all SDK components:
- Configuration management
- Image vectorization
- Database operations
- Integration tests
"""

import sys
import os
import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import yaml
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sdk import ImageVectorSDK, ConfigManager
from vectorization import CLIPEncoder, ImageVectorizer
from database import MilvusClient, VectorStore


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
        self.env_file = Path(self.temp_dir) / ".env"
        
        # Create test configuration
        test_config = {
            'milvus': {
                'host': '${MILVUS_HOST:localhost}',
                'port': '${MILVUS_PORT:19530}',
                'collection_name': 'test_collection'
            },
            'clip': {
                'model_name': 'ViT-B/32',
                'device': 'cpu'
            },
            'vector': {
                'dimension': 512,
                'normalize': True
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        # Create test .env file
        with open(self.env_file, 'w') as f:
            f.write("MILVUS_HOST=test-host\n")
            f.write("MILVUS_PORT=19530\n")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_config_loading(self):
        """Test configuration loading."""
        config_manager = ConfigManager(self.config_file, self.env_file)
        
        # Test basic configuration access
        self.assertEqual(config_manager.get('milvus.host'), 'test-host')
        self.assertEqual(config_manager.get('milvus.port'), 19530)
        self.assertEqual(config_manager.get('clip.model_name'), 'ViT-B/32')
    
    def test_env_substitution(self):
        """Test environment variable substitution."""
        config_manager = ConfigManager(self.config_file, self.env_file)
        
        # Test that environment variables are substituted
        self.assertEqual(config_manager.get('milvus.host'), 'test-host')
        
        # Test default values
        self.assertEqual(config_manager.get('nonexistent.key', 'default'), 'default')
    
    def test_config_validation(self):
        """Test configuration validation."""
        config_manager = ConfigManager(self.config_file, self.env_file)
        
        errors = config_manager.validate_config()
        # Should have no errors with valid configuration
        self.assertEqual(len(errors), 0)
    
    def test_specialized_config_getters(self):
        """Test specialized configuration getters."""
        config_manager = ConfigManager(self.config_file, self.env_file)
        
        milvus_config = config_manager.get_milvus_config()
        self.assertIn('host', milvus_config)
        self.assertIn('port', milvus_config)
        
        clip_config = config_manager.get_clip_config()
        self.assertIn('model_name', clip_config)
        self.assertIn('device', clip_config)


class TestCLIPEncoder(unittest.TestCase):
    """Test cases for CLIPEncoder."""
    
    @patch('open_clip.create_model_and_transforms')
    @patch('open_clip.get_tokenizer')
    def test_clip_encoder_initialization(self, mock_tokenizer, mock_create_model):
        """Test CLIP encoder initialization."""
        # Mock the model creation
        mock_model = Mock()
        mock_model.visual.output_dim = 512
        mock_model.eval = Mock()
        
        mock_preprocess = Mock()
        mock_tokenizer_instance = Mock()
        
        mock_create_model.return_value = (mock_model, None, mock_preprocess)
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Initialize encoder
        encoder = CLIPEncoder(model_name="ViT-B/32", device="cpu")
        
        # Verify initialization
        self.assertEqual(encoder.model_name, "ViT-B/32")
        self.assertEqual(encoder.embedding_dim, 512)
        mock_create_model.assert_called_once()
        mock_tokenizer.assert_called_once()
    
    @patch('open_clip.create_model_and_transforms')
    @patch('open_clip.get_tokenizer')
    def test_device_selection(self, mock_tokenizer, mock_create_model):
        """Test device selection logic."""
        mock_model = Mock()
        mock_model.visual.output_dim = 512
        mock_model.eval = Mock()
        mock_preprocess = Mock()
        mock_tokenizer_instance = Mock()
        
        mock_create_model.return_value = (mock_model, None, mock_preprocess)
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Test auto device selection
        encoder = CLIPEncoder(device="auto")
        self.assertIsNotNone(encoder.device)
        
        # Test specific device
        encoder_cpu = CLIPEncoder(device="cpu")
        self.assertEqual(str(encoder_cpu.device), "cpu")


class TestImageVectorizer(unittest.TestCase):
    """Test cases for ImageVectorizer."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock the CLIP encoder to avoid loading actual models
        self.mock_encoder = Mock()
        self.mock_encoder.encode_image.return_value = np.random.random(512)
        self.mock_encoder.encode_images_batch.return_value = [np.random.random(512) for _ in range(3)]
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    @patch('vectorization.image_vectorizer.CLIPEncoder')
    def test_vectorizer_initialization(self, mock_clip_encoder):
        """Test image vectorizer initialization."""
        mock_clip_encoder.return_value = self.mock_encoder
        
        vectorizer = ImageVectorizer(
            model_name="ViT-B/32",
            cache_enabled=True,
            cache_size=100
        )
        
        self.assertEqual(vectorizer.model_name, "ViT-B/32")
        self.assertTrue(vectorizer.cache_enabled)
        self.assertEqual(vectorizer.cache_size, 100)
    
    @patch('vectorization.image_vectorizer.CLIPEncoder')
    def test_cache_functionality(self, mock_clip_encoder):
        """Test caching functionality."""
        mock_clip_encoder.return_value = self.mock_encoder
        
        vectorizer = ImageVectorizer(cache_enabled=True)
        
        # Test cache statistics
        stats = vectorizer.get_cache_stats()
        self.assertIn('cache_enabled', stats)
        self.assertIn('hit_rate', stats)
        
        # Test cache clearing
        vectorizer.clear_cache()
        self.assertEqual(len(vectorizer.embedding_cache), 0)


class TestMilvusClient(unittest.TestCase):
    """Test cases for MilvusClient."""
    
    @patch('database.milvus_client.connections')
    @patch('database.milvus_client.utility')
    def test_milvus_client_initialization(self, mock_utility, mock_connections):
        """Test Milvus client initialization."""
        client = MilvusClient(
            host="localhost",
            port=19530,
            database="test_db"
        )
        
        self.assertEqual(client.host, "localhost")
        self.assertEqual(client.port, 19530)
        self.assertEqual(client.database, "test_db")
    
    @patch('database.milvus_client.connections')
    @patch('database.milvus_client.utility')
    def test_connection(self, mock_utility, mock_connections):
        """Test database connection."""
        # Mock successful connection
        mock_utility.list_collections.return_value = []
        
        client = MilvusClient()
        success = client.connect()
        
        self.assertTrue(success)
        mock_connections.connect.assert_called_once()
    
    @patch('database.milvus_client.connections')
    @patch('database.milvus_client.utility')
    @patch('database.milvus_client.Collection')
    def test_collection_creation(self, mock_collection, mock_utility, mock_connections):
        """Test collection creation."""
        # Mock that collection doesn't exist
        mock_utility.has_collection.return_value = False
        
        # Mock collection instance
        mock_collection_instance = Mock()
        mock_collection_instance.create_index = Mock()
        mock_collection_instance.load = Mock()
        mock_collection.return_value = mock_collection_instance
        
        client = MilvusClient()
        client.connection = True  # Simulate connected state
        
        success = client.create_collection(
            collection_name="test_collection",
            dimension=512
        )
        
        self.assertTrue(success)
        mock_collection.assert_called_once()


class TestVectorStore(unittest.TestCase):
    """Test cases for VectorStore."""
    
    def setUp(self):
        """Set up test environment."""
        self.milvus_config = {
            'host': 'localhost',
            'port': 19530,
            'database': 'test'
        }
        
        # Mock Milvus client
        self.mock_client = Mock()
        self.mock_client.connect.return_value = True
        self.mock_client.insert_vectors.return_value = [1, 2, 3]
        self.mock_client.search_vectors.return_value = [[
            {'id': 1, 'distance': 0.1, 'score': 0.9, 'image_path': 'test.jpg'}
        ]]
    
    @patch('database.vector_store.MilvusClient')
    def test_vector_store_initialization(self, mock_milvus_client):
        """Test vector store initialization."""
        mock_milvus_client.return_value = self.mock_client
        
        store = VectorStore(
            milvus_config=self.milvus_config,
            collection_name="test_vectors",
            dimension=512
        )
        
        self.assertEqual(store.collection_name, "test_vectors")
        self.assertEqual(store.dimension, 512)
    
    @patch('database.vector_store.MilvusClient')
    def test_vector_storage(self, mock_milvus_client):
        """Test vector storage functionality."""
        mock_milvus_client.return_value = self.mock_client
        
        store = VectorStore(self.milvus_config)
        store.connected = True  # Simulate connected state
        
        # Test data
        test_vectors = [np.random.random(512) for _ in range(3)]
        test_metadata = [
            {'image_path': 'test1.jpg', 'model_name': 'ViT-B/32'},
            {'image_path': 'test2.jpg', 'model_name': 'ViT-B/32'},
            {'image_path': 'test3.jpg', 'model_name': 'ViT-B/32'}
        ]
        
        image_data = [
            {'embedding': vec, **meta}
            for vec, meta in zip(test_vectors, test_metadata)
        ]
        
        ids = store.store_image_vectors(image_data)
        
        self.assertEqual(ids, [1, 2, 3])
        self.mock_client.insert_vectors.assert_called_once()
    
    @patch('database.vector_store.MilvusClient')
    def test_vector_search(self, mock_milvus_client):
        """Test vector search functionality."""
        mock_milvus_client.return_value = self.mock_client
        
        store = VectorStore(self.milvus_config)
        store.connected = True  # Simulate connected state
        
        query_vector = np.random.random(512)
        results = store.search_similar_images(query_vector, top_k=5)
        
        self.assertEqual(len(results), 1)
        self.assertIn('similarity', results[0])
        self.mock_client.search_vectors.assert_called_once()


class TestImageVectorSDK(unittest.TestCase):
    """Integration tests for the main SDK."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration files
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
        self.env_file = Path(self.temp_dir) / ".env"
        
        test_config = {
            'milvus': {
                'host': 'localhost',
                'port': 19530,
                'collection_name': 'test_collection'
            },
            'clip': {
                'model_name': 'ViT-B/32',
                'device': 'cpu'
            },
            'vector': {
                'dimension': 512
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        with open(self.env_file, 'w') as f:
            f.write("MILVUS_HOST=localhost\n")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    @patch('sdk.image_vector_sdk.ImageVectorizer')
    @patch('sdk.image_vector_sdk.VectorStore')
    def test_sdk_initialization(self, mock_vector_store, mock_vectorizer):
        """Test SDK initialization."""
        # Mock components
        mock_vectorizer_instance = Mock()
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.connect.return_value = True
        
        mock_vectorizer.return_value = mock_vectorizer_instance
        mock_vector_store.return_value = mock_vector_store_instance
        
        sdk = ImageVectorSDK(
            config_file=self.config_file,
            env_file=self.env_file,
            auto_connect=True
        )
        
        self.assertTrue(sdk.connected)
        mock_vectorizer.assert_called_once()
        mock_vector_store.assert_called_once()
    
    @patch('sdk.image_vector_sdk.ImageVectorizer')
    @patch('sdk.image_vector_sdk.VectorStore')
    def test_health_check(self, mock_vector_store, mock_vectorizer):
        """Test SDK health check."""
        # Mock components
        mock_vectorizer_instance = Mock()
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.connect.return_value = True
        mock_vector_store_instance.get_stats.return_value = {'total_vectors': 100}
        
        mock_vectorizer.return_value = mock_vectorizer_instance
        mock_vector_store.return_value = mock_vector_store_instance
        
        sdk = ImageVectorSDK(
            config_file=self.config_file,
            env_file=self.env_file,
            auto_connect=False
        )
        
        # Simulate connection
        sdk.connected = True
        
        health = sdk.health_check()
        
        self.assertIn('overall', health)
        self.assertIn('config', health)
        self.assertIn('vectorizer', health)
        self.assertIn('database', health)
    
    @patch('sdk.image_vector_sdk.ImageVectorizer')
    @patch('sdk.image_vector_sdk.VectorStore')
    def test_statistics(self, mock_vector_store, mock_vectorizer):
        """Test SDK statistics tracking."""
        # Mock components
        mock_vectorizer_instance = Mock()
        mock_vectorizer_instance.get_cache_stats.return_value = {
            'cache_enabled': True,
            'hit_rate': 0.75
        }
        
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.connect.return_value = True
        
        mock_vectorizer.return_value = mock_vectorizer_instance
        mock_vector_store.return_value = mock_vector_store_instance
        
        sdk = ImageVectorSDK(
            config_file=self.config_file,
            env_file=self.env_file,
            auto_connect=False
        )
        
        # Update statistics
        sdk.stats['images_processed'] = 50
        sdk.stats['vectors_stored'] = 45
        sdk.stats['searches_performed'] = 10
        
        stats = sdk.get_sdk_stats()
        
        self.assertEqual(stats['images_processed'], 50)
        self.assertEqual(stats['vectors_stored'], 45)
        self.assertEqual(stats['searches_performed'], 10)
        self.assertIn('session_duration_seconds', stats)


def create_test_suite():
    """Create and return the test suite."""
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestConfigManager))
    suite.addTest(unittest.makeSuite(TestCLIPEncoder))
    suite.addTest(unittest.makeSuite(TestImageVectorizer))
    suite.addTest(unittest.makeSuite(TestMilvusClient))
    suite.addTest(unittest.makeSuite(TestVectorStore))
    suite.addTest(unittest.makeSuite(TestImageVectorSDK))
    
    return suite


def main():
    """Run the test suite."""
    print("=== Image Vector SDK - Test Suite ===\n")
    
    # Create test suite
    suite = create_test_suite()
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True
    )
    
    result = runner.run(suite)
    
    # Print summary
    print(f"\n=== Test Results ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())