#!/usr/bin/env python3
"""
Advanced Usage Example for Image Vector SDK

This example demonstrates advanced features of the Image Vector SDK:
- Custom configuration management
- Batch processing with custom metadata
- Advanced search with filters
- Database management operations
- Performance optimization
"""

import sys
import os
from pathlib import Path
import numpy as np
import time
import json

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sdk import ImageVectorSDK, ConfigManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_custom_config():
    """Create a custom configuration for advanced usage."""
    config = {
        'milvus': {
            'host': 'localhost',
            'port': 19530,
            'collection_name': 'advanced_example_vectors',
            'metric_type': 'COSINE',  # Use cosine similarity
            'index_type': 'HNSW'      # Use HNSW for better performance
        },
        'clip': {
            'model_name': 'ViT-L/14',  # Use larger model for better accuracy
            'device': 'auto'
        },
        'vector': {
            'dimension': 768,  # ViT-L/14 has 768 dimensions
            'normalize': True
        },
        'performance': {
            'enable_cache': True,
            'cache_size': 2000,
            'parallel_processing': True,
            'max_workers': 8
        }
    }
    
    # Save custom config
    config_path = Path("examples/custom_config.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


def demonstrate_batch_processing(sdk: ImageVectorSDK):
    """Demonstrate advanced batch processing with metadata."""
    print("\n--- Advanced Batch Processing ---")
    
    # Create test image directory
    test_dir = Path("examples/test_dataset")
    test_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different categories
    categories = ['nature', 'architecture', 'people', 'objects']
    for category in categories:
        (test_dir / category).mkdir(exist_ok=True)
        
        # Create placeholder README files
        readme_path = test_dir / category / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f"""# {category.title()} Images

Place {category} images in this directory for categorized testing.

This directory is used for demonstrating:
- Batch processing with metadata
- Category-based filtering
- Advanced search operations
""")
    
    # Find all images in the test dataset
    image_paths = []
    metadata_list = []
    
    for category in categories:
        category_dir = test_dir / category
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in category_dir.glob(ext):
                image_paths.append(img_path)
                metadata_list.append({
                    'category': category,
                    'source_directory': str(category_dir),
                    'filename': img_path.name,
                    'processing_timestamp': time.time()
                })
    
    if image_paths:
        print(f"Found {len(image_paths)} images across {len(categories)} categories")
        
        # Process images with metadata
        start_time = time.time()
        vector_ids = sdk.process_and_store_images(
            image_paths,
            metadata_list=metadata_list,
            batch_size=4,
            parallel=True
        )
        processing_time = time.time() - start_time
        
        print(f"✅ Processed {len(vector_ids)} images in {processing_time:.2f}s")
        print(f"Average: {processing_time/max(len(vector_ids), 1):.3f}s per image")
        
        return vector_ids, categories
    else:
        print("No test images found. Created directory structure for future testing.")
        return [], categories


def demonstrate_advanced_search(sdk: ImageVectorSDK, categories: list):
    """Demonstrate advanced search capabilities with filters."""
    print("\n--- Advanced Search Operations ---")
    
    # Find a sample image to use as query
    test_dir = Path("examples/test_dataset")
    sample_images = []
    
    for category in categories:
        category_dir = test_dir / category
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            sample_images.extend(list(category_dir.glob(ext)))
    
    if not sample_images:
        print("No sample images available for search demonstration")
        return
    
    query_image = sample_images[0]
    print(f"Using query image: {query_image}")
    
    # 1. Basic similarity search
    print("\n1. Basic similarity search:")
    results = sdk.search_similar_images(
        query_image=query_image,
        top_k=10,
        threshold=0.1
    )
    
    print(f"Found {len(results)} similar images")
    for i, result in enumerate(results[:3], 1):
        print(f"  {i}. {Path(result['image_path']).name} - similarity: {result['similarity']:.3f}")
    
    # 2. Category-filtered search
    print("\n2. Category-filtered search:")
    filters = {'model_name': 'ViT-L/14'}  # Example filter
    
    filtered_results = sdk.search_similar_images(
        query_image=query_image,
        top_k=5,
        threshold=0.1,
        filters=filters
    )
    
    print(f"Found {len(filtered_results)} filtered results")
    
    # 3. Threshold-based search
    print("\n3. High-threshold search (very similar only):")
    precise_results = sdk.search_similar_images(
        query_image=query_image,
        top_k=10,
        threshold=0.8
    )
    
    print(f"Found {len(precise_results)} highly similar images")


def demonstrate_database_management(sdk: ImageVectorSDK):
    """Demonstrate database management operations."""
    print("\n--- Database Management ---")
    
    # 1. Get detailed statistics
    print("1. Database statistics:")
    db_stats = sdk.get_database_stats()
    
    for key, value in db_stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    
    # 2. SDK performance statistics
    print("\n2. SDK performance statistics:")
    sdk_stats = sdk.get_sdk_stats()
    
    print(f"  - Images processed: {sdk_stats['images_processed']}")
    print(f"  - Vectors stored: {sdk_stats['vectors_stored']}")
    print(f"  - Searches performed: {sdk_stats['searches_performed']}")
    print(f"  - Average processing time: {sdk_stats['avg_processing_time_per_image']:.3f}s")
    print(f"  - Cache hit rate: {sdk_stats['vectorizer_cache_stats'].get('hit_rate', 0):.2%}")
    
    # 3. Index management
    print("\n3. Index management:")
    print("Rebuilding index with optimized parameters...")
    
    success = sdk.rebuild_index(
        index_type='HNSW',
        metric_type='COSINE'
    )
    
    if success:
        print("✅ Index rebuilt successfully")
    else:
        print("❌ Index rebuild failed")


def demonstrate_configuration_management():
    """Demonstrate advanced configuration management."""
    print("\n--- Configuration Management ---")
    
    # Create a custom configuration manager
    config_manager = ConfigManager(
        config_file="examples/custom_config.yaml",
        env_file=".env"
    )
    
    # 1. Show configuration summary
    print("1. Configuration summary:")
    summary = config_manager.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 2. Validate configuration
    print("\n2. Configuration validation:")
    errors = config_manager.validate_config()
    
    if errors:
        print("Configuration errors found:")
        for category, error_list in errors.items():
            print(f"  {category}: {error_list}")
    else:
        print("✅ Configuration is valid")
    
    # 3. Dynamic configuration updates
    print("\n3. Dynamic configuration updates:")
    original_cache_size = config_manager.get('performance.cache_size')
    print(f"Original cache size: {original_cache_size}")
    
    # Update cache size
    config_manager.set('performance.cache_size', 5000)
    new_cache_size = config_manager.get('performance.cache_size')
    print(f"Updated cache size: {new_cache_size}")
    
    # Save updated configuration
    config_manager.save_config("examples/updated_config.yaml")
    print("✅ Updated configuration saved")


def demonstrate_performance_optimization(sdk: ImageVectorSDK):
    """Demonstrate performance optimization techniques."""
    print("\n--- Performance Optimization ---")
    
    # Create some test vectors for performance testing
    test_vectors = [np.random.random(768) for _ in range(100)]
    
    # 1. Batch vs individual processing comparison
    print("1. Performance comparison:")
    
    # Individual processing (simulated)
    individual_time = 0.05 * len(test_vectors)  # Simulate 50ms per image
    print(f"Estimated individual processing time: {individual_time:.2f}s")
    
    # Actual batch processing time from SDK stats
    sdk_stats = sdk.get_sdk_stats()
    if sdk_stats['images_processed'] > 0:
        actual_time = sdk_stats['avg_processing_time_per_image'] * sdk_stats['images_processed']
        print(f"Actual batch processing time: {actual_time:.2f}s")
        
        if actual_time < individual_time:
            speedup = individual_time / actual_time
            print(f"✅ Batch processing {speedup:.1f}x faster")
    
    # 2. Cache effectiveness
    print("\n2. Cache effectiveness:")
    cache_stats = sdk_stats['vectorizer_cache_stats']
    
    print(f"  - Cache enabled: {cache_stats['cache_enabled']}")
    print(f"  - Cache size: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
    print(f"  - Hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"  - Total hits: {cache_stats['hits']}")
    print(f"  - Total misses: {cache_stats['misses']}")
    
    # 3. Memory usage optimization tips
    print("\n3. Optimization recommendations:")
    
    recommendations = []
    
    if cache_stats['hit_rate'] < 0.5:
        recommendations.append("Consider increasing cache size for better performance")
    
    if sdk_stats['images_processed'] > 100:
        recommendations.append("For large datasets, consider processing in smaller batches to reduce memory usage")
    
    if not recommendations:
        recommendations.append("Current configuration appears optimal")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")


def main():
    """Advanced usage demonstration."""
    print("=== Image Vector SDK - Advanced Usage Example ===\n")
    
    # 1. Create custom configuration
    print("1. Creating custom configuration...")
    custom_config_path = create_custom_config()
    print(f"✅ Custom configuration created: {custom_config_path}")
    
    # 2. Demonstrate configuration management
    demonstrate_configuration_management()
    
    # 3. Initialize SDK with custom configuration
    print("\n3. Initializing SDK with custom configuration...")
    
    try:
        sdk = ImageVectorSDK(
            config_file=custom_config_path,
            env_file=".env",
            auto_connect=True
        )
        
        if not sdk.connected:
            print("❌ Failed to connect to database")
            return
        
        print("✅ SDK initialized with custom configuration")
        
        # 4. Demonstrate batch processing
        vector_ids, categories = demonstrate_batch_processing(sdk)
        
        # 5. Demonstrate advanced search
        demonstrate_advanced_search(sdk, categories)
        
        # 6. Demonstrate database management
        demonstrate_database_management(sdk)
        
        # 7. Demonstrate performance optimization
        demonstrate_performance_optimization(sdk)
        
        # 8. Export functionality demonstration
        print("\n--- Data Export/Import ---")
        print("Note: Export/Import functionality requires additional implementation")
        print("This would allow you to:")
        print("  - Export vector database to files")
        print("  - Import vectors from backup files")
        print("  - Migrate data between different Milvus instances")
        
    except Exception as e:
        logger.error(f"Advanced example failed: {e}")
        print(f"❌ Advanced example failed: {e}")
    
    print("\n=== Advanced Example Completed ===")
    print("This example demonstrated:")
    print("- Custom configuration management")
    print("- Advanced batch processing")
    print("- Sophisticated search operations")
    print("- Database management")
    print("- Performance optimization")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAdvanced example interrupted by user")
    except Exception as e:
        logger.error(f"Example failed with error: {e}")
        print(f"\n❌ Example failed: {e}")
    finally:
        print("Goodbye!")