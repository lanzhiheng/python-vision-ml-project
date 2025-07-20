#!/usr/bin/env python3
"""
Basic Usage Example for Image Vector SDK

This example demonstrates the basic functionality of the Image Vector SDK:
- Initializing the SDK
- Processing and storing images
- Searching for similar images
- Managing the vector database
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sdk import ImageVectorSDK
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Basic usage example."""
    print("=== Image Vector SDK - Basic Usage Example ===\n")
    
    # 1. Initialize the SDK
    print("1. Initializing SDK...")
    
    # You can specify custom config files or use defaults
    sdk = ImageVectorSDK(
        config_file="configs/sdk_config.yaml",
        env_file=".env",
        auto_connect=True
    )
    
    # Check if connection is successful
    if not sdk.connected:
        print("❌ Failed to connect to database. Please check your Milvus configuration.")
        print("Make sure Milvus is running and configuration is correct.")
        return
    
    print("✅ SDK initialized and connected to database")
    
    # 2. Health check
    print("\n2. Performing health check...")
    health = sdk.health_check()
    print(f"Overall health: {health['overall']['status']}")
    for component, status in health.items():
        if component != 'overall':
            print(f"  - {component}: {status['status']} - {status['details']}")
    
    # 3. Get configuration summary
    print("\n3. Configuration summary:")
    config_summary = sdk.get_config_summary()
    print(f"  - CLIP model: {config_summary['clip_model']}")
    print(f"  - Vector dimension: {config_summary['vector_dimension']}")
    print(f"  - Milvus host: {config_summary['milvus_host']}")
    
    # 4. Process a single image (if exists)
    print("\n4. Processing a single image...")
    
    # Create a sample directory with some test images
    sample_dir = Path("examples/sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    # Check if we have any sample images
    sample_images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
    
    if sample_images:
        print(f"Found {len(sample_images)} sample images")
        
        # Process the first image
        image_path = sample_images[0]
        print(f"Processing: {image_path}")
        
        # Add custom metadata
        metadata = {
            "category": "sample",
            "source": "example_script",
            "description": "Test image for SDK demonstration"
        }
        
        vector_id = sdk.process_and_store_image(image_path, metadata)
        
        if vector_id:
            print(f"✅ Image processed and stored with ID: {vector_id}")
            
            # 5. Search for similar images
            print("\n5. Search for similar images...")
            similar_images = sdk.search_similar_images(
                query_image=image_path,
                top_k=5,
                threshold=0.5
            )
            
            print(f"Found {len(similar_images)} similar images:")
            for i, result in enumerate(similar_images, 1):
                print(f"  {i}. {result['image_path']} (similarity: {result['similarity']:.3f})")
                
        else:
            print("❌ Failed to process and store image")
            
        # 6. Process multiple images if available
        if len(sample_images) > 1:
            print(f"\n6. Processing multiple images ({len(sample_images)} total)...")
            
            vector_ids = sdk.process_and_store_images(
                sample_images[1:],  # Skip the first one we already processed
                batch_size=2
            )
            
            print(f"✅ Processed and stored {len(vector_ids)} images")
    else:
        print("No sample images found. Creating placeholder instructions...")
        print("To test image processing:")
        print("1. Create 'examples/sample_images' directory")
        print("2. Add some .jpg or .png images to that directory")
        print("3. Run this example again")
        
        # Create a README file in the sample images directory
        readme_path = sample_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write("""# Sample Images Directory

Place your test images here to try the Image Vector SDK.

Supported formats:
- .jpg, .jpeg
- .png
- .bmp
- .tiff, .tif
- .webp

Example images you can use:
1. Download some images from the internet
2. Use your own photos
3. Generate images using AI tools

The SDK will:
1. Convert each image to a vector representation using CLIP
2. Store the vectors in Milvus database
3. Enable similarity search across all stored images
""")
    
    # 7. Get database statistics
    print("\n7. Database statistics:")
    db_stats = sdk.get_database_stats()
    print(f"  - Total vectors: {db_stats.get('total_vectors', 0)}")
    print(f"  - Connection status: {db_stats.get('connection_status', 'unknown')}")
    
    # 8. Get SDK statistics
    print("\n8. SDK usage statistics:")
    sdk_stats = sdk.get_sdk_stats()
    print(f"  - Images processed: {sdk_stats['images_processed']}")
    print(f"  - Vectors stored: {sdk_stats['vectors_stored']}")
    print(f"  - Searches performed: {sdk_stats['searches_performed']}")
    print(f"  - Session duration: {sdk_stats['session_duration_seconds']:.1f}s")
    
    # 9. Demonstrate vector extraction without storage
    if sample_images:
        print("\n9. Extract vector without storing...")
        vector = sdk.get_image_vector(sample_images[0])
        if vector is not None:
            print(f"✅ Extracted vector with dimension: {len(vector)}")
            print(f"Vector norm: {vector.dot(vector)**0.5:.3f}")
        else:
            print("❌ Failed to extract vector")
    
    # 10. Clean up (optional - uncomment to delete test vectors)
    # print("\n10. Cleaning up test data...")
    # if vector_ids:
    #     success = sdk.delete_vectors(ids=vector_ids)
    #     if success:
    #         print("✅ Test vectors deleted")
    #     else:
    #         print("❌ Failed to delete test vectors")
    
    print("\n=== Example completed ===")
    print("Check the logs for detailed information about each operation.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user")
    except Exception as e:
        logger.error(f"Example failed with error: {e}")
        print(f"\n❌ Example failed: {e}")
        print("Please check your configuration and ensure Milvus is running.")
    finally:
        print("Goodbye!")