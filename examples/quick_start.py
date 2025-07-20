#!/usr/bin/env python3
"""
Quick Start Example for Image Vector SDK

This is the simplest possible example to get started with the Image Vector SDK.
Perfect for testing and initial setup.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sdk import ImageVectorSDK


def main():
    """Quick start example - minimal setup."""
    print("🚀 Image Vector SDK - Quick Start\n")
    
    # Step 1: Initialize SDK (will use default configuration)
    print("📋 Step 1: Initializing SDK...")
    
    try:
        sdk = ImageVectorSDK(auto_connect=True)
        
        if not sdk.connected:
            print("❌ Could not connect to Milvus database.")
            print("\n💡 Quick Setup Guide:")
            print("1. Install Milvus: https://milvus.io/docs/install_standalone-docker.md")
            print("2. Start Milvus: docker run -p 19530:19530 milvusdb/milvus:latest")
            print("3. Copy .env.example to .env and adjust settings if needed")
            return
        
        print("✅ SDK connected successfully!")
        
        # Step 2: Check health
        print("\n🔍 Step 2: Health check...")
        health = sdk.health_check()
        print(f"Overall status: {health['overall']['status']}")
        
        # Step 3: Create a sample image directory if it doesn't exist
        print("\n📁 Step 3: Setting up sample images...")
        sample_dir = Path("examples/sample_images")
        sample_dir.mkdir(exist_ok=True)
        
        # Check for images
        image_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
        
        if not image_files:
            print("📷 No sample images found.")
            print(f"   Add some .jpg or .png files to: {sample_dir}")
            print("   Then run this script again!")
            
            # Create instructions file
            instructions = sample_dir / "HOW_TO_ADD_IMAGES.txt"
            with open(instructions, 'w') as f:
                f.write("""How to add sample images:

1. Download some images from the internet (jpg or png format)
2. Copy them to this directory
3. Run the quick_start.py script again

The SDK will:
- Convert each image to a vector using CLIP
- Store vectors in Milvus database  
- Enable similarity search

Example images you can try:
- Photos of animals, people, objects
- Screenshots, artwork, logos
- Any image you want to search by similarity
""")
            print(f"   Created instructions: {instructions}")
            
        else:
            print(f"📷 Found {len(image_files)} sample images")
            
            # Step 4: Process first image
            print("\n⚙️  Step 4: Processing first image...")
            first_image = image_files[0]
            print(f"   Processing: {first_image.name}")
            
            vector_id = sdk.process_and_store_image(first_image)
            
            if vector_id:
                print(f"✅ Image processed! Vector ID: {vector_id}")
                
                # Step 5: Search for similar images
                print("\n🔎 Step 5: Finding similar images...")
                similar = sdk.search_similar_images(first_image, top_k=3)
                
                print(f"   Found {len(similar)} similar images:")
                for i, result in enumerate(similar, 1):
                    name = Path(result['image_path']).name
                    similarity = result['similarity']
                    print(f"   {i}. {name} (similarity: {similarity:.3f})")
                
                # Step 6: Show statistics
                print("\n📊 Step 6: Quick statistics...")
                stats = sdk.get_sdk_stats()
                print(f"   Images processed: {stats['images_processed']}")
                print(f"   Vectors stored: {stats['vectors_stored']}")
                print(f"   Searches performed: {stats['searches_performed']}")
                
            else:
                print("❌ Failed to process image")
        
        print("\n🎉 Quick start completed!")
        print("\nNext steps:")
        print("1. Try examples/basic_usage.py for more features")
        print("2. Try examples/advanced_usage.py for advanced features")
        print("3. Read README.md for detailed documentation")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Milvus is running (docker ps)")
        print("2. Check your .env configuration")
        print("3. Verify Python dependencies are installed")
        print("4. Run tests/test_sdk.py to diagnose issues")


if __name__ == "__main__":
    main()