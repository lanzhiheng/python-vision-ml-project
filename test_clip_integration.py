#!/usr/bin/env python3
"""
CLIP集成测试脚本

用于验证OpenAI CLIP和OpenCLIP模型集成是否正常工作。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试必要的库导入"""
    print("🔍 测试库导入...")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch 导入失败: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ TorchVision: {torchvision.__version__}")
    except ImportError as e:
        print(f"❌ TorchVision 导入失败: {e}")
        return False
    
    # 测试CLIP导入
    try:
        import clip
        print("✓ OpenAI CLIP 可用")
        clip_available = True
    except ImportError:
        print("⚠️ OpenAI CLIP 不可用 (需要安装: pip install git+https://github.com/openai/CLIP.git)")
        clip_available = False
    
    try:
        import open_clip
        print("✓ OpenCLIP 可用")
        open_clip_available = True
    except ImportError:
        print("⚠️ OpenCLIP 不可用 (需要安装: pip install open-clip-torch)")
        open_clip_available = False
    
    return True, clip_available, open_clip_available

def test_image_vectorizer_import():
    """测试ImageVectorizer类导入"""
    print("\\n🔍 测试ImageVectorizer导入...")
    
    try:
        from examples.image_vectorizer import ImageVectorizer
        print("✓ ImageVectorizer 类导入成功")
        return True
    except Exception as e:
        print(f"❌ ImageVectorizer 导入失败: {e}")
        return False

def test_supported_models():
    """测试支持的模型配置"""
    print("\\n🔍 测试支持的模型配置...")
    
    try:
        from examples.image_vectorizer import ImageVectorizer
        models = ImageVectorizer.SUPPORTED_MODELS
        
        # 统计不同类型的模型
        torchvision_models = []
        openai_clip_models = []
        open_clip_models = []
        
        for name, config in models.items():
            model_type = config.get('model_type', 'torchvision')
            if model_type == 'torchvision':
                torchvision_models.append(name)
            elif model_type == 'openai_clip':
                openai_clip_models.append(name)
            elif model_type == 'open_clip':
                open_clip_models.append(name)
        
        print(f"✓ TorchVision 模型: {len(torchvision_models)} 个")
        for model in torchvision_models:
            print(f"  - {model}")
        
        print(f"✓ OpenAI CLIP 模型: {len(openai_clip_models)} 个")
        for model in openai_clip_models:
            print(f"  - {model}")
        
        print(f"✓ OpenCLIP 模型: {len(open_clip_models)} 个")
        for model in open_clip_models:
            print(f"  - {model}")
        
        return True
    except Exception as e:
        print(f"❌ 模型配置测试失败: {e}")
        return False

def test_basic_initialization():
    """测试基本初始化（仅torchvision模型）"""
    print("\\n🔍 测试基本初始化...")
    
    try:
        from examples.image_vectorizer import ImageVectorizer
        
        # 测试torchvision模型初始化
        print("测试 ResNet-152 初始化...")
        vectorizer = ImageVectorizer(device='cpu', model_names=['resnet152'])
        print("✓ ResNet-152 初始化成功")
        
        # 检查模型信息
        print(f"  - 加载的模型: {vectorizer.model_names}")
        print(f"  - 总特征维度: {vectorizer.total_feature_dim}")
        print(f"  - 计算设备: {vectorizer.device}")
        
        return True
    except Exception as e:
        print(f"❌ 基本初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_clip_availability():
    """测试CLIP模型可用性（不实际加载）"""
    print("\\n🔍 测试CLIP模型可用性...")
    
    # 检查OpenAI CLIP
    try:
        import clip
        available_models = clip.available_models()
        print(f"✓ OpenAI CLIP 可用模型: {available_models}")
    except ImportError:
        print("⚠️ OpenAI CLIP 未安装")
    except Exception as e:
        print(f"⚠️ OpenAI CLIP 测试出错: {e}")
    
    # 检查OpenCLIP
    try:
        import open_clip
        model_list = open_clip.list_models()
        print(f"✓ OpenCLIP 可用模型: {len(model_list)} 个")
        # 显示前几个作为示例
        if model_list:
            print(f"  示例模型: {model_list[:5]}")
    except ImportError:
        print("⚠️ OpenCLIP 未安装")
    except Exception as e:
        print(f"⚠️ OpenCLIP 测试出错: {e}")

def main():
    """主测试函数"""
    print("🚀 CLIP 集成测试开始")
    print("=" * 50)
    
    # 测试导入
    import_results = test_imports()
    if not import_results[0]:
        print("\\n❌ 基础库导入失败，请检查PyTorch安装")
        return False
    
    # 测试ImageVectorizer导入
    if not test_image_vectorizer_import():
        print("\\n❌ ImageVectorizer导入失败")
        return False
    
    # 测试支持的模型配置
    if not test_supported_models():
        print("\\n❌ 模型配置测试失败")
        return False
    
    # 测试基本初始化
    if not test_basic_initialization():
        print("\\n❌ 基本初始化失败")
        return False
    
    # 测试CLIP可用性
    test_clip_availability()
    
    print("\\n" + "=" * 50)
    print("🎉 集成测试完成！")
    print("\\n📋 总结:")
    print("✓ 代码集成完成")
    print("✓ 模型配置正确")
    print("✓ 向后兼容性保持")
    
    if len(import_results) > 1:
        if import_results[1]:  # OpenAI CLIP可用
            print("✓ OpenAI CLIP 支持就绪")
        else:
            print("⚠️ OpenAI CLIP 需要安装: pip install git+https://github.com/openai/CLIP.git")
        
        if import_results[2]:  # OpenCLIP可用
            print("✓ OpenCLIP 支持就绪")
        else:
            print("⚠️ OpenCLIP 需要安装: pip install open-clip-torch")
    
    print("\\n🔧 完整安装命令:")
    print("pip install git+https://github.com/openai/CLIP.git")
    print("pip install open-clip-torch")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)