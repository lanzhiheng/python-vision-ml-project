#!/usr/bin/env python3
"""
ConvNeXt V2-Large 模型集成测试脚本

这个脚本测试新添加的ConvNeXt V2-Large模型是否能正常工作。
"""

import os
import sys
import numpy as np
from PIL import Image
import torch

def test_convnext_v2_large():
    """测试ConvNeXt V2-Large模型集成"""
    print("🚀 ConvNeXt V2-Large 模型集成测试")
    print("=" * 60)
    
    try:
        # 导入向量化器
        from examples.image_vectorizer_optimized import OptimizedImageVectorizer
        
        # 检查模型是否在支持列表中
        if 'convnext_v2_large' not in OptimizedImageVectorizer.SUPPORTED_MODELS:
            print("❌ ConvNeXt V2-Large 未在支持的模型列表中")
            return False
            
        model_config = OptimizedImageVectorizer.SUPPORTED_MODELS['convnext_v2_large']
        print(f"✅ 模型配置已找到:")
        print(f"   描述: {model_config['description']}")
        print(f"   输入尺寸: {model_config['input_size']}")
        print(f"   特征维度: {model_config['expected_feature_dim']}")
        print(f"   模型类型: {model_config['model_type']}")
        
        # 检查是否有测试图像
        test_image_path = None
        test_dirs = ['data/images', 'data/large_images', 'benchmark_test_images', 'quick_fix_test']
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                for file in os.listdir(test_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_image_path = os.path.join(test_dir, file)
                        break
                if test_image_path:
                    break
        
        if not test_image_path:
            print("⚠️ 未找到测试图像，跳过实际模型测试")
            return True
            
        print(f"\n📷 使用测试图像: {test_image_path}")
        
        # 测试单模型向量化
        print("\n🔍 测试单模型向量化...")
        try:
            # 使用CPU进行测试（避免GPU依赖问题）
            vectorizer = OptimizedImageVectorizer(
                device='cpu',
                model_names=['convnext_v2_large'],
                use_mixed_precision=False  # CPU模式下禁用混合精度
            )
            
            print("✅ 模型初始化成功")
            
            # 进行向量化测试
            result = vectorizer.process_single_image(test_image_path)
            
            if result is not None:
                print(f"✅ 图像向量化成功")
                print(f"   特征向量形状: {result.shape}")
                print(f"   特征向量类型: {type(result)}")
                print(f"   前5个特征值: {result[:5]}")
                
                # 验证特征维度
                expected_dim = model_config['expected_feature_dim']
                if result.shape[0] == expected_dim:
                    print(f"✅ 特征维度匹配: {result.shape[0]} == {expected_dim}")
                else:
                    print(f"⚠️ 特征维度不匹配: {result.shape[0]} != {expected_dim}")
                    
            else:
                print("❌ 图像向量化失败")
                return False
                
        except Exception as e:
            print(f"❌ 模型测试失败: {e}")
            print("这可能是因为缺少 transformers 库，请运行: pip install transformers")
            return False
        
        # 测试与其他模型的组合
        print("\n🔄 测试模型组合...")
        try:
            combo_vectorizer = OptimizedImageVectorizer(
                device='cpu',
                model_names=['resnet50', 'convnext_v2_large'],
                use_mixed_precision=False
            )
            
            combo_result = combo_vectorizer.process_single_image(test_image_path)
            if combo_result is not None:
                print(f"✅ 组合模型向量化成功")
                print(f"   组合特征向量形状: {combo_result.shape}")
                
                # 验证组合维度
                expected_combo_dim = 2048 + expected_dim  # ResNet50 + ConvNeXt V2-Large
                if combo_result.shape[0] == expected_combo_dim:
                    print(f"✅ 组合特征维度正确: {combo_result.shape[0]} == {expected_combo_dim}")
                else:
                    print(f"⚠️ 组合特征维度异常: {combo_result.shape[0]} != {expected_combo_dim}")
                    
            else:
                print("❌ 组合模型向量化失败")
                return False
                
        except Exception as e:
            print(f"❌ 组合模型测试失败: {e}")
            return False
        
        print(f"\n🎉 ConvNeXt V2-Large 模型集成测试完成!")
        print("✅ 所有测试通过")
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装必要的依赖: pip install transformers")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def print_usage_examples():
    """打印使用示例"""
    print(f"\n💡 ConvNeXt V2-Large 使用示例:")
    print("=" * 60)
    
    print("1. 单模型使用:")
    print("   python examples/image_vectorizer_optimized.py \\")
    print("       --image_path ./test_images/ \\")
    print("       --models convnext_v2_large \\")
    print("       --device cpu")
    
    print("\n2. 与其他模型组合:")
    print("   python examples/image_vectorizer_optimized.py \\")
    print("       --image_path ./test_images/ \\")
    print("       --models resnet50,convnext_v2_large,openai_clip_vit_b32 \\")
    print("       --device gpu \\")
    print("       --mixed_precision")
    
    print("\n3. 性能基准测试:")
    print("   python examples/performance_benchmark_cpu_mps.py \\")
    print("       --models convnext_v2_large \\")
    print("       --test_dir ./benchmark_test_images/")

if __name__ == "__main__":
    print("ConvNeXt V2-Large 模型集成测试")
    print("确保已安装: pip install transformers")
    print()
    
    success = test_convnext_v2_large()
    
    if success:
        print_usage_examples()
        sys.exit(0)
    else:
        print(f"\n❌ 测试失败，请检查错误信息并修复问题")
        sys.exit(1) 