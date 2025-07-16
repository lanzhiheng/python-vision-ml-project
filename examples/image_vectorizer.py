#!/usr/bin/env python3
"""
图像向量化工具 - 多模型集成 & CPU/GPU性能对比

这个脚本使用多个预训练模型集成对图像进行向量化：
- ResNet-152: 深度残差网络 (2048维)
- EfficientNet-B7: 高效网络 (2560维)
- Vision Transformer (ViT-L/16): 现代Transformer架构 (1024维)

特点:
- 显著增加计算复杂度，便于清晰对比CPU/GPU性能差异
- 最终特征维度: 5632维 (所有模型拼接)
- 支持动态模型选择和批量处理

使用方法:
    python examples/image_vectorizer.py --image_path /path/to/image.jpg --device cpu
    python examples/image_vectorizer.py --image_path /path/to/images/ --device gpu --batch_size 8
    python examples/image_vectorizer.py --models resnet152,vit_l_16 --image_path image.jpg
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import time
import numpy as np
from typing import Dict, Union, Optional
import sys

class ImageVectorizer:
    """
    图像向量化器，使用多模型集成提取图像特征向量。
    
    特点：
    - 支持模型集成 (ResNet-152, EfficientNet-B7, ViT-L/16)
    - 显著增加计算复杂度，便于对比CPU/GPU性能
    - 支持单张图片和批量处理
    - 自动设备检测和切换
    - 详细的性能统计
    - 动态模型选择
    """

    # 支持的模型配置
    SUPPORTED_MODELS = {
        'resnet152': {
            'input_size': 224,
            'expected_feature_dim': 2048,
            'description': 'ResNet-152 深度残差网络'
        },
        'resnet101': {
            'input_size': 224,
            'expected_feature_dim': 2048,
            'description': 'ResNet-101 深度残差网络'
        },
        'resnet50': {
            'input_size': 224,
            'expected_feature_dim': 2048,
            'description': 'ResNet-50 深度残差网络'
        },
        'efficientnet_b7': {
            'input_size': 600,
            'expected_feature_dim': 2560,
            'description': 'EfficientNet-B7 高效网络'
        },
        'vit_l_16': {
            'input_size': 224, 
            'expected_feature_dim': 1024,
            'description': 'Vision Transformer Large'
        }
    }

    def __init__(self, device: str = 'cpu', model_names: list = None):
        """
        初始化向量化器。

        Args:
            device (str): 使用的设备 ('cpu' 或 'gpu')
            model_names (list): 要使用的模型列表，默认使用所有模型
        """
        self.device_name = device
        self.model_names = model_names if model_names else ['resnet152']  # 默认使用单个模型进行测试
        self.models = {}
        self.feature_extractors = {}
        self.preprocessors = {}
        self.device = None
        self.total_feature_dim = 0
        self._setup_models()

    def _setup_models(self):
        """
        加载多个预训练模型并设置预处理管道。
        """
        print(f"正在设置模型集成 {self.model_names} 在设备: '{self.device_name}'")
        
        # 1. 设置计算设备
        if self.device_name in ('gpu', 'cuda') and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"✓ CUDA GPU已选择。使用设备: {torch.cuda.get_device_name(0)}")
            print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        elif self.device_name == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("✓ Apple Silicon GPU (MPS) 已选择")
        else:
            if self.device_name in ('gpu', 'cuda'):
                print("⚠️ CUDA GPU不可用，回退到CPU")
            elif self.device_name == 'mps':
                print("⚠️ Apple Silicon GPU (MPS) 不可用，回退到CPU")
            self.device = torch.device("cpu")
            print("✓ 使用设备: CPU")

        # 2. 逐一加载所有模型
        print("正在加载多个预训练模型...")
        print("=" * 50)
        
        successful_models = []
        self.total_feature_dim = 0
        
        for model_name in self.model_names:
            if model_name not in self.SUPPORTED_MODELS:
                print(f"⚠️ 不支持的模型: {model_name}，跳过")
                continue
                
            try:
                model_config = self.SUPPORTED_MODELS[model_name]
                print(f"正在加载: {model_config['description']} ({model_name})")
                
                # 加载模型
                if model_name == 'resnet152':
                    model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
                elif model_name == 'resnet101':
                    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
                elif model_name == 'resnet50':
                    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                elif model_name == 'efficientnet_b7':
                    try:
                        model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
                    except AttributeError:
                        print(f"  EfficientNet-B7 不可用，替换为 ResNet-101")
                        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
                        model_name = 'resnet101'  # 更新模型名称
                elif model_name == 'vit_l_16':
                    try:
                        model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
                    except AttributeError:
                        print(f"  Vision Transformer 不可用，替换为 ResNet-50")
                        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                        model_name = 'resnet50'  # 更新模型名称
                else:
                    raise ValueError(f"Unsupported model: {model_name}")
                
                # 移除最后的分类层，获取特征提取器
                if 'vit' in model_name:
                    # Vision Transformer 的特殊处理 - 移除heads部分
                    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
                else:
                    # ResNet 和 EfficientNet 的处理 - 移除最后的分类层
                    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
                
                feature_extractor.eval()
                
                # 移动到设备
                print(f"  正在将 {model_name} 移动到计算设备...")
                start_time = time.time()
                feature_extractor.to(self.device)
                end_time = time.time()
                
                # 设置预处理管道
                input_size = model_config['input_size']
                preprocessor = transforms.Compose([
                    transforms.Resize(int(input_size * 1.14)),  # 略大于目标尺寸
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]
                    ),
                ])
                
                # 获取实际特征维度
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, input_size, input_size).to(self.device)
                    dummy_output = feature_extractor(dummy_input)
                    actual_feature_dim = dummy_output.squeeze().numel()
                
                # 保存模型组件
                self.models[model_name] = model
                self.feature_extractors[model_name] = feature_extractor
                self.preprocessors[model_name] = preprocessor
                self.total_feature_dim += actual_feature_dim
                
                successful_models.append(model_name)
                print(f"✓ {model_name} 加载完成 (维度: {actual_feature_dim}, 耗时: {end_time - start_time:.2f}秒)")
                
            except Exception as e:
                print(f"❌ 模型 {model_name} 加载失败: {e}")
                print(f"  将继续加载其他模型...")
                continue
        
        if not successful_models:
            raise RuntimeError("所有模型加载都失败！")
        
        self.model_names = successful_models
        print("=" * 50)
        print(f"✓ 模型集成初始化完成！")
        print(f"  加载成功的模型: {self.model_names}")
        print(f"  总特征维度: {self.total_feature_dim}")
        print(f"  计算复杂度倍数: {len(self.model_names)}x")
        print("=" * 50)

    def vectorize_image(self, image_path: str) -> np.ndarray:
        """
        对单张图像进行多模型向量化。

        Args:
            image_path (str): 图像文件路径

        Returns:
            np.ndarray: 拼接后的高维特征向量
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件未找到: {image_path}")

        try:
            # 加载原始图像
            img = Image.open(image_path).convert('RGB')
            
            # 存储所有模型的特征向量
            feature_vectors = []
            
            # 对每个模型进行推理
            for model_name in self.model_names:
                # 使用对应模型的预处理器
                preprocessor = self.preprocessors[model_name]
                feature_extractor = self.feature_extractors[model_name]
                
                # 预处理图像
                img_tensor = preprocessor(img)
                batch_tensor = torch.unsqueeze(img_tensor, 0).to(self.device)

                # 提取特征
                with torch.no_grad():
                    features = feature_extractor(batch_tensor)
                
                # 展平为一维向量并添加到列表
                vector = features.squeeze().cpu().numpy().flatten()
                feature_vectors.append(vector)
            
            # 拼接所有特征向量
            concatenated_vector = np.concatenate(feature_vectors)
            return concatenated_vector
            
        except Exception as e:
            print(f"❌ 处理图像时出错 {image_path}: {e}")
            raise

    def vectorize_directory(self, dir_path: str, batch_size: int = 8) -> Optional[Dict]:
        """
        批量处理目录中的所有图像。

        Args:
            dir_path (str): 图像目录路径
            batch_size (int): 批处理大小

        Returns:
            dict: 包含向量和性能指标的字典
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"目录未找到: {dir_path}")

        # 查找所有图像文件
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
        image_files = []
        
        for file in os.listdir(dir_path):
            if file.lower().endswith(supported_formats):
                image_files.append(os.path.join(dir_path, file))
        
        if not image_files:
            print("❌ 目录中未找到图像文件")
            return None

        print(f"📁 找到 {len(image_files)} 张图像")
        print(f"🔧 批处理大小: {batch_size}")
        print("-" * 30)

        all_vectors = {}
        total_time = 0
        processed_count = 0
        failed_count = 0
        
        # 批量处理
        for i in range(0, len(image_files), batch_size):
            batch_paths = image_files[i:i+batch_size]
            batch_tensors = []
            valid_paths = []
            
            # 直接记录有效路径，预处理将在每个模型中进行
            for path in batch_paths:
                try:
                    # 简单验证图像文件是否可读
                    img = Image.open(path).convert('RGB')
                    valid_paths.append(path)
                except Exception as e:
                    print(f"⚠️ 跳过损坏的图像: {os.path.basename(path)} ({e})")
                    failed_count += 1
                    continue
            
            if not valid_paths:
                continue
            
            # 记录推理时间并进行多模型推理
            start_time = time.time()
            
            # 对每个模型进行批量推理
            all_model_features = {}
            for model_name in self.model_names:
                # 重新预处理当前批次（针对不同模型的输入尺寸）
                preprocessor = self.preprocessors[model_name]
                feature_extractor = self.feature_extractors[model_name]
                
                model_batch_tensors = []
                for path in valid_paths:
                    img = Image.open(path).convert('RGB')
                    img_tensor = preprocessor(img)
                    model_batch_tensors.append(img_tensor)
                
                if model_batch_tensors:
                    model_batch_tensor = torch.stack(model_batch_tensors).to(self.device)
                    
                    with torch.no_grad():
                        features = feature_extractor(model_batch_tensor)
                    
                    all_model_features[model_name] = features.squeeze().cpu().numpy()
            
            # 同步GPU操作 (如果使用GPU)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time
            
            # 拼接所有模型的特征向量
            batch_size = len(valid_paths)
            for j, path in enumerate(valid_paths):
                filename = os.path.basename(path)
                
                # 收集当前样本的所有模型特征
                sample_features = []
                for model_name in self.model_names:
                    model_features = all_model_features[model_name]
                    
                    # 处理单个样本的情况
                    if batch_size == 1:
                        if model_features.ndim == 1:
                            sample_feature = model_features
                        else:
                            sample_feature = model_features.flatten()
                    else:
                        if model_features.ndim == 1:
                            sample_feature = model_features
                        else:
                            sample_feature = model_features[j].flatten()
                    
                    sample_features.append(sample_feature)
                
                # 拼接当前样本的所有特征
                concatenated_features = np.concatenate(sample_features)
                all_vectors[filename] = concatenated_features
                processed_count += 1
            
            # 显示进度
            avg_time_per_image = batch_time/len(valid_paths) if len(valid_paths) > 0 else 0
            print(f"📊 批次 {i//batch_size + 1}: {len(valid_paths)} 张图像，"
                  f"耗时 {batch_time:.3f}秒 "
                  f"(平均 {avg_time_per_image:.3f}秒/张)")

        return {
            "vectors": all_vectors,
            "total_time": total_time,
            "processed_count": processed_count,
            "failed_count": failed_count,
            "avg_time_per_image": total_time / processed_count if processed_count > 0 else 0,
            "device": str(self.device),
            "models": self.model_names,
            "feature_dim": self.total_feature_dim,
            "num_models": len(self.model_names)
        }

    def get_device_info(self) -> Dict[str, Union[str, float]]:
        """获取设备信息"""
        info = {
            "device_type": self.device.type,
            "device_name": str(self.device)
        }
        
        if self.device.type == 'cuda':
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "gpu_memory_allocated": torch.cuda.memory_allocated(0) / 1024**3,
                "gpu_memory_reserved": torch.cuda.memory_reserved(0) / 1024**3,
            })
        
        return info

def create_test_images(output_dir: str, count: int = 10):
    """
    创建测试图像文件
    
    Args:
        output_dir (str): 输出目录
        count (int): 创建图像数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在创建 {count} 张测试图像到 {output_dir}")
    
    for i in range(count):
        # 创建随机彩色图像
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # 添加一些简单的几何形状
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # 随机绘制矩形和圆形
        for _ in range(3):
            x1, y1 = np.random.randint(0, 150, 2)
            x2, y2 = x1 + np.random.randint(20, 74), y1 + np.random.randint(20, 74)
            color = tuple(np.random.randint(0, 255, 3))
            
            if np.random.random() > 0.5:
                draw.rectangle([x1, y1, x2, y2], fill=color)
            else:
                draw.ellipse([x1, y1, x2, y2], fill=color)
        
        # 保存图像
        img.save(os.path.join(output_dir, f"test_image_{i+1:03d}.jpg"))
    
    print(f"✓ 已创建 {count} 张测试图像")

def main():
    parser = argparse.ArgumentParser(
        description="图像向量化工具 - 支持CPU/GPU性能对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单张图像 (CPU, 使用所有模型)
  python examples/image_vectorizer.py --image_path image.jpg --device cpu
  
  # 批量处理目录 (GPU, 使用所有模型)
  python examples/image_vectorizer.py --image_path ./images/ --device gpu --batch_size 16
  
  # 使用特定模型组合
  python examples/image_vectorizer.py --image_path image.jpg --models resnet152,vit_l_16
  
  # 使用单个模型（向后兼容）
  python examples/image_vectorizer.py --image_path image.jpg --model resnet152
  
  # 创建测试图像
  python examples/image_vectorizer.py --create_test_images ./test_images --count 20
  
  # 性能对比测试 (多模型集成 vs CPU/GPU)
  python examples/image_vectorizer.py --image_path ./test_images/ --device cpu --batch_size 4
  python examples/image_vectorizer.py --image_path ./test_images/ --device gpu --batch_size 4
        """
    )
    
    parser.add_argument("--image_path", type=str, 
                       help="单张图像路径或图像目录路径")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="计算设备 (例如: cpu, cuda, mps, gpu) (默认: cpu)")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="批处理大小 (默认: 8)")
    parser.add_argument("--models", type=str, default="resnet152",
                       help="要使用的模型列表，用逗号分隔 (默认: resnet152; 完整集成: resnet152,efficientnet_b7,vit_l_16)")
    parser.add_argument("--model", type=str, default=None,
                       help="单个模型名称（向后兼容，建议使用--models）")
    parser.add_argument("--create_test_images", type=str,
                       help="创建测试图像到指定目录")
    parser.add_argument("--count", type=int, default=10,
                       help="创建测试图像的数量 (默认: 10)")
    
    args = parser.parse_args()

    # 创建测试图像
    if args.create_test_images:
        create_test_images(args.create_test_images, args.count)
        return

    # 检查参数
    if not args.image_path:
        parser.print_help()
        return

    try:
        print("🚀 图像向量化工具启动")
        print("=" * 50)
        
        # 处理模型参数
        if args.model:
            # 向后兼容：使用单个模型
            model_names = [args.model]
        else:
            # 使用多模型列表
            model_names = [m.strip() for m in args.models.split(',') if m.strip()]
        
        print(f"选择的模型: {model_names}")
        
        # 初始化向量化器
        vectorizer = ImageVectorizer(device=args.device, model_names=model_names)
        
        # 显示设备信息
        device_info = vectorizer.get_device_info()
        print("\n📱 设备信息:")
        if device_info["device_type"] == "cuda":
            print(f"  GPU: {device_info['gpu_name']}")
            print(f"  总内存: {device_info['gpu_memory_total']:.1f} GB")
        else:
            print(f"  设备: CPU")

        if os.path.isdir(args.image_path):
            # 处理目录中的图像
            print(f"\n📂 批量处理模式")
            print(f"目录路径: {args.image_path}")
            
            results = vectorizer.vectorize_directory(args.image_path, args.batch_size)
            
            if results:
                print("\n" + "=" * 50)
                print("🎉 向量化完成!")
                print("-" * 30)
                print(f"设备: {results['device']}")
                print(f"模型: {results['models']} ({results['num_models']}个模型)")
                print(f"特征维度: {results['feature_dim']}")
                print(f"成功处理: {results['processed_count']} 张图像")
                if results['failed_count'] > 0:
                    print(f"失败: {results['failed_count']} 张图像")
                print(f"总耗时: {results['total_time']:.3f} 秒")
                print(f"平均每张: {results['avg_time_per_image']:.3f} 秒")
                if results['total_time'] > 0:
                    print(f"处理速度: {results['processed_count']/results['total_time']:.2f} 张/秒")
                else:
                    print(f"处理速度: N/A (无有效处理)")
                
                # 显示第一个向量示例
                if results['vectors']:
                    first_image = list(results['vectors'].keys())[0]
                    first_vector = results['vectors'][first_image]
                    print(f"\n📊 示例特征向量 ('{first_image}'):")
                    print(f"  形状: {first_vector.shape}")
                    print(f"  前10个元素: {first_vector[:10]}")
                    print(f"  统计: 均值={first_vector.mean():.4f}, "
                          f"标准差={first_vector.std():.4f}")

        elif os.path.isfile(args.image_path):
            # 处理单张图像
            print(f"\n🖼️ 单图像处理模式")
            print(f"图像路径: {args.image_path}")
            
            start_time = time.time()
            vector = vectorizer.vectorize_image(args.image_path)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            print("\n" + "=" * 50)
            print("🎉 向量化完成!")
            print("-" * 30)
            print(f"设备: {vectorizer.device}")
            print(f"模型: {vectorizer.model_names} ({len(vectorizer.model_names)}个模型)")
            print(f"处理时间: {processing_time:.4f} 秒")
            print(f"特征向量形状: {vector.shape}")
            print(f"前10个元素: {vector[:10]}")
            print(f"统计信息: 均值={vector.mean():.4f}, 标准差={vector.std():.4f}")

        else:
            print(f"❌ 错误: 路径 '{args.image_path}' 不是有效的文件或目录")

    except KeyboardInterrupt:
        print("\n⚡ 用户中断操作")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()