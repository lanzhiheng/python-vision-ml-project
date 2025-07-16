#!/usr/bin/env python3
"""
图像向量化工具 - 支持CPU/GPU性能对比

这个脚本使用预训练的ResNet-152模型对图像进行向量化，
并支持CPU和GPU切换，便于性能对比测试。

使用方法:
    python examples/image_vectorizer.py --image_path /path/to/image.jpg --device cpu
    python examples/image_vectorizer.py --image_path /path/to/images/ --device gpu --batch_size 16
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
    图像向量化器，使用深度学习模型提取图像特征向量。
    
    特点：
    - 使用ResNet-152深度模型，计算复杂度高，便于对比CPU/GPU性能
    - 支持单张图片和批量处理
    - 自动设备检测和切换
    - 详细的性能统计
    """

    def __init__(self, device: str = 'cpu', model_name: str = 'resnet152'):
        """
        初始化向量化器。

        Args:
            device (str): 使用的设备 ('cpu' 或 'gpu')
            model_name (str): 预训练模型名称
        """
        self.device_name = device
        self.model_name = model_name
        self.model = None
        self.feature_extractor = None
        self.preprocess = None
        self.device = None
        self._setup_model()

    def _setup_model(self):
        """
        加载预训练模型并设置预处理管道。
        """
        print(f"正在设置模型 '{self.model_name}' 在设备: '{self.device_name}'")
        
        # 1. 设置计算设备
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

        # 2. 加载强大的预训练模型 (ResNet-152 非常深，计算复杂)
        print("正在加载预训练模型...")
        try:
            if self.model_name == 'resnet152':
                weights = models.ResNet152_Weights.IMAGENET1K_V2
                self.model = models.resnet152(weights=weights)
                print("✓ ResNet-152 模型加载完成")
            elif self.model_name == 'resnet101':
                weights = models.ResNet101_Weights.IMAGENET1K_V2
                self.model = models.resnet101(weights=weights)
                print("✓ ResNet-101 模型加载完成")
            else:  # 默认使用ResNet-50
                weights = models.ResNet50_Weights.IMAGENET1K_V2
                self.model = models.resnet50(weights=weights)
                print("✓ ResNet-50 模型加载完成")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

        # 3. 移除最后的分类层，获取特征向量
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        # 4. 设置为评估模式并移动到指定设备
        self.feature_extractor.eval()
        print("正在将模型移动到计算设备...")
        start_time = time.time()
        self.feature_extractor.to(self.device)
        end_time = time.time()
        print(f"✓ 模型已移动到设备 (耗时: {end_time - start_time:.2f}秒)")

        # 5. 定义复杂的预处理管道
        self.preprocess = transforms.Compose([
            transforms.Resize(256),                    # 调整大小
            transforms.CenterCrop(224),               # 中心裁剪
            transforms.ToTensor(),                    # 转换为张量
            transforms.Normalize(                     # 标准化 (ImageNet统计)
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        print("✓ 预处理管道已设置")
        
        # 获取特征向量维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_output = self.feature_extractor(dummy_input)
            self.feature_dim = dummy_output.squeeze().shape[0]
        
        print(f"✓ 特征向量维度: {self.feature_dim}")
        print("=" * 50)

    def vectorize_image(self, image_path: str) -> np.ndarray:
        """
        对单张图像进行向量化。

        Args:
            image_path (str): 图像文件路径

        Returns:
            np.ndarray: 特征向量
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件未找到: {image_path}")

        try:
            # 加载并预处理图像
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.preprocess(img)
            batch_tensor = torch.unsqueeze(img_tensor, 0).to(self.device)

            # 提取特征
            with torch.no_grad():
                features = self.feature_extractor(batch_tensor)
            
            # 展平为一维向量
            vector = features.squeeze().cpu().numpy()
            return vector
            
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
            
            # 预处理当前批次
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.preprocess(img)
                    batch_tensors.append(img_tensor)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"⚠️ 跳过损坏的图像: {os.path.basename(path)} ({e})")
                    failed_count += 1
                    continue
            
            if not batch_tensors:
                continue
            
            # 创建批次张量并移动到设备
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # 记录推理时间
            start_time = time.time()
            with torch.no_grad():
                features = self.feature_extractor(batch_tensor)
            
            # 同步GPU操作 (如果使用GPU)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time
            
            # 提取特征向量
            vectors = features.squeeze().cpu().numpy()
            
            # 处理单个样本的情况
            if len(valid_paths) == 1:
                vectors = np.expand_dims(vectors, axis=0)

            # 保存结果
            for j, path in enumerate(valid_paths):
                filename = os.path.basename(path)
                all_vectors[filename] = vectors[j]
                processed_count += 1
            
            # 显示进度
            print(f"📊 批次 {i//batch_size + 1}: {len(valid_paths)} 张图像，"
                  f"耗时 {batch_time:.3f}秒 "
                  f"(平均 {batch_time/len(valid_paths):.3f}秒/张)")

        return {
            "vectors": all_vectors,
            "total_time": total_time,
            "processed_count": processed_count,
            "failed_count": failed_count,
            "avg_time_per_image": total_time / processed_count if processed_count > 0 else 0,
            "device": str(self.device),
            "model": self.model_name,
            "feature_dim": self.feature_dim
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
  # 处理单张图像 (CPU)
  python examples/image_vectorizer.py --image_path image.jpg --device cpu
  
  # 批量处理目录 (GPU)
  python examples/image_vectorizer.py --image_path ./images/ --device gpu --batch_size 16
  
  # 创建测试图像
  python examples/image_vectorizer.py --create_test_images ./test_images --count 20
  
  # 性能对比测试
  python examples/image_vectorizer.py --image_path ./test_images/ --device cpu --batch_size 8
  python examples/image_vectorizer.py --image_path ./test_images/ --device gpu --batch_size 8
        """
    )
    
    parser.add_argument("--image_path", type=str, 
                       help="单张图像路径或图像目录路径")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="计算设备 (例如: cpu, cuda, mps, gpu) (默认: cpu)")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="批处理大小 (默认: 8)")
    parser.add_argument("--model", type=str, default="resnet152",
                       choices=["resnet50", "resnet101", "resnet152"],
                       help="使用的模型 (默认: resnet152)")
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
        
        # 初始化向量化器
        vectorizer = ImageVectorizer(device=args.device, model_name=args.model)
        
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
                print(f"模型: {results['model']}")
                print(f"特征维度: {results['feature_dim']}")
                print(f"成功处理: {results['processed_count']} 张图像")
                if results['failed_count'] > 0:
                    print(f"失败: {results['failed_count']} 张图像")
                print(f"总耗时: {results['total_time']:.3f} 秒")
                print(f"平均每张: {results['avg_time_per_image']:.3f} 秒")
                print(f"处理速度: {results['processed_count']/results['total_time']:.2f} 张/秒")
                
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
            print(f"模型: {args.model}")
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