#!/usr/bin/env python3
"""
多模型视觉 AI 性能基准测试工具

这个脚本可以测试多种视觉模型在不同计算设备上的性能差异：
- CPU: 利用所有CPU核心进行多线程推理
- MPS (Metal Performance Shaders): 利用 Apple Silicon 的 GPU 加速
- CUDA: 利用 NVIDIA GPU 加速

支持的模型包括：
- ConvNeXt V2-Large: 现代卷积网络
- ResNet-50: 经典残差网络  
- OpenCLIP ViT-G14: 大型视觉Transformer
- OpenAI CLIP ViT-L/14@336px: OpenAI CLIP模型

测试指标包括：
- 单张图像处理时间
- 批处理吞吐量
- 内存使用情况
- 模型加载时间
- 特征提取质量对比

支持功能：
- 自动设备检测和兼容性检查
- 动态批处理大小优化
- 混合精度推理（MPS/CUDA）
- 详细的性能分析报告
- 可视化性能对比图表
- 多模型对比分析
"""

import os
import sys
import time
import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import psutil
import warnings

# 禁用一些警告以保持输出整洁
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class VisionModelBenchmark:
    """多模型视觉 AI 性能基准测试类"""
    
    def __init__(self, test_image_dir: str = "benchmark_test_images", num_test_images: int = 50):
        """
        初始化基准测试
        
        Args:
            test_image_dir: 测试图像目录
            num_test_images: 测试图像数量
        """
        self.test_image_dir = test_image_dir
        self.num_test_images = num_test_images
        
        # 支持的模型配置
        self.model_configs = {
            'convnext_v2_large': {
                'name': 'ConvNeXt V2-Large',
                'hf_model_name': 'facebook/convnextv2-large-1k-224',
                'input_size': 224,
                'expected_feature_dim': 768,
                'model_type': 'transformers',
                'description': 'ConvNeXt V2-Large 现代卷积网络'
            },
            'resnet50': {
                'name': 'ResNet-50',
                'hf_model_name': 'microsoft/resnet-50',
                'input_size': 224,
                'expected_feature_dim': 2048,
                'model_type': 'transformers',
                'description': 'ResNet-50 经典残差网络'
            },
            # 可能后续会使用的模型
            'open_clip_vit_g14': {
                'name': 'OpenCLIP ViT-G/14',
                'model_name': 'ViT-g-14',
                'pretrained': 'laion2b_s34b_b88k',
                'input_size': 224,
                'expected_feature_dim': 1024,
                'model_type': 'open_clip',
                'description': 'OpenCLIP ViT-G/14 大型视觉Transformer'
            },
            # 目前正在使用的模型
            'openai_clip_vit_l14_336': {
                'name': 'OpenAI CLIP ViT-L/14@336px',
                'model_name': 'ViT-L/14@336px',
                'input_size': 336,
                'expected_feature_dim': 768,
                'model_type': 'clip',
                'description': 'OpenAI CLIP ViT-L/14@336px'
            }
        }
        
        # 性能统计
        self.results = {}
        self.device_info = {}
        
        # 检查依赖
        self._check_dependencies()
        
        # 检测可用设备
        self._detect_devices()
        
        # 准备测试图像
        self._prepare_test_images()
    
    def _check_dependencies(self):
        """检查必要的依赖库"""
        print("🔍 检查依赖库...")
        
        missing_deps = []
        optional_missing = []
        
        try:
            import torch
            print(f"  ✅ PyTorch: {torch.__version__}")
        except ImportError:
            missing_deps.append("torch")
        
        try:
            import transformers
            print(f"  ✅ Transformers: {transformers.__version__}")
        except ImportError:
            missing_deps.append("transformers")
        
        try:
            from PIL import Image
            print(f"  ✅ Pillow: Available")
        except ImportError:
            missing_deps.append("Pillow")
        
        try:
            import psutil
            print(f"  ✅ PSUtil: {psutil.__version__}")
        except ImportError:
            missing_deps.append("psutil")
        
        # 检查可选依赖
        try:
            import open_clip
            print(f"  ✅ OpenCLIP: {open_clip.__version__}")
        except ImportError:
            optional_missing.append("open_clip")
            print(f"  ⚠️ OpenCLIP: 未安装 (影响 OpenCLIP 模型)")
        
        try:
            import clip
            print(f"  ✅ OpenAI CLIP: Available")
        except ImportError:
            optional_missing.append("clip")
            print(f"  ⚠️ OpenAI CLIP: 未安装 (影响 OpenAI CLIP 模型)")
        
        if missing_deps:
            print(f"❌ 缺少必要依赖库: {', '.join(missing_deps)}")
            print("请运行: pip install torch transformers Pillow psutil")
            sys.exit(1)
        
        if optional_missing:
            print(f"💡 可选依赖库未安装: {', '.join(optional_missing)}")
            print("完整功能请运行: pip install open_clip_torch git+https://github.com/openai/CLIP.git")
        
        print("✅ 必要依赖库已安装")
    
    def _detect_devices(self):
        """检测可用的计算设备"""
        print("\n🖥️ 检测计算设备...")
        
        devices = []
        
        # CPU 始终可用
        cpu_info = {
            'device': 'cpu',
            'name': f'CPU ({psutil.cpu_count()} cores)',
            'memory': f'{psutil.virtual_memory().total / (1024**3):.1f} GB',
            'available': True
        }
        devices.append('cpu')
        self.device_info['cpu'] = cpu_info
        print(f"  ✅ CPU: {psutil.cpu_count()} 核心, {psutil.virtual_memory().total / (1024**3):.1f} GB RAM")
        
        # 检查 MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            mps_info = {
                'device': 'mps',
                'name': 'Apple Silicon GPU (MPS)',
                'memory': 'Unified Memory',
                'available': True
            }
            devices.append('mps')
            self.device_info['mps'] = mps_info
            print(f"  ✅ MPS: Apple Silicon GPU 可用")
        else:
            print(f"  ❌ MPS: 不可用（非 Apple Silicon 或版本不支持）")
        
        # 检查 CUDA
        if torch.cuda.is_available():
            cuda_info = {
                'device': 'cuda',
                'name': f'CUDA GPU ({torch.cuda.get_device_name()})',
                'memory': f'{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB',
                'available': True
            }
            devices.append('cuda')
            self.device_info['cuda'] = cuda_info
            print(f"  ✅ CUDA: {torch.cuda.get_device_name()}")
        else:
            print(f"  ❌ CUDA: 不可用")
        
        self.available_devices = devices
        print(f"\n📊 总共检测到 {len(devices)} 个可用设备: {', '.join(devices)}")
    
    def _prepare_test_images(self):
        """准备测试图像"""
        print(f"\n📁 准备测试图像...")
        
        if not os.path.exists(self.test_image_dir):
            print(f"创建测试图像目录: {self.test_image_dir}")
            os.makedirs(self.test_image_dir)
            self._create_synthetic_images()
        else:
            # 检查现有图像
            image_files = list(Path(self.test_image_dir).glob("*.jpg"))
            if len(image_files) < self.num_test_images:
                print(f"图像数量不足 ({len(image_files)}/{self.num_test_images})，创建更多测试图像...")
                self._create_synthetic_images()
            else:
                print(f"✅ 测试图像已准备: {len(image_files)} 张")
        
        # 获取图像路径列表
        self.image_paths = list(Path(self.test_image_dir).glob("*.jpg"))[:self.num_test_images]
        print(f"✅ 使用 {len(self.image_paths)} 张测试图像")
    
    def _create_synthetic_images(self):
        """创建合成测试图像"""
        print(f"🎨 创建 {self.num_test_images} 张合成测试图像...")
        
        for i in range(self.num_test_images):
            # 创建随机彩色图像
            img = Image.new('RGB', (512, 512))
            pixels = []
            
            for y in range(512):
                for x in range(512):
                    # 创建渐变和噪声效果
                    r = int((x / 512) * 255) ^ (i * 7)
                    g = int((y / 512) * 255) ^ (i * 11)
                    b = int(((x + y) / 1024) * 255) ^ (i * 13)
                    pixels.append((r % 256, g % 256, b % 256))
            
            img.putdata(pixels)
            img_path = os.path.join(self.test_image_dir, f"test_{i+1:04d}.jpg")
            img.save(img_path, quality=95)
        
        print(f"✅ 创建完成: {self.num_test_images} 张测试图像")
    
    def _load_model(self, model_key: str, device: str) -> Tuple[torch.nn.Module, callable]:
        """
        加载指定模型
        
        Args:
            model_key: 模型配置键名
            device: 目标设备 ('cpu', 'mps', 'cuda')
            
        Returns:
            (model, preprocessor): 模型和预处理器
        """
        if model_key not in self.model_configs:
            raise ValueError(f"不支持的模型: {model_key}")
        
        config = self.model_configs[model_key]
        print(f"📦 加载 {config['name']} 模型到 {device.upper()}...")
        
        start_time = time.time()
        
        try:
            if config['model_type'] == 'transformers':
                model, preprocessor = self._load_transformers_model(config, device)
            elif config['model_type'] == 'open_clip':
                model, preprocessor = self._load_open_clip_model(config, device)
            elif config['model_type'] == 'clip':
                model, preprocessor = self._load_openai_clip_model(config, device)
            else:
                raise ValueError(f"不支持的模型类型: {config['model_type']}")
            
            # 测试模型是否正常工作
            print(f"  正在验证模型功能...")
            input_size = config['input_size']
            test_input = torch.randn(1, 3, input_size, input_size).to(device)
            
            with torch.no_grad():
                if config['model_type'] == 'transformers':
                    test_output = model(test_input)
                    if hasattr(test_output, 'last_hidden_state'):
                        feature_dim = test_output.last_hidden_state.shape[-1]
                    elif hasattr(test_output, 'pooler_output'):
                        feature_dim = test_output.pooler_output.shape[-1]
                    else:
                        feature_dim = test_output.shape[-1]
                elif config['model_type'] in ['open_clip', 'clip']:
                    test_output = model.encode_image(test_input)
                    feature_dim = test_output.shape[-1]
                
                # 同步设备
                if device == 'mps' and hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
                elif device == 'cuda':
                    torch.cuda.synchronize()
            
            # 验证输出维度
            expected_dim = config['expected_feature_dim']
            if feature_dim != expected_dim:
                print(f"  ⚠️ 特征维度不匹配: {feature_dim} != {expected_dim}")
            else:
                print(f"  ✅ 特征维度验证通过: {feature_dim}")
            
            load_time = time.time() - start_time
            print(f"  ✅ 模型加载完成，耗时: {load_time:.2f} 秒")
            
            return model, preprocessor
            
        except Exception as e:
            print(f"  ❌ 模型加载失败: {e}")
            raise
    
    def _load_transformers_model(self, config: Dict, device: str) -> Tuple[torch.nn.Module, callable]:
        """加载 Transformers 模型"""
        from transformers import AutoModel, AutoImageProcessor
        
        print(f"  正在下载模型权重: {config['hf_model_name']}")
        model = AutoModel.from_pretrained(config['hf_model_name'])
        preprocessor = AutoImageProcessor.from_pretrained(config['hf_model_name'])
        
        # 设置为评估模式并移动到设备
        model.eval()
        model = model.to(device)
        
        return model, preprocessor
    
    def _load_open_clip_model(self, config: Dict, device: str) -> Tuple[torch.nn.Module, callable]:
        """加载 OpenCLIP 模型"""
        try:
            import open_clip
        except ImportError:
            raise ImportError("OpenCLIP 未安装，请运行: pip install open_clip_torch")
        
        print(f"  正在下载模型: {config['model_name']} with {config['pretrained']}")
        model, _, preprocessor = open_clip.create_model_and_transforms(
            config['model_name'], 
            pretrained=config['pretrained'],
            device=device
        )
        
        model.eval()
        return model, preprocessor
    
    def _load_openai_clip_model(self, config: Dict, device: str) -> Tuple[torch.nn.Module, callable]:
        """加载 OpenAI CLIP 模型"""
        try:
            import clip
        except ImportError:
            raise ImportError("OpenAI CLIP 未安装，请运行: pip install git+https://github.com/openai/CLIP.git")
        
        print(f"  正在下载模型: {config['model_name']}")
        model, preprocessor = clip.load(config['model_name'], device=device)
        
        model.eval()
        return model, preprocessor
    
    def _preprocess_images(self, image_paths: List[str], preprocessor, model_type: str, input_size: int) -> torch.Tensor:
        """
        预处理图像批次
        
        Args:
            image_paths: 图像路径列表
            preprocessor: 预处理器
            model_type: 模型类型
            input_size: 输入尺寸
            
        Returns:
            预处理后的张量
        """
        images = []
        
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"  ⚠️ 加载图像失败 {path}: {e}")
                # 创建默认图像
                images.append(Image.new('RGB', (input_size, input_size), color='red'))
        
        if model_type == 'transformers':
            # 使用 transformers 预处理器
            inputs = preprocessor(images, return_tensors="pt")
            return inputs['pixel_values']
        elif model_type in ['open_clip', 'clip']:
            # 使用 CLIP 预处理器
            if model_type == 'open_clip':
                # OpenCLIP 预处理器可以直接处理图像列表
                return torch.stack([preprocessor(img) for img in images])
            else:
                # OpenAI CLIP 预处理器
                return torch.stack([preprocessor(img) for img in images])
    
    def _extract_features(self, model: torch.nn.Module, input_tensor: torch.Tensor, 
                         device: str, model_type: str, use_mixed_precision: bool = False) -> torch.Tensor:
        """
        提取特征
        
        Args:
            model: 模型
            input_tensor: 输入张量
            device: 设备
            model_type: 模型类型
            use_mixed_precision: 是否使用混合精度
            
        Returns:
            特征向量
        """
        model.eval()
        
        with torch.no_grad():
            if use_mixed_precision and device in ['mps', 'cuda']:
                # 使用混合精度推理
                if device == 'mps':
                    # MPS 支持自动混合精度
                    with torch.autocast(device_type='mps', dtype=torch.float16):
                        features = self._forward_model(model, input_tensor, model_type)
                else:
                    # CUDA 混合精度
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        features = self._forward_model(model, input_tensor, model_type)
            else:
                # 标准精度推理
                features = self._forward_model(model, input_tensor, model_type)
            
        return features
    
    def _forward_model(self, model: torch.nn.Module, input_tensor: torch.Tensor, model_type: str) -> torch.Tensor:
        """根据模型类型执行前向传播"""
        if model_type == 'transformers':
            outputs = model(input_tensor)
            
            if hasattr(outputs, 'last_hidden_state'):
                # ConvNeXt V2 类型输出
                last_hidden_state = outputs.last_hidden_state
                # 全局平均池化: 对序列维度求平均
                features = torch.mean(last_hidden_state, dim=1)
            elif hasattr(outputs, 'pooler_output'):
                # ResNet 类型输出
                features = outputs.pooler_output
            else:
                # 直接特征输出
                features = outputs
        
        elif model_type in ['open_clip', 'clip']:
            # CLIP 模型使用 encode_image
            features = model.encode_image(input_tensor)
        
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return features

    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        available = []
        
        for model_key, config in self.model_configs.items():
            if config['model_type'] == 'transformers':
                available.append(model_key)
            elif config['model_type'] == 'open_clip':
                try:
                    import open_clip
                    available.append(model_key)
                except ImportError:
                    continue
            elif config['model_type'] == 'clip':
                try:
                    import clip
                    available.append(model_key)
                except ImportError:
                    continue
        
        return available

    def benchmark_single_image(self, model_key: str, device: str, num_runs: int = 10) -> Dict:
        """
        单张图像处理性能测试
        
        Args:
            model_key: 模型配置键名
            device: 测试设备
            num_runs: 运行次数
            
        Returns:
            性能统计结果
        """
        config = self.model_configs[model_key]
        print(f"\n🔍 单张图像处理性能测试 - {config['name']} on {device.upper()}")
        print("-" * 50)
        
        # 加载模型
        model, preprocessor = self._load_model(model_key, device)
        
        # 准备测试图像
        test_image_path = self.image_paths[0]
        
        # 预热
        print(f"  🔥 模型预热...")
        input_tensor = self._preprocess_images([test_image_path], preprocessor, 
                                             config['model_type'], config['input_size']).to(device)
        
        for _ in range(3):
            features = self._extract_features(model, input_tensor, device, config['model_type'])
            if device == 'mps' and hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
        
        # 正式测试
        print(f"  ⏱️ 开始 {num_runs} 次测试...")
        processing_times = []
        memory_usage = []
        
        for i in range(num_runs):
            # 获取进程对象（仅CPU需要）
            if device == 'cpu':
                process = psutil.Process()
            
            # 计时开始
            start_time = time.time()
            
            # 特征提取
            features = self._extract_features(model, input_tensor, device, config['model_type'])
            
            # 同步（确保计算完成）
            if device == 'mps' and hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
            
            # 计时结束
            end_time = time.time()
            
            processing_time = (end_time - start_time) * 1000  # 转换为毫秒
            processing_times.append(processing_time)
            
            # 内存使用监控（监测总内存使用量）
            if device == 'cpu':
                # CPU 内存监控：当前进程总内存使用
                current_memory = process.memory_info().rss / (1024**2)  # MB
                memory_usage.append(current_memory)
            elif device == 'mps':
                current_memory = torch.mps.current_allocated_memory() / (1024**2)  # MB
                memory_usage.append(current_memory)
            elif device == 'cuda':
                current_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                memory_usage.append(current_memory)
            
            if (i + 1) % 5 == 0:
                print(f"    进度: {i + 1}/{num_runs}")
        
        # 统计结果
        result = {
            'model_key': model_key,
            'device': device,
            'model_name': config['name'],
            'test_type': 'single_image',
            'num_runs': num_runs,
            'processing_times_ms': processing_times,
            'avg_time_ms': statistics.mean(processing_times),
            'median_time_ms': statistics.median(processing_times),
            'std_time_ms': statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
            'min_time_ms': min(processing_times),
            'max_time_ms': max(processing_times),
            'memory_usage_mb': memory_usage,
            'avg_memory_mb': statistics.mean(memory_usage) if memory_usage else 0,
            'feature_shape': features.shape,
            'feature_sample': features[0][:5].cpu().tolist() if features.numel() > 0 else []
        }
        
        # 打印结果
        print(f"  📊 结果统计:")
        print(f"    平均处理时间: {result['avg_time_ms']:.2f} ± {result['std_time_ms']:.2f} ms")
        print(f"    中位数时间: {result['median_time_ms']:.2f} ms")
        print(f"    时间范围: {result['min_time_ms']:.2f} - {result['max_time_ms']:.2f} ms")
        print(f"    平均内存使用: {result['avg_memory_mb']:.2f} MB")
        print(f"    特征形状: {result['feature_shape']}")
        
        return result
    
    def benchmark_batch_processing(self, model_key: str, device: str, batch_sizes: List[int] = None) -> Dict:
        """
        批处理性能测试
        
        Args:
            model_key: 模型配置键名
            device: 测试设备
            batch_sizes: 测试的批次大小列表
            
        Returns:
            批处理性能结果
        """
        config = self.model_configs[model_key]
        print(f"\n📦 批处理性能测试 - {config['name']} on {device.upper()}")
        print("-" * 50)
        
        if batch_sizes is None:
            # 根据设备类型选择合适的批次大小
            if device == 'cpu':
                batch_sizes = [1, 2, 4, 8, 16]
            else:  # mps or cuda
                batch_sizes = [1, 4, 8, 16, 32]
        
        # 加载模型
        model, preprocessor = self._load_model(model_key, device)
        
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"\n  🧪 测试批次大小: {batch_size}")
            
            try:
                # 准备批次数据
                test_paths = self.image_paths[:batch_size]
                input_tensor = self._preprocess_images(test_paths, preprocessor, 
                                                     config['model_type'], config['input_size']).to(device)
                
                # 预热
                for _ in range(2):
                    features = self._extract_features(model, input_tensor, device, config['model_type'])
                    if device == 'mps' and hasattr(torch.mps, 'synchronize'):
                        torch.mps.synchronize()
                
                # 性能测试
                num_runs = 5
                processing_times = []
                throughputs = []
                memory_usage = []
                
                for run in range(num_runs):
                    # 获取进程对象（仅CPU需要）
                    if device == 'cpu':
                        process = psutil.Process()
                    
                    # 计时
                    start_time = time.time()
                    features = self._extract_features(model, input_tensor, device, config['model_type'])
                    
                    # 同步
                    if device == 'mps' and hasattr(torch.mps, 'synchronize'):
                        torch.mps.synchronize()
                    elif device == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    
                    processing_time = end_time - start_time
                    throughput = batch_size / processing_time  # 图像/秒
                    
                    processing_times.append(processing_time * 1000)  # ms
                    throughputs.append(throughput)
                    
                    # 内存使用监控（监测总内存使用量）
                    if device == 'cpu':
                        # CPU 内存监控：当前进程总内存使用
                        current_memory = process.memory_info().rss / (1024**2)  # MB
                        memory_usage.append(current_memory)
                    elif device == 'mps':
                        current_memory = torch.mps.current_allocated_memory() / (1024**2)  # MB
                        memory_usage.append(current_memory)
                    elif device == 'cuda':
                        current_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                        memory_usage.append(current_memory)
                
                # 统计结果
                batch_result = {
                    'batch_size': batch_size,
                    'avg_time_ms': statistics.mean(processing_times),
                    'avg_throughput': statistics.mean(throughputs),
                    'std_throughput': statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
                    'avg_memory_mb': statistics.mean(memory_usage) if memory_usage else 0,
                    'feature_shape': features.shape
                }
                
                batch_results[batch_size] = batch_result
                
                print(f"    平均处理时间: {batch_result['avg_time_ms']:.2f} ms")
                print(f"    平均吞吐量: {batch_result['avg_throughput']:.2f} 图像/秒")
                print(f"    内存使用: {batch_result['avg_memory_mb']:.2f} MB")
                
            except Exception as e:
                print(f"    ❌ 批次大小 {batch_size} 测试失败: {e}")
                if "out of memory" in str(e).lower():
                    print(f"    内存不足，跳过更大的批次大小")
                    break
        
        return {
            'model_key': model_key,
            'device': device,
            'model_name': config['name'],
            'test_type': 'batch_processing',
            'batch_results': batch_results
        }
    
    def run_comprehensive_benchmark(self, model_keys: List[str] = None, devices: List[str] = None) -> Dict:
        """
        运行综合性能基准测试
        
        Args:
            model_keys: 要测试的模型键名列表，None 表示测试所有可用模型
            devices: 要测试的设备列表，None 表示测试所有可用设备
            
        Returns:
            完整的基准测试结果
        """
        if model_keys is None:
            model_keys = self.get_available_models()
        
        if devices is None:
            devices = self.available_devices
        
        print(f"\n🚀 多模型综合性能基准测试")
        print("=" * 60)
        print(f"测试设备: {', '.join(d.upper() for d in devices)}")
        print(f"测试模型: {', '.join([self.model_configs[m]['name'] for m in model_keys])}")
        print(f"测试图像: {len(self.image_paths)} 张")
        
        results = {
            'model_configs': {k: self.model_configs[k] for k in model_keys},
            'device_info': self.device_info,
            'test_config': {
                'num_test_images': len(self.image_paths),
                'devices_tested': devices,
                'models_tested': model_keys
            },
            'single_image_results': {},
            'batch_processing_results': {},
            'comparison_analysis': {}
        }
        
        # 单张图像测试
        print(f"\n" + "="*60)
        print(f"第一阶段: 单张图像处理性能测试")
        print(f"="*60)
        
        for model_key in model_keys:
            results['single_image_results'][model_key] = {}
            for device in devices:
                try:
                    result = self.benchmark_single_image(model_key, device, num_runs=20)
                    results['single_image_results'][model_key][device] = result
                except Exception as e:
                    print(f"❌ {model_key.upper()} on {device.upper()} 单张图像测试失败: {e}")
                    results['single_image_results'][model_key][device] = {'error': str(e)}
        
        # 批处理测试
        print(f"\n" + "="*60)
        print(f"第二阶段: 批处理性能测试")
        print(f"="*60)
        
        for model_key in model_keys:
            results['batch_processing_results'][model_key] = {}
            for device in devices:
                try:
                    result = self.benchmark_batch_processing(model_key, device)
                    results['batch_processing_results'][model_key][device] = result
                except Exception as e:
                    print(f"❌ {model_key.upper()} on {device.upper()} 批处理测试失败: {e}")
                    results['batch_processing_results'][model_key][device] = {'error': str(e)}
        
        # 对比分析
        results['comparison_analysis'] = self._generate_comparison_analysis(results)
        
        return results
    
    def _generate_comparison_analysis(self, results: Dict) -> Dict:
        """生成对比分析"""
        print(f"\n" + "="*60)
        print(f"第三阶段: 性能对比分析")
        print(f"="*60)
        
        analysis = {
            'summary': {},
            'recommendations': [],
            'performance_ratios': {},
            'device_comparison': {},
            'model_comparison': {}
        }
        
        # 获取测试配置
        devices_tested = results['test_config']['devices_tested']
        models_tested = results['test_config']['models_tested']
        single_results = results['single_image_results']
        
        # 设备性能对比分析
        print(f"\n📊 设备性能对比分析:")
        print("-" * 50)
        
        # 为每个模型分析设备性能
        for model_key in models_tested:
            model_name = results['model_configs'][model_key]['name']
            print(f"\n模型: {model_name}")
            print(f"{'设备':<8} {'平均时间(ms)':<12} {'吞吐量(fps)':<12} {'内存(MB)':<10}")
            print("-" * 45)
            
            model_device_performance = {}
            for device in devices_tested:
                if device in single_results.get(model_key, {}):
                    result = single_results[model_key][device]
                    if 'error' not in result:
                        avg_time = result['avg_time_ms']
                        throughput = 1000 / avg_time  # fps
                        memory = result['avg_memory_mb']
                        
                        model_device_performance[device] = {
                            'avg_time_ms': avg_time,
                            'throughput_fps': throughput,
                            'memory_mb': memory
                        }
                        
                        print(f"{device.upper():<8} {avg_time:<12.2f} {throughput:<12.2f} {memory:<10.2f}")
            
            # 计算相对于CPU的加速比
            if 'cpu' in model_device_performance and len(model_device_performance) > 1:
                cpu_throughput = model_device_performance['cpu']['throughput_fps']
                print(f"  设备加速比 (相对于CPU):")
                
                for device, perf in model_device_performance.items():
                    if device != 'cpu':
                        speedup = perf['throughput_fps'] / cpu_throughput
                        analysis['performance_ratios'][f'{model_key}_{device}_vs_cpu'] = speedup
                        
                        status = "🚀" if speedup > 2.0 else "📈" if speedup > 1.2 else "⚠️" if speedup > 0.8 else "🐌"
                        print(f"    {device.upper()}: {speedup:.2f}x {status}")
                        
                        # 生成设备建议
                        if speedup > 2.0:
                            analysis['recommendations'].append(
                                f"{model_name} 在 {device.upper()} 上显著优于 CPU ({speedup:.1f}x)，推荐用于生产环境"
                            )
                        elif speedup < 0.8:
                            analysis['recommendations'].append(
                                f"{model_name} 在 {device.upper()} 上性能不如 CPU，建议使用 CPU 处理"
                            )
            
            analysis['device_comparison'][model_key] = model_device_performance
        
        # 模型间性能对比
        print(f"\n🏆 模型间性能对比 (在CPU上):")
        print("-" * 50)
        print(f"{'模型':<20} {'平均时间(ms)':<12} {'吞吐量(fps)':<12} {'内存(MB)':<10}")
        print("-" * 55)
        
        cpu_model_performances = {}
        for model_key in models_tested:
            if 'cpu' in single_results.get(model_key, {}):
                result = single_results[model_key]['cpu']
                if 'error' not in result:
                    model_name = results['model_configs'][model_key]['name']
                    avg_time = result['avg_time_ms']
                    throughput = 1000 / avg_time
                    memory = result['avg_memory_mb']
                    
                    cpu_model_performances[model_key] = {
                        'name': model_name,
                        'avg_time_ms': avg_time,
                        'throughput_fps': throughput,
                        'memory_mb': memory
                    }
                    
                    print(f"{model_name:<20} {avg_time:<12.2f} {throughput:<12.2f} {memory:<10.2f}")
        
        # 找出最快的模型作为基准
        if cpu_model_performances:
            fastest_model = max(cpu_model_performances.items(), key=lambda x: x[1]['throughput_fps'])
            fastest_key, fastest_perf = fastest_model
            
            print(f"\n🥇 最快模型: {fastest_perf['name']} ({fastest_perf['throughput_fps']:.2f} fps)")
            print(f"模型性能排名:")
            
            # 按吞吐量排序
            sorted_models = sorted(cpu_model_performances.items(), 
                                 key=lambda x: x[1]['throughput_fps'], reverse=True)
            
            for i, (model_key, perf) in enumerate(sorted_models, 1):
                ratio = perf['throughput_fps'] / fastest_perf['throughput_fps']
                print(f"  {i}. {perf['name']}: {perf['throughput_fps']:.2f} fps ({ratio:.2f}x)")
                analysis['model_comparison'][model_key] = {
                    'rank': i,
                    'relative_performance': ratio,
                    **perf
                }
        
        # 批处理性能分析
        batch_results = results['batch_processing_results']
        print(f"\n📦 批处理性能最优配置:")
        print("-" * 50)
        
        for model_key in models_tested:
            model_name = results['model_configs'][model_key]['name']
            print(f"\n{model_name}:")
            
            for device in devices_tested:
                if device in batch_results.get(model_key, {}):
                    result = batch_results[model_key][device]
                    if 'error' not in result and 'batch_results' in result:
                        batch_data = result['batch_results']
                        if batch_data:
                            best_batch = max(batch_data.items(), key=lambda x: x[1]['avg_throughput'])
                            best_size, best_perf = best_batch
                            
                            print(f"  {device.upper()}: 批次大小 {best_size}, "
                                  f"吞吐量 {best_perf['avg_throughput']:.2f} 图像/秒")
                            
                            analysis['summary'][f'{model_key}_{device}_best_batch_size'] = best_size
                            analysis['summary'][f'{model_key}_{device}_best_throughput'] = best_perf['avg_throughput']
        
        # 生成总体建议
        print(f"\n💡 总体建议:")
        print("-" * 40)
        
        if not analysis['recommendations']:
            analysis['recommendations'].append("所有测试模型在不同设备上的性能表现相近，可根据具体需求选择")
        
        # 添加基于性能的通用建议
        if cpu_model_performances:
            fastest_key = max(cpu_model_performances.items(), key=lambda x: x[1]['throughput_fps'])[0]
            fastest_name = cpu_model_performances[fastest_key]['name']
            analysis['recommendations'].append(f"推荐使用 {fastest_name} 以获得最佳性能")
            
            # 内存使用建议
            lowest_memory = min(cpu_model_performances.items(), key=lambda x: x[1]['memory_mb'])
            if lowest_memory[1]['memory_mb'] > 0:
                analysis['recommendations'].append(
                    f"内存受限环境推荐使用 {lowest_memory[1]['name']} (内存使用: {lowest_memory[1]['memory_mb']:.1f} MB)"
                )
        
        for i, recommendation in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {recommendation}")
        
        return analysis
    
    def save_results(self, results: Dict, output_file: str = "vision_model_benchmark_results.json"):
        """保存结果到文件"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 结果已保存到: {output_file}")
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
    
    def print_device_status(self):
        """打印设备状态信息"""
        print(f"\n🖥️ 设备状态报告:")
        print("-" * 40)
        
        for device in self.available_devices:
            info = self.device_info[device]
            print(f"  {device.upper()}: {info['name']}")
            
            if device == 'cpu':
                # CPU 详细信息
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    print(f"    频率: {cpu_freq.current:.0f} MHz")
                print(f"    使用率: {psutil.cpu_percent(interval=1):.1f}%")
                
            elif device == 'mps':
                # MPS 状态
                if hasattr(torch.mps, 'current_allocated_memory'):
                    mps_memory = torch.mps.current_allocated_memory() / (1024**2)
                    print(f"    当前内存使用: {mps_memory:.1f} MB")
                print(f"    状态: 可用")
                
            elif device == 'cuda':
                # CUDA 详细信息
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated() / (1024**3)
                print(f"    总内存: {gpu_memory:.1f} GB")
                print(f"    已分配: {allocated:.1f} GB")
                print(f"    利用率: {allocated/gpu_memory*100:.1f}%")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多模型视觉 AI 性能基准测试')
    parser.add_argument('--test_dir', type=str, default='benchmark_test_images', 
                        help='测试图像目录')
    parser.add_argument('--num_images', type=int, default=50, 
                        help='测试图像数量')
    parser.add_argument('--models', type=str, nargs='+', required=False,
                        help='要测试的模型列表 (必需参数) - 可选: convnext_v2_large, resnet50, open_clip_vit_g14, openai_clip_vit_l14_336')
    parser.add_argument('--devices', type=str, nargs='+',
                        help='要测试的设备列表 (cpu, mps, cuda)')
    parser.add_argument('--output', type=str, default='vision_model_benchmark_results.json',
                        help='结果输出文件')
    parser.add_argument('--single_only', action='store_true',
                        help='仅运行单张图像测试')
    parser.add_argument('--batch_only', action='store_true',
                        help='仅运行批处理测试')
    parser.add_argument('--list_models', action='store_true',
                        help='列出所有可用模型并退出')
    
    args = parser.parse_args()
    
    print("多模型视觉 AI 性能基准测试")
    print("=" * 60)
    
    # 初始化基准测试
    benchmark = VisionModelBenchmark(
        test_image_dir=args.test_dir,
        num_test_images=args.num_images
    )
    
    # 如果只是列出模型，则显示并退出
    if args.list_models:
        print("\n📋 可用模型列表:")
        print("=" * 50)
        available_models = benchmark.get_available_models()
        
        for model_key in benchmark.model_configs:
            config = benchmark.model_configs[model_key]
            status = "✅" if model_key in available_models else "❌"
            print(f"{status} {model_key}:")
            print(f"   名称: {config['name']}")
            print(f"   描述: {config['description']}")
            print(f"   输入尺寸: {config['input_size']}")
            print(f"   特征维度: {config['expected_feature_dim']}")
            if model_key not in available_models:
                if config['model_type'] == 'open_clip':
                    print(f"   ⚠️ 需要安装: pip install open_clip_torch")
                elif config['model_type'] == 'clip':
                    print(f"   ⚠️ 需要安装: pip install git+https://github.com/openai/CLIP.git")
            print()
        
        return
    
    # 打印设备状态
    benchmark.print_device_status()
    
    # 确定要测试的模型
    if not args.models:
        print(f"❌ 请指定要测试的模型！")
        print(f"使用 --models 参数指定一个或多个模型")
        print(f"可用模型: {', '.join(benchmark.get_available_models())}")
        print(f"\n示例:")
        print(f"  python {sys.argv[0]} --models convnext_v2_large")
        print(f"  python {sys.argv[0]} --models convnext_v2_large resnet50")
        print(f"  python {sys.argv[0]} --list_models  # 查看所有可用模型")
        sys.exit(1)
    
    test_models = [m for m in args.models if m in benchmark.get_available_models()]
    if not test_models:
        print(f"❌ 指定的模型都不可用: {args.models}")
        print(f"可用模型: {', '.join(benchmark.get_available_models())}")
        sys.exit(1)
    
    # 确定要测试的设备
    if args.devices:
        test_devices = [d for d in args.devices if d in benchmark.available_devices]
        if not test_devices:
            print(f"❌ 指定的设备都不可用: {args.devices}")
            print(f"可用设备: {', '.join(benchmark.available_devices)}")
            sys.exit(1)
    else:
        test_devices = benchmark.available_devices
    
    print(f"\n🎯 将测试以下模型: {', '.join([benchmark.model_configs[m]['name'] for m in test_models])}")
    print(f"🖥️ 将测试以下设备: {', '.join(d.upper() for d in test_devices)}")
    
    # 运行基准测试
    if args.single_only:
        # 仅单张图像测试
        print(f"\n🔍 运行单张图像测试...")
        results = {
            'model_configs': {k: benchmark.model_configs[k] for k in test_models},
            'device_info': benchmark.device_info,
            'single_image_results': {}
        }
        
        for model_key in test_models:
            results['single_image_results'][model_key] = {}
            for device in test_devices:
                try:
                    result = benchmark.benchmark_single_image(model_key, device, num_runs=20)
                    results['single_image_results'][model_key][device] = result
                except Exception as e:
                    print(f"❌ {model_key.upper()} on {device.upper()} 测试失败: {e}")
                    results['single_image_results'][model_key][device] = {'error': str(e)}
                
    elif args.batch_only:
        # 仅批处理测试
        print(f"\n📦 运行批处理测试...")
        results = {
            'model_configs': {k: benchmark.model_configs[k] for k in test_models},
            'device_info': benchmark.device_info,
            'batch_processing_results': {}
        }
        
        for model_key in test_models:
            results['batch_processing_results'][model_key] = {}
            for device in test_devices:
                try:
                    result = benchmark.benchmark_batch_processing(model_key, device)
                    results['batch_processing_results'][model_key][device] = result
                except Exception as e:
                    print(f"❌ {model_key.upper()} on {device.upper()} 测试失败: {e}")
                    results['batch_processing_results'][model_key][device] = {'error': str(e)}
    else:
        # 综合测试
        results = benchmark.run_comprehensive_benchmark(test_models, test_devices)
    
    # 保存结果
    benchmark.save_results(results, args.output)
    
    print(f"\n🎉 基准测试完成!")
    print(f"详细结果已保存到: {args.output}")


if __name__ == "__main__":
    main() 