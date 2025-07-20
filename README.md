# Python 机器学习和计算机视觉项目 

基于Python 3.10的机器学习项目，专注于计算机视觉和深度学习应用开发。

## 🆕 Image Vector SDK

本项目现在包含了一个完整的**图片向量化SDK**，支持使用CLIP模型将图片转换为向量表示，并存储到Milvus向量数据库中进行相似性搜索。

### 主要功能
- 🖼️ **图片向量化**: 使用CLIP模型将图片转换为高维向量
- 🗄️ **向量存储**: 集成Milvus向量数据库进行高效存储
- 🔍 **相似性搜索**: 基于向量相似度快速搜索相似图片
- ⚙️ **配置管理**: 灵活的环境变量和配置文件管理
- 📊 **批量处理**: 支持大规模图片批量处理
- 🚀 **高性能**: 支持并行处理和智能缓存

## 项目概述

本项目集成了多个核心机器学习和计算机视觉库，为快速开发和部署ML/CV应用提供完整的解决方案。

### 核心技术栈

#### 传统机器学习和计算机视觉
- **Python 3.10** - 核心编程语言
- **PyTorch** - 深度学习框架
- **OpenCV** - 计算机视觉处理库
- **PyQt5** - GUI界面开发框架
- **scikit-learn** - 机器学习算法库
- **NumPy** - 数值计算
- **Pandas** - 数据处理
- **Matplotlib** - 数据可视化

#### 图片向量化SDK新增技术
- **CLIP (OpenAI)** - 多模态图片-文本理解模型
- **Open-CLIP** - 开源CLIP模型实现
- **Milvus** - 高性能向量数据库
- **PyMilvus** - Milvus Python SDK
- **python-dotenv** - 环境变量管理

### 项目特色

- 🚀 完整的机器学习和计算机视觉开发环境
- 🎯 支持深度学习模型训练和推理
- 🖼️ 集成计算机视觉图像处理功能
- 🖥️ 提供PyQt5 GUI界面开发支持
- 📊 包含数据处理和可视化工具
- 🔧 模块化代码结构，易于扩展

## 项目结构

```
ml-cv-project/
├── src/                    # 源代码目录
│   ├── models/            # 机器学习模型
│   ├── computer_vision/   # 计算机视觉模块
│   ├── deep_learning/     # 深度学习模块
│   ├── sdk/               # Image Vector SDK主模块
│   ├── vectorization/     # 图片向量化模块
│   └── database/          # 向量数据库模块
├── gui/                   # GUI界面代码
├── data/                  # 数据目录
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后数据
│   └── models/           # 训练好的模型
├── utils/                 # 工具函数
├── tests/                 # 测试代码
├── configs/               # 配置文件
│   ├── model_config.yaml  # 模型配置  
│   └── sdk_config.yaml    # SDK配置
├── notebooks/             # Jupyter笔记本
├── examples/              # 示例代码
│   ├── quick_start.py     # 快速开始示例
│   ├── basic_usage.py     # 基本使用示例
│   └── advanced_usage.py  # 高级使用示例
├── scripts/               # 脚本文件
├── .env.example           # 环境变量示例
└── logs/                  # 日志文件
```

## Image Vector SDK 使用指南

### 快速开始

#### 1. 环境准备

首先需要启动Milvus向量数据库：

```bash
# 使用Docker启动Milvus
docker run -d --name milvus \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest
```

#### 2. 配置环境变量

复制环境变量示例文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件设置你的配置：

```bash
# Milvus 数据库配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=image_vectors

# CLIP 模型配置
CLIP_MODEL_NAME=ViT-B/32
CLIP_DEVICE=auto
```

#### 3. 快速开始示例

```bash
# 运行快速开始示例
python examples/quick_start.py
```

### 基本使用

```python
from src.sdk import ImageVectorSDK

# 1. 初始化SDK
sdk = ImageVectorSDK(auto_connect=True)

# 2. 处理单张图片
vector_id = sdk.process_and_store_image("path/to/image.jpg")

# 3. 搜索相似图片
similar_images = sdk.search_similar_images(
    query_image="path/to/query.jpg",
    top_k=10
)

# 4. 批量处理图片
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
vector_ids = sdk.process_and_store_images(image_paths)
```

### 高级用法

#### 自定义配置

```python
from src.sdk import ImageVectorSDK, ConfigManager

# 使用自定义配置
sdk = ImageVectorSDK(
    config_file="custom_config.yaml",
    env_file=".env",
    auto_connect=True
)
```

#### 批量处理带元数据

```python
# 批量处理带自定义元数据
image_paths = ["cat1.jpg", "cat2.jpg", "dog1.jpg"]
metadata_list = [
    {"category": "cat", "breed": "persian"},
    {"category": "cat", "breed": "siamese"},
    {"category": "dog", "breed": "golden_retriever"}
]

vector_ids = sdk.process_and_store_images(
    image_paths, 
    metadata_list=metadata_list
)
```

#### 高级搜索

```python
# 使用过滤器搜索
results = sdk.search_similar_images(
    query_image="query.jpg",
    top_k=10,
    threshold=0.8,  # 只返回相似度高于0.8的结果
    filters={"category": "cat"}  # 只在猫类图片中搜索
)
```

#### 数据库管理

```python
# 获取统计信息
stats = sdk.get_database_stats()
print(f"已存储向量数量: {stats['total_vectors']}")

# 重建索引
sdk.rebuild_index(index_type='HNSW', metric_type='COSINE')

# 健康检查
health = sdk.health_check()
print(f"整体状态: {health['overall']['status']}")
```

### API 参考

#### ImageVectorSDK 主要方法

| 方法 | 描述 | 参数 |
|------|------|------|
| `process_and_store_image()` | 处理并存储单张图片 | `image_path`, `metadata` |
| `process_and_store_images()` | 批量处理并存储图片 | `image_paths`, `metadata_list`, `batch_size` |
| `search_similar_images()` | 搜索相似图片 | `query_image`, `top_k`, `threshold`, `filters` |
| `get_image_vector()` | 获取图片向量（不存储） | `image_path` |
| `delete_vectors()` | 删除向量 | `ids`, `image_paths` |
| `health_check()` | 健康检查 | 无 |

### 示例文件

项目提供了多个示例文件：

- `examples/quick_start.py` - 最简单的快速开始示例
- `examples/basic_usage.py` - 基本功能演示
- `examples/advanced_usage.py` - 高级特性和性能优化

### 测试

运行测试套件：

```bash
python tests/test_sdk.py
```

## 安装和设置

### 1. 环境要求

- Python 3.10+
- pip包管理器

### 2. 安装依赖

#### 创建虚拟环境
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\\Scripts\\activate   # Windows
```

#### 安装依赖包
```bash
pip install -r requirements.txt
```

### 3. 核心依赖说明

#### 传统机器学习依赖

| 库名称 | 版本 | 用途 |
|--------|------|------|
| torch | >=2.0.0 | PyTorch深度学习框架 |
| torchvision | >=0.15.0 | PyTorch计算机视觉库 |
| opencv-contrib-python | >=4.8.0 | OpenCV计算机视觉处理 |
| pyqt5 | ==5.15.10 | GUI界面开发 |
| numpy | >=1.24.0 | 数值计算 |
| scikit-learn | >=1.3.0 | 机器学习算法 |
| pandas | >=2.0.0 | 数据处理 |
| matplotlib | >=3.7.0 | 数据可视化 |

#### Image Vector SDK 新增依赖

| 库名称 | 版本 | 用途 |
|--------|------|------|
| open-clip-torch | >=2.20.0 | 开源CLIP模型实现 |
| pymilvus | >=2.3.0 | Milvus向量数据库Python SDK |
| python-dotenv | >=1.0.0 | 环境变量管理 |
| pillow | >=10.0.0 | 图片处理库 |
| pyyaml | >=6.0 | YAML配置文件解析 |

## 快速开始

### 1. 验证安装

```bash
python scripts/verify_installation.py
```

### 2. 运行示例

#### 传统机器学习示例
```bash
# 计算机视觉示例
python examples/cv_example.py

# 深度学习示例
python examples/dl_example.py

# GUI界面示例
python gui/main.py
```

#### Image Vector SDK 示例
```bash
# 快速开始示例
python examples/quick_start.py

# 基本使用示例
python examples/basic_usage.py

# 高级用法示例
python examples/advanced_usage.py

# 运行测试
python tests/test_sdk.py
```

### 3. 训练模型

```bash
python src/models/train.py --config configs/model_config.yaml
```

## 开发指南

### 代码结构

- `src/models/` - 放置机器学习模型定义
- `src/computer_vision/` - 计算机视觉相关功能
- `src/deep_learning/` - 深度学习模型和训练代码
- `gui/` - PyQt5 GUI界面代码
- `utils/` - 通用工具函数

### 配置管理

项目使用YAML格式的配置文件，位于`configs/`目录：

- `model_config.yaml` - 模型配置
- `training_config.yaml` - 训练配置
- `gui_config.yaml` - GUI配置

### 数据管理

- `data/raw/` - 存放原始数据
- `data/processed/` - 存放预处理后的数据
- `data/models/` - 存放训练好的模型文件

## 示例应用

### 1. 图像分类
```python
from src.computer_vision.image_classifier import ImageClassifier

classifier = ImageClassifier()
result = classifier.predict("path/to/image.jpg")
print(f"预测结果: {result}")
```

### 2. 目标检测
```python
from src.computer_vision.object_detector import ObjectDetector

detector = ObjectDetector()
boxes = detector.detect("path/to/image.jpg")
```

### 3. GUI应用
```python
from gui.main import MLCVApp

app = MLCVApp()
app.run()
```

## 开发工具

### 代码格式化
```bash
black src/ gui/ utils/
```

### 代码检查
```bash
flake8 src/ gui/ utils/
```

### 运行测试
```bash
pytest tests/
```

## 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。详情请参阅 `LICENSE` 文件。

## 联系方式

- 项目主页: https://github.com/username/ml-cv-project
- 问题反馈: https://github.com/username/ml-cv-project/issues
- 文档: https://ml-cv-project.readthedocs.io/

## 更新日志

### v0.1.0 (2024-01-01)
- 初始项目结构
- 集成PyTorch、OpenCV、PyQt5等核心库
- 提供基础的机器学习和计算机视觉功能
- 添加GUI界面支持

---

## 致谢

感谢所有贡献者和开源社区的支持。本项目使用了以下优秀的开源库：

- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/)
- [scikit-learn](https://scikit-learn.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)