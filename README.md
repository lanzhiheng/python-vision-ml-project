# Python 机器学习和计算机视觉项目

基于Python 3.10的机器学习项目，专注于计算机视觉和深度学习应用开发。

## 项目概述

本项目集成了多个核心机器学习和计算机视觉库，为快速开发和部署ML/CV应用提供完整的解决方案。

### 核心技术栈

- **Python 3.10** - 核心编程语言
- **PyTorch** - 深度学习框架
- **OpenCV** - 计算机视觉处理库
- **PyQt5** - GUI界面开发框架
- **scikit-learn** - 机器学习算法库
- **NumPy** - 数值计算
- **Pandas** - 数据处理
- **Matplotlib** - 数据可视化

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
│   └── deep_learning/     # 深度学习模块
├── gui/                   # GUI界面代码
├── data/                  # 数据目录
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后数据
│   └── models/           # 训练好的模型
├── utils/                 # 工具函数
├── tests/                 # 测试代码
├── configs/               # 配置文件
├── notebooks/             # Jupyter笔记本
├── examples/              # 示例代码
├── scripts/               # 脚本文件
└── logs/                  # 日志文件
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

## 快速开始

### 1. 验证安装

```bash
python scripts/verify_installation.py
```

### 2. 运行示例

```bash
# 计算机视觉示例
python examples/cv_example.py

# 深度学习示例
python examples/dl_example.py

# GUI界面示例
python gui/main.py
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