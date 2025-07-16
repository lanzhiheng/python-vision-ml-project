# Contributing to ML/CV Project

感谢您对本项目的贡献！本文档提供了参与项目开发的指南。

## 开发环境设置

### 1. 克隆仓库
```bash
git clone https://github.com/username/ml-cv-project.git
cd ml-cv-project
```

### 2. 创建虚拟环境
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\\Scripts\\activate   # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 验证安装
```bash
PYTHONPATH=src python examples/simple_test.py
```

## 开发流程

### 1. 创建分支
```bash
git checkout -b feature/your-feature-name
```

### 2. 进行开发
- 遵循PEP 8编码规范
- 为新功能添加适当的文档
- 编写测试用例

### 3. 代码质量检查
```bash
# 代码格式化
black src/ gui/ utils/

# 代码风格检查
flake8 src/ gui/ utils/

# 导入排序
isort src/ gui/ utils/
```

### 4. 运行测试
```bash
PYTHONPATH=src python examples/simple_test.py
```

### 5. 提交代码
```bash
git add .
git commit -m "Add: 简要描述您的更改"
git push origin feature/your-feature-name
```

### 6. 创建Pull Request
- 描述您的更改
- 确保所有测试通过
- 等待代码审查

## 代码规范

### Python代码风格
- 使用PEP 8规范
- 函数和变量使用下划线命名法
- 类名使用驼峰命名法
- 常量使用全大写

### 文档字符串
```python
def example_function(param1: str, param2: int) -> bool:
    """
    示例函数的简要描述。
    
    Args:
        param1 (str): 参数1的描述
        param2 (int): 参数2的描述
        
    Returns:
        bool: 返回值的描述
        
    Raises:
        ValueError: 在什么情况下抛出此异常
    """
    pass
```

### 测试编写
- 为新功能编写单元测试
- 测试文件放在`tests/`目录
- 使用pytest框架
- 测试覆盖率应达到80%以上

## 贡献类型

### 🐛 Bug修复
- 在Issues中报告bug
- 提供复现步骤
- 如可能，提供修复方案

### ✨ 新功能
- 在开发前先讨论新功能
- 确保功能符合项目目标
- 提供充分的文档和测试

### 📖 文档改进
- 修正文档中的错误
- 添加使用示例
- 改进API文档

### 🔧 性能优化
- 提供性能基准测试
- 解释优化方案
- 确保不影响现有功能

## 项目结构

```
ml-cv-project/
├── src/                    # 源代码
│   ├── models/            # ML模型
│   ├── computer_vision/   # CV模块
│   └── deep_learning/     # DL模块
├── gui/                   # GUI界面
├── tests/                 # 测试代码
├── examples/              # 示例代码
├── configs/               # 配置文件
├── data/                  # 数据目录
├── utils/                 # 工具函数
└── docs/                  # 文档
```

## 发布流程

1. 更新版本号
2. 更新CHANGELOG.md
3. 创建发布标签
4. 生成发布说明
5. 发布到PyPI（如适用）

## 社区准则

- 尊重所有贡献者
- 保持友好和专业的交流
- 欢迎新手参与
- 提供建设性的反馈

## 许可证

通过贡献代码，您同意您的贡献将在MIT许可证下发布。

## 联系方式

- 项目主页: https://github.com/username/ml-cv-project
- 问题报告: https://github.com/username/ml-cv-project/issues
- 讨论区: https://github.com/username/ml-cv-project/discussions

感谢您的贡献！