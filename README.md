# 基于多模态特征融合与事实核查的虚假新闻检测模型 (EAKM-Net)

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 项目简介

本项目实现了一个基于多模态特征融合与事实核查的虚假新闻检测模型（EAKM-Net）。该模型通过融合文本和图像特征，并结合外部知识库进行事实核查，能够有效识别虚假新闻。

### 主要特性

- 🔍 **多模态特征提取**：同时分析文本和图像内容
- 📚 **知识增强**：利用外部知识库进行事实核查
- 🔗 **实体对齐**：识别并验证新闻中的关键实体
- 🎯 **端到端训练**：支持完整的训练和评估流程
- 📊 **CFND数据集支持**：针对中文虚假新闻检测优化

## 🏗️ 项目结构

```
fake_news_detection/
├── train.py                      # 模型训练脚本
├── evaluate.py                   # 模型评估脚本
├── eakm_net.py                   # EAKM-Net 网络架构
├── text_feature_extractor.py     # 文本特征提取器
├── image_feature_extractor.py    # 图像特征提取器
├── knowledge_enhancement.py      # 知识增强模块
├── entity_alignment.py           # 实体对齐模块
├── cfnd_dataset.py               # CFND数据集加载器
├── data_preprocessing.py         # 数据预处理工具
├── requirements.txt              # Python依赖包
├── CFND_dataset/                 # CFND数据集目录
│   ├── train_data.csv           # 训练集
│   ├── val_data.csv             # 验证集
│   ├── test_data.csv            # 测试集
│   └── images/                  # 图像数据
│       ├── real_news_image/     # 真实新闻图像
│       └── fake_news_image/     # 虚假新闻图像
├── checkpoints/                  # 模型检查点保存目录
├── 使用说明.md                   # 详细使用说明
├── CFND使用说明.md              # CFND数据集使用说明
└── 课程报告.md                   # 项目报告
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- CUDA 10.2+ (可选，用于GPU加速)
- 至少 8GB RAM
- 建议使用 GPU 进行训练

### 安装步骤

1. **克隆仓库**
```bash
git clone <repository-url>
cd fake_news_detection
```

2. **安装Python依赖**
```bash
pip install -r requirements.txt
```

3. **安装spaCy中文模型（用于实体识别）**
```bash
python -m spacy download zh_core_web_sm
```

或者使用英文模型：
```bash
python -m spacy download en_core_web_sm
```

4. **安装中文字体（Linux系统，用于图像中文显示）**
   
   如果使用Linux系统，需要安装中文字体以确保评估结果图像中的中文正常显示：
   
   **Ubuntu/Debian:**
   ```bash
   sudo apt-get update
   sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei
   # 或者安装Noto字体
   sudo apt-get install fonts-noto-cjk
   ```
   
   **CentOS/RHEL:**
   ```bash
   sudo yum install wqy-microhei-fonts wqy-zenhei-fonts
   # 或者安装Noto字体
   sudo yum install google-noto-cjk-fonts
   ```
   
   Windows和macOS系统通常已包含中文字体，无需额外安装。

5. **验证安装**
```bash
python -c "import torch; import transformers; print('安装成功！')"
```

## 📦 数据集

本项目使用 **CFND (Chinese Fake News Detection)** 数据集，包含：

- **训练集**：用于模型训练
- **验证集**：用于超参数调优和模型选择
- **测试集**：用于最终性能评估

数据集格式：
- **总数据量**：26,665条新闻
- **真实新闻**：16,394条
- **虚假新闻**：10,271条
- **数据划分**：训练集60%，验证集20%，测试集20%
- CSV文件包含：`num`, `title`, `image`, `label` 等字段
- 图像文件存储在 `CFND_dataset/images/` 目录下
- `label`: 0 表示真实新闻，1 表示虚假新闻

详细数据集说明请参考 [CFND使用说明.md](CFND使用说明.md)

## 🎯 使用方法

### 数据预处理

在训练前，建议先进行数据预处理：

```bash
python data_preprocessing.py \
    --data_dir CFND_dataset \
    --output_dir processed_data
```

### 模型训练

**使用CFND数据集训练：**
```bash
python train.py \
    --data_path ./CFND_dataset \
    --dataset_type cfnd \
    --text_model bert-base-chinese \
    --batch_size 8 \
    --num_epochs 50 \
    --learning_rate 2e-5 \
    --model_save_dir ./checkpoints
```

**主要参数说明：**
- `--data_path`: CFND数据集目录路径（包含train_data.csv等文件）
- `--dataset_type`: 数据集类型，使用CFND数据集时设置为 `cfnd`
- `--text_model`: 文本模型，**使用CFND数据集时必须使用 `bert-base-chinese`**
- `--batch_size`: 批次大小（根据GPU内存调整，建议8-32）
- `--num_epochs`: 训练轮数
- `--learning_rate`: 学习率（建议2e-5）
- `--model_save_dir`: 模型保存目录
- `--use_entities`: 是否使用实体识别（需要安装中文spaCy模型）
- `--device`: 设备类型（cuda/cpu，默认cuda）

### 模型评估

**评估CFND测试集：**
```bash
python evaluate.py \
    --model_path ./checkpoints/best_model.pth \
    --data_path ./CFND_dataset \
    --text_model bert-base-chinese \
    --batch_size 8 \
    --output_dir ./results
```

**主要参数说明：**
- `--model_path`: 训练好的模型路径
- `--data_path`: CFND数据集目录路径（会自动识别test_data.csv）
- `--text_model`: 文本模型，必须与训练时使用的模型一致（`bert-base-chinese`）
- `--batch_size`: 批次大小
- `--output_dir`: 结果保存目录（包含评估报告和可视化图表）

评估指标包括：
- **准确率 (Accuracy)**
- **精确率 (Precision)**
- **召回率 (Recall)**
- **F1分数 (F1-Score)**
- **混淆矩阵 (Confusion Matrix)**

### 模型推理

**Python API使用示例：**

```python
import torch
from eakm_net import EAKMNet
from PIL import Image

# 加载模型
model = EAKMNet(num_classes=2)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 准备输入
text = "这是一条新闻文本..."
image = Image.open('path/to/image.jpg')

# 预测
with torch.no_grad():
    output = model(text, image)
    prediction = torch.softmax(output, dim=1)
    label = torch.argmax(prediction, dim=1).item()
    
print(f"预测结果: {'虚假新闻' if label == 1 else '真实新闻'}")
print(f"置信度: {prediction[0][label].item():.4f}")
```

## 🧠 模型架构

EAKM-Net (Entity-Aligned Knowledge-Enhanced Multimodal Network) 主要包含以下模块：

1. **文本特征提取器** (`text_feature_extractor.py`)
   - 基于预训练语言模型（BERT/RoBERTa）
   - 提取文本语义特征

2. **图像特征提取器** (`image_feature_extractor.py`)
   - 基于预训练视觉模型（ResNet/ViT）
   - 提取图像视觉特征

3. **实体对齐模块** (`entity_alignment.py`)
   - 识别文本中的关键实体
   - 对齐文本和图像中的实体信息

4. **知识增强模块** (`knowledge_enhancement.py`)
   - 查询外部知识库
   - 验证实体和事实的真实性

5. **多模态融合网络** (`eakm_net.py`)
   - 融合文本、图像和知识特征
   - 输出最终分类结果

## 📊 实验结果

模型在CFND数据集上的性能表现：

| 指标 | 数值 |
|------|------|
| 准确率 (Accuracy) | - |
| 精确率 (Precision) | - |
| 召回率 (Recall) | - |
| F1分数 (F1-Score) | - |

*注：具体数值请运行评估脚本获取*

## 🔧 常见问题

### Q1: 内存不足（Out of Memory）

**解决方案：**
- 减小 `batch_size`（如改为4或2）
- 禁用实体识别（不使用 `--use_entities` 参数）
- 使用更小的模型

```bash
python train.py --data_path ./CFND_dataset --dataset_type cfnd --batch_size 4 --no_entities
```

### Q2: 训练速度慢

**解决方案：**
- 使用GPU训练（`--device cuda`）
- 减小batch_size（如改为4或2）
- 使用预训练的BERT模型（会自动下载）

### Q3: 实体识别失败

**解决方案：**
- 安装中文spaCy模型：`python -m spacy download zh_core_web_sm`
- 或者禁用实体识别：训练时不使用 `--use_entities` 参数
- 检查语言设置（中文/英文）
- 验证文本编码格式

### Q4: 图像加载失败

**解决方案：**
- 检查图像路径是否正确
- 验证图像文件格式（支持JPG、PNG等）
- 确保图像文件未损坏

### Q5: 知识库查询失败

**解决方案：**
- 检查网络连接
- 验证知识库API配置
- 查看日志文件获取详细错误信息

### Q6: 评估结果图像中中文显示为方块

**解决方案：**
- **Linux系统**：安装中文字体
  ```bash
  # Ubuntu/Debian
  sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei
  # 或
  sudo apt-get install fonts-noto-cjk
  ```
- **验证字体安装**：运行评估脚本时，会输出使用的字体名称
- 如果仍无法显示，检查matplotlib字体缓存：
  ```bash
  python -c "import matplotlib; print(matplotlib.get_cachedir())"
  rm -rf ~/.cache/matplotlib
  ```

## 📝 依赖包

主要依赖包列表（详见 `requirements.txt`）：

- `torch>=1.9.0` - PyTorch深度学习框架
- `torchvision>=0.10.0` - 计算机视觉工具
- `transformers>=4.20.0` - Hugging Face预训练模型
- `pandas>=1.3.0` - 数据处理
- `numpy>=1.21.0` - 数值计算
- `scikit-learn>=1.0.0` - 机器学习工具
- `Pillow>=8.3.0` - 图像处理
- `opencv-python>=4.5.0` - 计算机视觉库
- `spacy>=3.4.0` - 自然语言处理
- `tqdm>=4.62.0` - 进度条显示

## 📚 相关文档

- [详细使用说明](使用说明.md) - 完整的使用指南和API文档
- [CFND数据集说明](CFND使用说明.md) - 数据集详细介绍
- [课程报告](课程报告.md) - 项目技术报告

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 👥 作者

- 项目组成员

## 🙏 致谢

- 感谢CFND数据集的提供者
- 感谢所有开源社区的支持

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件

---

**注意：** 本项目仅用于学术研究目的。使用本模型进行虚假新闻检测时，请结合人工审核，确保结果的准确性。
