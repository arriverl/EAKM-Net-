# CFND数据集使用说明

## 数据集介绍

CFND (Chinese Fake News Detection) 数据集是一个中文多模态虚假新闻检测数据集，包含：
- **总数据量**：26,665条新闻
- **真实新闻**：16,394条
- **虚假新闻**：10,271条
- **数据划分**：训练集60%，验证集20%，测试集20%
- **标签**：0=真实新闻，1=虚假新闻

## 数据集结构

```
CFND_dataset/
├── train_data.csv          # 训练集
├── val_data.csv            # 验证集
├── test_data.csv           # 测试集
├── images/
│   ├── real_news_image/    # 真实新闻图像
│   └── fake_news_image/    # 虚假新闻图像
└── README.md
```

## CSV文件格式

CSV文件包含以下列：
- `num`: 编号
- `title`: 新闻标题（文本内容）
- `image`: 图像路径（相对于CFND_dataset目录）
- `label`: 标签（0=真实，1=虚假）

## 使用方法

### 1. 训练模型

```bash
# 使用CFND数据集训练
python train.py \
    --data_path ./CFND_dataset \
    --dataset_type cfnd \
    --text_model bert-base-chinese \
    --batch_size 8 \
    --num_epochs 50 \
    --learning_rate 2e-5 \
    --model_save_dir ./checkpoints
```

### 2. 评估模型

```bash
# 评估CFND测试集
python evaluate.py \
    --model_path ./checkpoints/best_model.pth \
    --data_path ./CFND_dataset \
    --text_model bert-base-chinese \
    --batch_size 8 \
    --output_dir ./results
```

### 3. 参数说明

**重要参数**：
- `--data_path`: CFND数据集目录路径（包含train_data.csv等文件）
- `--dataset_type cfnd`: 指定使用CFND数据集
- `--text_model bert-base-chinese`: **必须使用中文BERT模型**
- `--use_entities`: 是否使用实体识别（需要安装中文spaCy模型）

### 4. 安装中文依赖

**中文BERT模型**（自动下载）：
```python
# 代码会自动下载bert-base-chinese模型
```

**中文spaCy模型**（可选，用于实体识别）：
```bash
python -m spacy download zh_core_web_sm
```

## 注意事项

1. **文本模型**：必须使用 `bert-base-chinese` 而不是 `bert-base-uncased`
2. **图像路径**：图像路径在CSV中是相对路径，代码会自动处理
3. **实体识别**：如果使用实体识别，需要安装中文spaCy模型
4. **内存需求**：CFND数据集较大，建议使用GPU训练

## 示例代码

### Python API使用

```python
from fake_news_detection.cfnd_dataset import create_cfnd_dataloaders

# 创建数据加载器
train_loader, val_loader, test_loader = create_cfnd_dataloaders(
    data_dir='./CFND_dataset',
    batch_size=8,
    use_entities=True,
    image_size=224
)

# 使用数据加载器
for texts, images, entities, labels in train_loader:
    # 训练代码
    pass
```

## 常见问题

### Q1: 实体识别失败怎么办？

**解决方案**：
1. 安装中文spaCy模型：`python -m spacy download zh_core_web_sm`
2. 或者禁用实体识别：训练时不使用 `--use_entities` 参数

### Q2: 图像加载失败？

**解决方案**：
1. 检查图像路径是否正确
2. 确保图像文件存在
3. 代码会自动处理缺失的图像（使用空白图像）

### Q3: 内存不足？

**解决方案**：
1. 减小batch_size（如改为4或2）
2. 禁用实体识别
3. 使用更小的模型

### Q4: 训练速度慢？

**解决方案**：
1. 使用GPU训练
2. 减小batch_size
3. 使用预训练的BERT模型（会自动下载）

---

**祝使用愉快！**




