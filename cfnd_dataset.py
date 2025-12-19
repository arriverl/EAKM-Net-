"""
CFND数据集加载器
适配CFND (Chinese Fake News Detection) 数据集格式
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional
from PIL import Image
import torchvision.transforms as transforms
import sys

# 添加当前目录到路径，以便直接运行脚本
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from .data_preprocessing import TextPreprocessor, ImagePreprocessor, EntityRecognizer
except ImportError:
    from data_preprocessing import TextPreprocessor, ImagePreprocessor, EntityRecognizer


class CFNDDataset(Dataset):
    """CFND数据集"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',  # 'train', 'val', 'test'
        text_col: str = 'title',
        image_col: str = 'image',
        label_col: str = 'label',
        max_length: int = 512,
        image_size: int = 224,
        use_entities: bool = True
    ):
        """
        初始化CFND数据集
        
        Args:
            data_dir: CFND数据集根目录（包含train_data.csv等文件）
            split: 数据集分割（'train', 'val', 'test'）
            text_col: 文本列名（默认'title'）
            image_col: 图像路径列名（默认'image'）
            label_col: 标签列名（默认'label'）
            max_length: 文本最大长度
            image_size: 图像大小
            use_entities: 是否提取实体
        """
        self.data_dir = data_dir
        self.split = split
        self.text_col = text_col
        self.image_col = image_col
        self.label_col = label_col
        self.max_length = max_length
        self.use_entities = use_entities
        
        # 加载CSV文件
        csv_file = os.path.join(data_dir, f'{split}_data.csv')
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"找不到数据文件: {csv_file}")
        
        self.df = pd.read_csv(csv_file)
        print(f"加载{split}数据集: {len(self.df)}条数据")
        
        # 初始化预处理器
        self.text_preprocessor = TextPreprocessor()
        self.image_preprocessor = ImagePreprocessor(image_size=image_size)
        self.entity_recognizer = EntityRecognizer(model_name='zh_core_web_sm') if use_entities else None
        
        # 预处理文本
        print("预处理文本...")
        self.df['cleaned_text'] = self.df[text_col].apply(
            lambda x: self.text_preprocessor.clean_text(str(x)) if pd.notna(x) else ""
        )
        
        # 提取实体（如果启用）
        if use_entities and self.entity_recognizer:
            print("提取实体...")
            try:
                self.df['entities'] = self.df['cleaned_text'].apply(
                    lambda x: self.entity_recognizer.extract_key_entities(x) if x else []
                )
            except Exception as e:
                print(f"实体提取警告: {e}")
                print("将使用空实体列表")
                self.df['entities'] = [[] for _ in range(len(self.df))]
        else:
            self.df['entities'] = [[] for _ in range(len(self.df))]
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取一个样本
        
        Returns:
            包含文本、图像、实体、标签的字典
        """
        row = self.df.iloc[idx]
        
        # 文本
        text = row.get('cleaned_text', '')
        if not text:
            text = str(row.get(self.text_col, ''))
        
        # 图像路径（相对于data_dir）
        image_path = row[self.image_col]
        if pd.isna(image_path):
            image_path = None
        else:
            # 构建完整路径
            if not os.path.isabs(image_path):
                image_path = os.path.join(self.data_dir, image_path)
            else:
                # 如果已经是绝对路径，检查是否存在
                if not os.path.exists(image_path):
                    # 尝试相对于data_dir
                    rel_path = os.path.relpath(image_path, self.data_dir)
                    image_path = os.path.join(self.data_dir, rel_path)
        
        # 加载图像
        if image_path and os.path.exists(image_path):
            image = self.image_preprocessor.preprocess(image_path)
            if image is None:
                image = torch.zeros(3, self.image_preprocessor.image_size, self.image_preprocessor.image_size)
        else:
            # 如果图像不存在，创建空白图像
            image = torch.zeros(3, self.image_preprocessor.image_size, self.image_preprocessor.image_size)
        
        # 实体
        entities = row.get('entities', [])
        if not isinstance(entities, list):
            entities = []
        
        # 标签（0=真实，1=虚假）
        label = int(row[self.label_col]) if pd.notna(row[self.label_col]) else 0
        
        return {
            'text': text,
            'image': image,
            'entities': entities,
            'label': label,
            'index': idx
        }


def collate_fn_cfnd(batch):
    """CFND数据集的自定义批处理函数"""
    texts = [item['text'] for item in batch]
    images = torch.stack([item['image'] for item in batch])
    entities = [item['entities'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    
    return texts, images, entities, labels


def create_cfnd_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 0,
    use_entities: bool = True,
    image_size: int = 224
):
    """
    创建CFND数据集的DataLoader
    
    Args:
        data_dir: CFND数据集根目录
        batch_size: 批次大小
        num_workers: 数据加载进程数
        use_entities: 是否提取实体
        image_size: 图像大小
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    # 创建数据集
    train_dataset = CFNDDataset(
        data_dir=data_dir,
        split='train',
        use_entities=use_entities,
        image_size=image_size
    )
    
    val_dataset = CFNDDataset(
        data_dir=data_dir,
        split='val',
        use_entities=use_entities,
        image_size=image_size
    )
    
    test_dataset = CFNDDataset(
        data_dir=data_dir,
        split='test',
        use_entities=use_entities,
        image_size=image_size
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_cfnd,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_cfnd,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_cfnd,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试代码
    print("测试CFND数据集加载器...")
    
    data_dir = "./CFND_dataset"
    if os.path.exists(data_dir):
        dataset = CFNDDataset(data_dir=data_dir, split='train', use_entities=False)
        print(f"数据集大小: {len(dataset)}")
        
        # 测试获取一个样本
        sample = dataset[0]
        print(f"\n样本信息:")
        print(f"  文本长度: {len(sample['text'])}")
        print(f"  图像形状: {sample['image'].shape}")
        print(f"  实体数量: {len(sample['entities'])}")
        print(f"  标签: {sample['label']}")
    else:
        print(f"数据集目录不存在: {data_dir}")

