"""
数据预处理模块
包含文本清洗、图像预处理、实体识别等功能
"""

import re
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict
import torchvision.transforms as transforms

try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("警告: spaCy未安装，实体识别功能将受限")

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers未安装，BERT特征提取将不可用")


class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self, remove_html: bool = True, remove_url: bool = True):
        """
        初始化文本预处理器
        
        Args:
            remove_html: 是否移除HTML标签
            remove_url: 是否移除URL
        """
        self.remove_html = remove_html
        self.remove_url = remove_url
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
        
        Returns:
            清洗后的文本
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # 移除HTML标签
        if self.remove_html:
            text = re.sub(r'<[^>]+>', '', text)
        
        # 移除URL
        if self.remove_url:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 移除多余空格
        text = ' '.join(text.split())
        
        return text.strip()
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """批量预处理文本"""
        return [self.clean_text(text) for text in texts]


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, image_size: int = 224, normalize: bool = True):
        """
        初始化图像预处理器
        
        Args:
            image_size: 目标图像大小
            normalize: 是否归一化
        """
        self.image_size = image_size
        
        if normalize:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])
    
    def preprocess(self, image_path: str) -> Optional[torch.Tensor]:
        """
        预处理图像
        
        Args:
            image_path: 图像路径
        
        Returns:
            预处理后的图像张量，形状为 (C, H, W)
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor
        except Exception as e:
            print(f"图像预处理失败 {image_path}: {e}")
            return None
    
    def preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        """从PIL Image预处理"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.transform(image)


class EntityRecognizer:
    """实体识别器"""
    
    def __init__(self, model_name: str = 'en_core_web_sm'):
        """
        初始化实体识别器
        
        Args:
            model_name: spaCy模型名称
        """
        self.model_name = model_name
        self.nlp = None
        
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                print(f"警告: 无法加载spaCy模型 {model_name}")
                print(f"请运行: python -m spacy download {model_name}")
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        提取文本中的实体
        
        Args:
            text: 输入文本
        
        Returns:
            实体列表，每个实体包含 {'text', 'label', 'start', 'end'}
        """
        if self.nlp is None:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def extract_key_entities(self, text: str, entity_types: List[str] = None) -> List[str]:
        """
        提取关键实体（人名、地名、组织等）
        
        Args:
            text: 输入文本
            entity_types: 要提取的实体类型，如 ['PERSON', 'ORG', 'GPE']
        
        Returns:
            实体文本列表
        """
        if entity_types is None:
            entity_types = ['PERSON', 'ORG', 'GPE', 'EVENT']
        
        entities = self.extract_entities(text)
        key_entities = [
            ent['text'] for ent in entities 
            if ent['label'] in entity_types
        ]
        
        return key_entities


class FakeNewsDataset(Dataset):
    """虚假新闻数据集"""
    
    def __init__(
        self,
        data_path: str,
        text_col: str = 'text',
        image_col: str = 'image',
        label_col: str = 'label',
        max_length: int = 512,
        image_size: int = 224,
        use_entities: bool = True
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径（CSV）
            text_col: 文本列名
            image_col: 图像路径列名
            label_col: 标签列名
            max_length: 文本最大长度
            image_size: 图像大小
            use_entities: 是否提取实体
        """
        # 加载数据
        self.df = pd.read_csv(data_path)
        
        self.text_col = text_col
        self.image_col = image_col
        self.label_col = label_col
        self.max_length = max_length
        self.use_entities = use_entities
        
        # 初始化预处理器
        self.text_preprocessor = TextPreprocessor()
        self.image_preprocessor = ImagePreprocessor(image_size=image_size)
        self.entity_recognizer = EntityRecognizer() if use_entities else None
        
        # 预处理文本
        print("预处理文本...")
        self.df['cleaned_text'] = self.df[text_col].apply(self.text_preprocessor.clean_text)
        
        # 提取实体（如果启用）
        if use_entities and self.entity_recognizer:
            print("提取实体...")
            self.df['entities'] = self.df['cleaned_text'].apply(
                lambda x: self.entity_recognizer.extract_key_entities(x)
            )
    
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
        text = row['cleaned_text']
        
        # 图像
        image_path = row[self.image_col]
        if pd.isna(image_path) or not os.path.exists(image_path):
            # 如果图像不存在，创建空白图像
            image = torch.zeros(3, 224, 224)
        else:
            image = self.image_preprocessor.preprocess(image_path)
            if image is None:
                image = torch.zeros(3, 224, 224)
        
        # 实体
        entities = row.get('entities', []) if self.use_entities else []
        
        # 标签
        label = int(row[self.label_col]) if self.label_col in row else 0
        
        return {
            'text': text,
            'image': image,
            'entities': entities,
            'label': label,
            'index': idx
        }


if __name__ == "__main__":
    # 测试代码
    print("测试数据预处理模块...")
    
    # 测试文本预处理
    preprocessor = TextPreprocessor()
    test_text = "<p>This is a test http://example.com</p>"
    cleaned = preprocessor.clean_text(test_text)
    print(f"原始文本: {test_text}")
    print(f"清洗后: {cleaned}")
    
    # 测试实体识别
    if SPACY_AVAILABLE:
        recognizer = EntityRecognizer()
        test_text = "Barack Obama was the president of the United States."
        entities = recognizer.extract_key_entities(test_text)
        print(f"\n文本: {test_text}")
        print(f"提取的实体: {entities}")
    
    print("数据预处理模块测试完成")





