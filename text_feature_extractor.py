"""
文本特征提取模块
使用BERT/RoBERTa提取文本特征，并进行实体识别
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple

try:
    from transformers import (
        AutoTokenizer, 
        AutoModel,
        BertTokenizer,
        BertModel,
        RobertaTokenizer,
        RobertaModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers未安装，文本特征提取将不可用")


class TextFeatureExtractor(nn.Module):
    """文本特征提取器"""
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 512,
        feature_dim: int = 768,
        freeze_backbone: bool = False
    ):
        """
        初始化文本特征提取器
        
        Args:
            model_name: 预训练模型名称
            max_length: 最大序列长度
            feature_dim: 特征维度
            freeze_backbone: 是否冻结骨干网络
        """
        super(TextFeatureExtractor, self).__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers未安装，请运行: pip install transformers")
        
        self.model_name = model_name
        self.max_length = max_length
        self.feature_dim = feature_dim
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # 冻结参数（如果指定）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 获取实际的特征维度
        self._get_feature_dim()
        
        # 特征投影层（如果需要调整维度）
        if self.backbone_feature_dim != feature_dim:
            self.projection = nn.Linear(self.backbone_feature_dim, feature_dim)
        else:
            self.projection = nn.Identity()
    
    def _get_feature_dim(self):
        """获取骨干网络的特征维度"""
        # 创建测试输入
        test_text = "This is a test sentence."
        inputs = self.tokenizer(
            test_text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.backbone(**inputs)
            # 使用[CLS] token的特征
            self.backbone_feature_dim = outputs.last_hidden_state.shape[-1]
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        提取文本特征
        
        Args:
            texts: 文本列表
        
        Returns:
            文本特征，形状为 (batch_size, feature_dim)
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        
        # 移动到设备
        device = next(self.backbone.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 提取特征
        with torch.set_grad_enabled(self.training):
            outputs = self.backbone(**inputs)
        
        # 使用[CLS] token的特征
        features = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)
        
        # 投影到目标维度
        features = self.projection(features)
        
        return features
    
    def extract_entity_features(
        self, 
        texts: List[str], 
        entity_positions: List[List[Tuple[int, int]]]
    ) -> torch.Tensor:
        """
        提取实体特征
        
        Args:
            texts: 文本列表
            entity_positions: 每个文本中实体的位置列表 [(start, end), ...]
        
        Returns:
            实体特征，形状为 (batch_size, num_entities, feature_dim)
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True
        )
        
        device = next(self.backbone.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 提取特征
        with torch.set_grad_enabled(self.training):
            outputs = self.backbone(**{k: v for k, v in inputs.items() if k != 'offset_mapping'})
        
        # 获取token级别的特征
        token_features = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        
        # 提取实体特征（简化版本：使用实体区域的平均池化）
        batch_entity_features = []
        for i, positions in enumerate(entity_positions):
            entity_features = []
            for start, end in positions:
                # 找到对应的token位置（简化处理）
                # 实际应用中需要更精确的映射
                entity_tokens = token_features[i, start:end+1, :]
                entity_feat = entity_tokens.mean(dim=0)  # 平均池化
                entity_features.append(entity_feat)
            
            if len(entity_features) == 0:
                # 如果没有实体，使用零向量
                entity_features = [torch.zeros(self.feature_dim).to(device)]
            
            # 填充到相同长度（简化处理）
            max_entities = 10
            while len(entity_features) < max_entities:
                entity_features.append(torch.zeros(self.feature_dim).to(device))
            entity_features = entity_features[:max_entities]
            
            batch_entity_features.append(torch.stack(entity_features))
        
        entity_features_tensor = torch.stack(batch_entity_features)
        return entity_features_tensor


class BERTFeatureExtractor(TextFeatureExtractor):
    """BERT特征提取器（便捷类）"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', **kwargs):
        super().__init__(model_name=model_name, **kwargs)


class RoBERTaFeatureExtractor(TextFeatureExtractor):
    """RoBERTa特征提取器（便捷类）"""
    
    def __init__(self, model_name: str = 'roberta-base', **kwargs):
        super().__init__(model_name=model_name, **kwargs)


if __name__ == "__main__":
    # 测试代码
    if TRANSFORMERS_AVAILABLE:
        print("测试文本特征提取模块...")
        
        # 创建特征提取器
        extractor = TextFeatureExtractor(
            model_name='bert-base-uncased',
            max_length=128,
            feature_dim=768
        )
        extractor.eval()
        
        # 测试文本
        test_texts = [
            "This is a fake news article about politics.",
            "Breaking: New scientific discovery changes everything."
        ]
        
        with torch.no_grad():
            features = extractor(test_texts)
            print(f"输入文本数量: {len(test_texts)}")
            print(f"输出特征形状: {features.shape}")
        
        print("文本特征提取模块测试完成")
    else:
        print("transformers未安装，跳过测试")





