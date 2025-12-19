"""
EAKM-Net: Entity-Aligned and Knowledge-Enhanced Multi-modal Network
基于实体对齐和知识增强的多模态虚假新闻检测网络
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import sys
import os

# 添加当前目录到路径，以便直接运行脚本
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from .text_feature_extractor import TextFeatureExtractor
    from .image_feature_extractor import ImageFeatureExtractor
    from .entity_alignment import EntityAlignmentModule
    from .knowledge_enhancement import KnowledgeEnhancementModule, KnowledgeBase
except ImportError:
    # 如果相对导入失败，使用绝对导入
    from text_feature_extractor import TextFeatureExtractor
    from image_feature_extractor import ImageFeatureExtractor
    from entity_alignment import EntityAlignmentModule
    from knowledge_enhancement import KnowledgeEnhancementModule, KnowledgeBase


class EAKMNet(nn.Module):
    """EAKM-Net模型"""
    
    def __init__(
        self,
        text_model_name: str = 'bert-base-uncased',
        image_backbone: str = 'resnet50',
        text_feature_dim: int = 768,
        image_feature_dim: int = 2048,
        alignment_dim: int = 512,
        knowledge_feature_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.5,
        knowledge_base: Optional[KnowledgeBase] = None
    ):
        """
        初始化EAKM-Net
        
        Args:
            text_model_name: 文本模型名称
            image_backbone: 图像骨干网络
            text_feature_dim: 文本特征维度
            image_feature_dim: 图像特征维度
            alignment_dim: 对齐空间维度
            knowledge_feature_dim: 知识特征维度
            num_classes: 分类类别数
            dropout: Dropout比率
            knowledge_base: 知识库实例
        """
        super(EAKMNet, self).__init__()
        
        # 文本特征提取器
        self.text_extractor = TextFeatureExtractor(
            model_name=text_model_name,
            feature_dim=text_feature_dim
        )
        
        # 图像特征提取器
        self.image_extractor = ImageFeatureExtractor(
            backbone=image_backbone,
            feature_dim=image_feature_dim
        )
        
        # 实体对齐模块
        self.entity_alignment = EntityAlignmentModule(
            text_feature_dim=text_feature_dim,
            image_feature_dim=image_feature_dim,
            alignment_dim=alignment_dim
        )
        
        # 知识增强模块
        self.knowledge_enhancement = KnowledgeEnhancementModule(
            knowledge_base=knowledge_base,
            feature_dim=knowledge_feature_dim
        )
        
        # 特征融合维度
        fusion_dim = text_feature_dim + image_feature_dim + alignment_dim + knowledge_feature_dim
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_dim // 2)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim // 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(
        self,
        texts: List[str],
        images: torch.Tensor,
        entities_list: List[List[str]],
        return_features: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            texts: 文本列表
            images: 图像张量，形状为 (batch_size, channels, height, width)
            entities_list: 每个文本的实体列表
            return_features: 是否返回中间特征
        
        Returns:
            分类logits，形状为 (batch_size, num_classes)
            如果return_features=True，还返回各模块的特征
        """
        batch_size = len(texts)
        
        # 1. 提取文本特征
        text_features = self.text_extractor(texts)  # (B, text_feature_dim)
        
        # 2. 提取图像特征
        image_features = self.image_extractor(images)  # (B, image_feature_dim)
        
        # 3. 提取文本实体特征（简化：使用文本特征）
        # 实际应用中需要从文本中提取实体特征
        batch_size = text_features.shape[0]
        
        # 获取每个样本的实体数量，使用最大实体数（至少为1）
        num_entities_list = [len(entities) if entities else 1 for entities in entities_list]
        max_num_entities = max(num_entities_list) if num_entities_list else 1
        max_num_entities = max(max_num_entities, 1)  # 至少为1
        
        # 简化处理：使用文本特征的变体作为实体特征
        # 为每个样本创建实体特征
        text_entity_features_list = []
        for i, num_ent in enumerate(num_entities_list):
            # 使用文本特征重复作为实体特征
            entity_feat = text_features[i:i+1].unsqueeze(1).repeat(1, max_num_entities, 1)  # (1, max_num_entities, feature_dim)
            text_entity_features_list.append(entity_feat)
        
        text_entity_features = torch.cat(text_entity_features_list, dim=0)  # (B, max_num_entities, feature_dim)
        
        # 4. 提取图像区域特征
        image_region_features = self.image_extractor.extract_region_features(images)
        
        # 5. 实体对齐
        alignment_scores, alignment_features = self.entity_alignment(
            text_entity_features,
            image_region_features
        )  # alignment_features: (B, alignment_dim)
        
        # 6. 知识增强
        knowledge_features = self.knowledge_enhancement(texts, entities_list)  # (B, knowledge_feature_dim)
        
        # 7. 特征融合
        fused_features = torch.cat([
            text_features,
            image_features,
            alignment_features,
            knowledge_features
        ], dim=1)  # (B, fusion_dim)
        
        # 8. 特征变换
        transformed_features = self.fusion(fused_features)  # (B, fusion_dim // 2)
        
        # 9. 分类
        logits = self.classifier(transformed_features)  # (B, num_classes)
        
        if return_features:
            return logits, {
                'text_features': text_features,
                'image_features': image_features,
                'alignment_features': alignment_features,
                'knowledge_features': knowledge_features,
                'fused_features': transformed_features
            }
        
        return logits


def create_eakm_net(
    text_model: str = 'bert-base-uncased',
    image_backbone: str = 'resnet50',
    num_classes: int = 2,
    **kwargs
) -> EAKMNet:
    """
    创建EAKM-Net模型的便捷函数
    
    Args:
        text_model: 文本模型名称
        image_backbone: 图像骨干网络
        num_classes: 分类类别数
        **kwargs: 其他模型参数
    
    Returns:
        EAKM-Net模型实例
    """
    model = EAKMNet(
        text_model_name=text_model,
        image_backbone=image_backbone,
        num_classes=num_classes,
        **kwargs
    )
    return model


if __name__ == "__main__":
    # 测试代码
    print("测试EAKM-Net模型...")
    
    # 创建模型
    model = create_eakm_net(
        text_model='bert-base-uncased',
        image_backbone='resnet50',
        num_classes=2
    )
    model.eval()
    
    # 测试输入
    batch_size = 2
    test_texts = [
        "This is a fake news article about politics.",
        "Breaking: New scientific discovery changes everything."
    ]
    test_images = torch.randn(batch_size, 3, 224, 224)
    test_entities = [
        ["politics", "news"],
        ["science", "discovery"]
    ]
    
    print(f"输入文本数量: {len(test_texts)}")
    print(f"输入图像形状: {test_images.shape}")
    
    with torch.no_grad():
        # 前向传播
        logits = model(test_texts, test_images, test_entities)
        print(f"输出logits形状: {logits.shape}")
        
        # 获取预测
        predictions = torch.softmax(logits, dim=1)
        print(f"预测概率:\n{predictions}")
        
        # 获取中间特征
        logits, features = model(test_texts, test_images, test_entities, return_features=True)
        print(f"\n中间特征:")
        for key, value in features.items():
            print(f"  {key}: {value.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    print("EAKM-Net模型测试完成")


