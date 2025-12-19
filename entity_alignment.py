"""
实体对齐模块
计算文本实体特征与图像区域特征之间的对齐分数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class EntityAlignmentModule(nn.Module):
    """实体对齐模块"""
    
    def __init__(
        self,
        text_feature_dim: int = 768,
        image_feature_dim: int = 2048,
        alignment_dim: int = 512,
        num_regions: int = 5
    ):
        """
        初始化实体对齐模块
        
        Args:
            text_feature_dim: 文本特征维度
            image_feature_dim: 图像特征维度
            alignment_dim: 对齐空间维度
            num_regions: 图像区域数量
        """
        super(EntityAlignmentModule, self).__init__()
        
        self.text_feature_dim = text_feature_dim
        self.image_feature_dim = image_feature_dim
        self.alignment_dim = alignment_dim
        self.num_regions = num_regions
        
        # 文本实体投影层
        self.text_projection = nn.Sequential(
            nn.Linear(text_feature_dim, alignment_dim),
            nn.ReLU(),
            nn.LayerNorm(alignment_dim)
        )
        
        # 图像区域投影层
        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_dim, alignment_dim),
            nn.ReLU(),
            nn.LayerNorm(alignment_dim)
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=alignment_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 对齐分数计算
        self.alignment_scorer = nn.Sequential(
            nn.Linear(alignment_dim * 2, alignment_dim),
            nn.ReLU(),
            nn.Linear(alignment_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        text_entity_features: torch.Tensor,
        image_region_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算实体对齐分数
        
        Args:
            text_entity_features: 文本实体特征，形状为 (batch_size, num_entities, text_feature_dim)
            image_region_features: 图像区域特征，形状为 (batch_size, num_regions, image_feature_dim)
        
        Returns:
            alignment_scores: 对齐分数，形状为 (batch_size, num_entities, num_regions)
            aggregated_alignment: 聚合后的对齐特征，形状为 (batch_size, alignment_dim)
        """
        batch_size = text_entity_features.shape[0]
        num_entities = text_entity_features.shape[1]
        num_regions = image_region_features.shape[1]
        
        # 投影到对齐空间
        text_aligned = self.text_projection(text_entity_features)  # (B, num_entities, alignment_dim)
        image_aligned = self.image_projection(image_region_features)  # (B, num_regions, alignment_dim)
        
        # 计算注意力
        # 使用图像特征作为query，文本实体特征作为key和value
        aligned_features, attention_weights = self.attention(
            query=image_aligned,
            key=text_aligned,
            value=text_aligned
        )  # (B, num_regions, alignment_dim)
        
        # 计算对齐分数矩阵
        # 扩展维度以计算所有实体-区域对
        # text_aligned: (B, num_entities, alignment_dim)
        # image_aligned: (B, num_regions, alignment_dim)
        text_expanded = text_aligned.unsqueeze(2)  # (B, num_entities, 1, alignment_dim)
        image_expanded = image_aligned.unsqueeze(1)  # (B, 1, num_regions, alignment_dim)
        
        # 使用广播机制扩展维度
        # text_expanded: (B, num_entities, 1, alignment_dim) -> (B, num_entities, num_regions, alignment_dim)
        # image_expanded: (B, 1, num_regions, alignment_dim) -> (B, num_entities, num_regions, alignment_dim)
        text_expanded = text_expanded.expand(-1, -1, num_regions, -1)  # (B, num_entities, num_regions, alignment_dim)
        image_expanded = image_expanded.expand(-1, num_entities, -1, -1)  # (B, num_entities, num_regions, alignment_dim)
        
        # 拼接特征
        combined = torch.cat([text_expanded, image_expanded], dim=-1)  # (B, num_entities, num_regions, 2*alignment_dim)
        
        # 计算对齐分数
        alignment_scores = self.alignment_scorer(combined).squeeze(-1)  # (B, num_entities, num_regions)
        
        # 聚合对齐特征
        # 使用对齐分数加权聚合图像特征
        alignment_weights = F.softmax(alignment_scores.view(batch_size, -1), dim=1)  # (B, num_entities*num_regions)
        alignment_weights = alignment_weights.view(batch_size, num_entities, num_regions)
        
        # 加权平均
        weighted_image = (image_aligned.unsqueeze(1) * alignment_weights.unsqueeze(-1)).sum(dim=2)  # (B, num_entities, alignment_dim)
        aggregated_alignment = weighted_image.mean(dim=1)  # (B, alignment_dim)
        
        return alignment_scores, aggregated_alignment
    
    def compute_similarity(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算文本和图像特征的余弦相似度（简化版本）
        
        Args:
            text_features: 文本特征，形状为 (batch_size, text_feature_dim)
            image_features: 图像特征，形状为 (batch_size, image_feature_dim)
        
        Returns:
            相似度分数，形状为 (batch_size,)
        """
        # 投影到对齐空间
        text_aligned = self.text_projection(text_features.unsqueeze(1))  # (B, 1, alignment_dim)
        image_aligned = self.image_projection(image_features.unsqueeze(1))  # (B, 1, alignment_dim)
        
        # 计算余弦相似度
        text_norm = F.normalize(text_aligned.squeeze(1), p=2, dim=1)
        image_norm = F.normalize(image_aligned.squeeze(1), p=2, dim=1)
        similarity = (text_norm * image_norm).sum(dim=1)
        
        return similarity


class CrossModalAttention(nn.Module):
    """跨模态注意力机制（用于实体对齐）"""
    
    def __init__(self, feature_dim: int, num_heads: int = 8):
        """
        初始化跨模态注意力
        
        Args:
            feature_dim: 特征维度
            num_heads: 注意力头数
        """
        super(CrossModalAttention, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(
        self,
        query_features: torch.Tensor,
        key_features: torch.Tensor,
        value_features: torch.Tensor
    ) -> torch.Tensor:
        """
        跨模态注意力计算
        
        Args:
            query_features: Query特征
            key_features: Key特征
            value_features: Value特征
        
        Returns:
            注意力输出
        """
        attn_output, _ = self.attention(
            query_features,
            key_features,
            value_features
        )
        
        # 残差连接和层归一化
        output = self.norm(attn_output + query_features)
        
        return output


if __name__ == "__main__":
    # 测试代码
    print("测试实体对齐模块...")
    
    # 创建对齐模块
    alignment_module = EntityAlignmentModule(
        text_feature_dim=768,
        image_feature_dim=2048,
        alignment_dim=512,
        num_regions=5
    )
    alignment_module.eval()
    
    # 测试输入
    batch_size = 2
    num_entities = 3
    
    text_entity_features = torch.randn(batch_size, num_entities, 768)
    image_region_features = torch.randn(batch_size, 5, 2048)
    
    with torch.no_grad():
        alignment_scores, aggregated = alignment_module(
            text_entity_features,
            image_region_features
        )
        
        print(f"文本实体特征形状: {text_entity_features.shape}")
        print(f"图像区域特征形状: {image_region_features.shape}")
        print(f"对齐分数形状: {alignment_scores.shape}")
        print(f"聚合对齐特征形状: {aggregated.shape}")
    
    print("实体对齐模块测试完成")


