"""
图像特征提取模块
使用ResNet或Vision Transformer提取图像特征
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

try:
    from transformers import ViTModel, ViTImageProcessor
    VIT_AVAILABLE = True
except ImportError:
    VIT_AVAILABLE = False


class ImageFeatureExtractor(nn.Module):
    """图像特征提取器"""
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        feature_dim: int = 2048,
        freeze_backbone: bool = False
    ):
        """
        初始化图像特征提取器
        
        Args:
            backbone: 骨干网络 ('resnet50', 'resnet101', 'vit')
            pretrained: 是否使用预训练权重
            feature_dim: 输出特征维度
            freeze_backbone: 是否冻结骨干网络
        """
        super(ImageFeatureExtractor, self).__init__()
        
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        
        # 加载模型
        if backbone.startswith('resnet'):
            self.backbone = self._load_resnet(backbone, pretrained)
        elif backbone == 'vit':
            self.backbone = self._load_vit(pretrained)
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")
        
        # 冻结参数（如果指定）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 获取特征维度
        self._get_feature_dim()
        
        # 特征投影层
        if self.backbone_feature_dim != feature_dim:
            self.projection = nn.Linear(self.backbone_feature_dim, feature_dim)
        else:
            self.projection = nn.Identity()
    
    def _load_resnet(self, model_name: str, pretrained: bool) -> nn.Module:
        """加载ResNet模型"""
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"不支持的ResNet模型: {model_name}")
        
        # 移除分类层
        model.fc = nn.Identity()
        return model
    
    def _load_vit(self, pretrained: bool) -> nn.Module:
        """加载Vision Transformer模型"""
        if not VIT_AVAILABLE:
            raise ImportError("transformers未安装，无法使用ViT")
        
        model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        return model
    
    def _get_feature_dim(self):
        """获取骨干网络的特征维度"""
        test_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            if self.backbone_name == 'vit':
                outputs = self.backbone(test_input)
                self.backbone_feature_dim = outputs.last_hidden_state.shape[-1]
            else:
                features = self.backbone(test_input)
                self.backbone_feature_dim = features.shape[1]
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        提取图像特征
        
        Args:
            images: 图像张量，形状为 (batch_size, channels, height, width)
        
        Returns:
            图像特征，形状为 (batch_size, feature_dim)
        """
        if self.backbone_name == 'vit':
            # ViT需要特殊处理
            outputs = self.backbone(images)
            # 使用[CLS] token的特征
            features = outputs.last_hidden_state[:, 0, :]
        else:
            # ResNet
            features = self.backbone(images)
        
        # 投影到目标维度
        features = self.projection(features)
        
        return features
    
    def extract_region_features(
        self, 
        images: torch.Tensor,
        regions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        提取图像区域特征（用于实体对齐）
        
        Args:
            images: 图像张量
            regions: 区域坐标，形状为 (batch_size, num_regions, 4) [x1, y1, x2, y2]
        
        Returns:
            区域特征，形状为 (batch_size, num_regions, feature_dim)
        """
        # 简化版本：使用全局特征
        # 实际应用中可以使用ROI pooling或crop+resize
        global_features = self.forward(images)  # (batch_size, feature_dim)
        
        # 复制为区域特征（简化处理）
        if regions is not None:
            num_regions = regions.shape[1]
            region_features = global_features.unsqueeze(1).repeat(1, num_regions, 1)
        else:
            # 默认使用5个区域
            num_regions = 5
            region_features = global_features.unsqueeze(1).repeat(1, num_regions, 1)
        
        return region_features


class ResNetFeatureExtractor(ImageFeatureExtractor):
    """ResNet特征提取器（便捷类）"""
    
    def __init__(self, model_name: str = 'resnet50', **kwargs):
        super().__init__(backbone=model_name, **kwargs)


class ViTFeatureExtractor(ImageFeatureExtractor):
    """Vision Transformer特征提取器（便捷类）"""
    
    def __init__(self, **kwargs):
        super().__init__(backbone='vit', **kwargs)


if __name__ == "__main__":
    # 测试代码
    print("测试图像特征提取模块...")
    
    # 创建特征提取器
    extractor = ImageFeatureExtractor(
        backbone='resnet50',
        pretrained=True,
        feature_dim=2048
    )
    extractor.eval()
    
    # 测试输入
    batch_size = 2
    test_images = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        features = extractor(test_images)
        print(f"输入图像形状: {test_images.shape}")
        print(f"输出特征形状: {features.shape}")
        
        # 测试区域特征
        region_features = extractor.extract_region_features(test_images)
        print(f"区域特征形状: {region_features.shape}")
    
    print("图像特征提取模块测试完成")





