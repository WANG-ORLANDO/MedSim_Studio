"""伪影分类模型 — EfficientNet-B3 多标签分类器"""

import torch
import torch.nn as nn
import timm
from typing import Optional


class ArtifactClassifier(nn.Module):
    """多标签伪影分类器

    使用 timm 预训练 backbone + 自定义分类头。
    输出 8 个类别的 Sigmoid 概率，支持多标签分类。

    Args:
        num_classes: 分类类别数 (默认 8)
        backbone: timm 模型名称
        pretrained: 是否使用预训练权重
        dropout: Dropout 比率
    """

    def __init__(
        self,
        num_classes: int = 8,
        backbone: str = "efficientnet_b3",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
        )
        feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) 输入图像

        Returns:
            (B, num_classes) 每个类别的概率 [0, 1]
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return self.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """预测二值标签

        Args:
            x: (B, 3, H, W) 输入图像
            threshold: 分类阈值

        Returns:
            (B, num_classes) 二值标签
        """
        probs = self.forward(x)
        return (probs >= threshold).float()


def create_classifier(
    num_classes: int = 8,
    backbone: str = "efficientnet_b3",
    pretrained: bool = True,
    device: Optional[str] = None,
) -> ArtifactClassifier:
    """工厂函数：创建分类器实例

    Args:
        num_classes: 分类类别数
        backbone: timm 模型名称
        pretrained: 是否使用预训练权重
        device: 目标设备 (cpu / cuda)

    Returns:
        ArtifactClassifier 实例
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ArtifactClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
    )
    model = model.to(device)
    return model


def save_classifier(model: ArtifactClassifier, path: str) -> None:
    """保存模型权重"""
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": model.classifier[-1].out_features,
    }, path)


def detect_backbone_from_checkpoint(state_dict: dict) -> str:
    """根据 checkpoint 中的 tensor shape 自动检测 backbone"""
    # EfficientNet 系列 classifier 输出维度
    backbone_features = {
        "efficientnet_b0": 1280,
        "efficientnet_b1": 1280,
        "efficientnet_b2": 1408,
        "efficientnet_b3": 1536,
        "efficientnet_b4": 1792,
        "efficientnet_b5": 2048,
        "efficientnet_b6": 2304,
        "efficientnet_b7": 2560,
    }
    # 查找 conv_head 或 bn2 的 shape 来确定特征维度
    for key, tensor in state_dict.items():
        if key == "backbone.conv_head.weight":
            out_features = tensor.shape[0]
            for name, dim in backbone_features.items():
                if dim == out_features:
                    return name
            break
        if key == "backbone.bn2.weight":
            out_features = tensor.shape[0]
            for name, dim in backbone_features.items():
                if dim == out_features:
                    return name
            break
    return "efficientnet_b3"


def load_classifier(path: str, device: Optional[str] = None, backbone: Optional[str] = None) -> ArtifactClassifier:
    """加载模型权重

    Args:
        path: 模型权重文件路径
        device: 目标设备
        backbone: 模型 backbone，None 则自动检测
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    num_classes = checkpoint.get("num_classes", 8)
    if backbone is None:
        backbone = detect_backbone_from_checkpoint(checkpoint["model_state_dict"])
    model = ArtifactClassifier(num_classes=num_classes, backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model
