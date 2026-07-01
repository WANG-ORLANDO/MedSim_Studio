"""伪影分类数据集 — 支持多标签分类 + albumentations 数据增强"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

CLASS_NAMES = [
    "clean", "metal", "motion", "noise",
    "ring", "streak", "beam_hardening", "mixed",
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}


def get_train_transforms() -> "A.Compose":
    """训练集数据增强流水线"""
    if not HAS_ALBUMENTATIONS:
        return None
    return A.Compose([
        A.Rotate(limit=15, p=0.5, border_mode=0, value=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.3, border_mode=0, value=0),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
    ])


def get_val_transforms() -> "A.Compose":
    """验证/测试集变换（无增强）"""
    return None


def slice_to_windowed(
    slice_hu: np.ndarray,
    window_level: float = 40.0,
    window_width: float = 400.0,
) -> np.ndarray:
    """HU 切片 → 窗口化灰度图 (0-255, uint8)"""
    low = window_level - window_width / 2
    high = window_level + window_width / 2
    img = np.clip((slice_hu - low) / (high - low) * 255, 0, 255)
    return img.astype(np.uint8)


class ArtifactClassificationDataset(Dataset):
    """伪影分类数据集

    Args:
        images:  (N, H, W) 灰度图列表，值域 [0, 255]
        labels:  (N, num_classes) 多热编码标签
        transform: albumentations 变换流水线
        window_level: HU 窗位（仅在输入为 HU 时使用）
        window_width: HU 窗宽
    """

    def __init__(
        self,
        images: List[np.ndarray],
        labels: List[List[int]],
        transform: Optional[A.Compose] = None,
    ):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx].copy()
        label = np.array(self.labels[idx], dtype=np.float32)

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]

        # HWC → CHW，灰度复制为 3 通道
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img.astype(np.float32) / 255.0

        # ImageNet 归一化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img = torch.from_numpy(img.transpose(2, 0, 1))  # CHW
        label_tensor = torch.from_numpy(label)

        return img, label_tensor


def build_dataset_from_volume(
    volume: np.ndarray,
    label_vector: List[int],
    slice_indices: Optional[List[int]] = None,
    window_level: float = 40.0,
    window_width: float = 400.0,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """从 CT 体积构建分类数据集

    Args:
        volume: (z, y, x) CT 体积，HU 值
        label_vector: 伪影类型标签向量 (8,)
        slice_indices: 要提取的切片索引列表，None 则取中间 1/3
        window_level: 窗位
        window_width: 窗宽

    Returns:
        (images, labels) 元组
    """
    nz = volume.shape[0]
    if slice_indices is None:
        start = nz // 3
        end = 2 * nz // 3
        slice_indices = list(range(start, end))

    images = []
    labels = []
    for z in slice_indices:
        if 0 <= z < nz:
            img = slice_to_windowed(volume[z], window_level, window_width)
            images.append(img)
            labels.append(label_vector[:])

    return images, labels


def build_multi_volume_dataset(
    volumes: List[np.ndarray],
    label_vectors: List[List[int]],
    slice_indices_per_volume: Optional[List[List[int]]] = None,
    window_level: float = 40.0,
    window_width: float = 400.0,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """从多个 CT 体积批量构建数据集"""
    all_images = []
    all_labels = []
    for i, vol in enumerate(volumes):
        slices = slice_indices_per_volume[i] if slice_indices_per_volume else None
        imgs, lbls = build_dataset_from_volume(
            vol, label_vectors[i], slices, window_level, window_width,
        )
        all_images.extend(imgs)
        all_labels.extend(lbls)
    return all_images, all_labels
