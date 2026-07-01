"""伪影分类推理接口 — 单切片 / 整卷分类"""

import os
import sys
import numpy as np
import torch
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from app.artifact.classifier.dataset import (
    slice_to_windowed,
    CLASS_NAMES,
    NUM_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
)
from app.artifact.classifier.model import load_classifier, ArtifactClassifier


class ArtifactInference:
    """伪影分类推理器

    支持:
    - 单张切片分类 (predict_slice)
    - 整卷分类，返回逐层 + 聚合结果 (predict_volume)

    Args:
        model_path: 模型权重文件路径
        device: 推理设备
        threshold: 多标签分类阈值
        window_level: HU 窗位
        window_width: HU 窗宽
    """

    CLASS_NAMES = CLASS_NAMES

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        threshold: float = 0.5,
        window_level: float = 40.0,
        window_width: float = 400.0,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.threshold = threshold
        self.window_level = window_level
        self.window_width = window_width
        self.model = load_classifier(model_path, device=device)
        self.model.eval()

    def _preprocess(self, slice_hu: np.ndarray) -> torch.Tensor:
        """HU 切片 → 模型输入张量 (1, 3, 224, 224)"""
        img = slice_to_windowed(slice_hu, self.window_level, self.window_width)
        img = img.astype(np.float32) / 255.0
        img = np.stack([img] * 3, axis=-1)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict_slice(self, slice_hu: np.ndarray) -> Dict[str, Any]:
        """对单张切片进行伪影分类

        Args:
            slice_hu: (H, W) HU 值切片

        Returns:
            {
                "scores": {"clean": 0.9, "metal": 0.1, ...},
                "labels": ["clean"],
                "dominant": "clean",
            }
        """
        tensor = self._preprocess(slice_hu)
        probs = self.model(tensor).squeeze(0).cpu().numpy()

        scores = {IDX_TO_CLASS[i]: float(probs[i]) for i in range(NUM_CLASSES)}
        labels = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES) if probs[i] >= self.threshold]
        dominant = max(scores, key=scores.get)

        return {
            "scores": scores,
            "labels": labels,
            "dominant": dominant,
        }

    @torch.no_grad()
    def predict_volume(
        self,
        volume: np.ndarray,
        slice_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """对整个 CT 体积进行伪影分类

        Args:
            volume: (z, y, x) CT 体积，HU 值
            slice_indices: 要分析的切片索引，None 则分析全部

        Returns:
            {
                "overall_scores": {"clean": 0.8, "metal": 0.2, ...},
                "per_slice_scores": [{"scores": {...}, "labels": [...], "dominant": "..."}],
                "dominant_artifact": "clean",
                "slice_count": 32,
            }
        """
        if slice_indices is None:
            nz = volume.shape[0]
            # 默认取中间 2/3
            start = nz // 6
            end = nz - nz // 6
            slice_indices = list(range(start, end))

        per_slice = []
        for z in slice_indices:
            if 0 <= z < volume.shape[0]:
                result = self.predict_slice(volume[z])
                result["slice_index"] = z
                per_slice.append(result)

        if not per_slice:
            return {
                "overall_scores": {name: 0.0 for name in CLASS_NAMES},
                "per_slice_scores": [],
                "dominant_artifact": "clean",
                "slice_count": 0,
            }

        # 聚合：每类取所有切片的平均分
        overall = {}
        for name in CLASS_NAMES:
            scores = [r["scores"][name] for r in per_slice]
            overall[name] = float(np.mean(scores))

        dominant = max(overall, key=overall.get)

        return {
            "overall_scores": overall,
            "per_slice_scores": per_slice,
            "dominant_artifact": dominant,
            "slice_count": len(per_slice),
        }

    def predict_from_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """对预处理好的张量直接推理（用于 API 端已准备好的输入）"""
        self.model.eval()
        with torch.no_grad():
            probs = self.model(tensor.to(self.device)).squeeze(0).cpu().numpy()

        scores = {IDX_TO_CLASS[i]: float(probs[i]) for i in range(NUM_CLASSES)}
        labels = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES) if probs[i] >= self.threshold]
        dominant = max(scores, key=scores.get)

        return {
            "scores": scores,
            "labels": labels,
            "dominant": dominant,
        }
