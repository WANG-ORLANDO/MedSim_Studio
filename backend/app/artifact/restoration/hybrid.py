"""混合修复策略 — 根据伪影类型自动选择修复方法"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from ..restoration.traditional import TraditionalRestorer
from ..restoration.deep_learning import REDCNNRestorer
from ..evaluation.metrics import compute_all_metrics, format_quality_report


# 伪影类型 → 修复优先级 (越小越优先)
ARTIFACT_PRIORITY = {
    "metal": 0,
    "beam_hardening": 1,
    "ring": 2,
    "streak": 3,
    "motion": 4,
    "noise": 5,
    "mixed": 6,
    "clean": 7,
}

# 伪影类型 → 推荐修复方法
RESTORATION_STRATEGY = {
    "metal": "sinogram_mar",
    "noise": "redcnn",
    "motion": "median_filter",
    "ring": "sinogram_ring",
    "streak": "sinogram_ring",
    "beam_hardening": "median_filter",
    "mixed": "multi_step",
    "clean": "passthrough",
}


class HybridRestorer:
    """混合修复策略 — 自动选择修复方法

    流程:
    Step 1: 伪影分类 → 确定伪影类型
    Step 2: 根据伪影类型选择修复策略
    Step 3: 后处理: 保边滤波 + HU 值裁剪
    Step 4: 输出修复结果 + 修复质量评估
    """

    def __init__(self, redcnn_model_path: Optional[str] = None, device: Optional[str] = None):
        self.traditional = TraditionalRestorer()
        self.redcnn = None
        if redcnn_model_path:
            try:
                self.redcnn = REDCNNRestorer(model_path=redcnn_model_path, device=device)
            except Exception:
                pass

    def restore(
        self,
        volume: np.ndarray,
        artifact_types: List[str],
        artifact_mask: Optional[np.ndarray] = None,
        clean_reference: Optional[np.ndarray] = None,
        hu_range: Tuple[float, float] = (-1024.0, 3071.0),
    ) -> Dict[str, Any]:
        """执行混合修复

        Args:
            volume: 含伪影的 CT 体积 (Z, H, W) float32
            artifact_types: 伪影类型列表, e.g. ["metal", "noise"]
            artifact_mask: 伪影掩码 (可选)
            clean_reference: 干净参考 (可选，用于质量评估)
            hu_range: HU 值裁剪范围

        Returns:
            {
                "restored_volume": np.ndarray,
                "steps": List[dict],         # 每步修复记录
                "quality_metrics": dict,     # 质量指标
                "report": str,               # 可读报告
            }
        """
        steps = []
        current = volume.copy()

        # 按优先级排序
        sorted_types = sorted(artifact_types, key=lambda t: ARTIFACT_PRIORITY.get(t, 99))

        for atype in sorted_types:
            strategy = RESTORATION_STRATEGY.get(atype, "passthrough")
            before = current.copy()

            if strategy == "passthrough":
                continue
            elif strategy == "sinogram_mar" and artifact_mask is not None:
                current = self.traditional.mar_sinogram_interpolation(current, artifact_mask)
            elif strategy == "sinogram_ring":
                current = self.traditional.sinogram_ring_correction(current, threshold=0.1)
            elif strategy == "redcnn" and self.redcnn is not None:
                current = self.redcnn.restore_volume(current)
            elif strategy == "median_filter":
                current = self.traditional.median_denoise(current, kernel_size=3)
            elif strategy == "multi_step":
                current = self._multi_step_restore(current, artifact_mask)

            # 记录步骤
            psnr_delta = compute_all_metrics(before, current).get("psnr", 0) - \
                         compute_all_metrics(volume, before).get("psnr", 0)
            steps.append({
                "artifact_type": atype,
                "strategy": strategy,
                "psnr_delta": round(psnr_delta, 2),
            })

        # Step 3: 后处理
        current = self._postprocess(current, hu_range)

        # Step 4: 质量评估
        quality = compute_all_metrics(
            clean_reference if clean_reference is not None else volume,
            current,
            artifact=volume,
        )
        report = format_quality_report(quality)

        return {
            "restored_volume": current,
            "steps": steps,
            "quality_metrics": quality,
            "report": report,
        }

    def _multi_step_restore(
        self,
        volume: np.ndarray,
        artifact_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """复合伪影分步修复: 先金属后噪声"""
        current = volume.copy()
        if artifact_mask is not None:
            current = self.traditional.mar_sinogram_interpolation(current, artifact_mask)
        current = self.traditional.median_denoise(current, kernel_size=3)
        if self.redcnn is not None:
            current = self.redcnn.restore_volume(current)
        return current

    def _postprocess(
        self,
        volume: np.ndarray,
        hu_range: Tuple[float, float] = (-1024.0, 3071.0),
    ) -> np.ndarray:
        """后处理: HU 值裁剪"""
        return np.clip(volume, hu_range[0], hu_range[1]).astype(np.float32)
