"""射束硬化伪影生成器 — 模拟多色 X 射线束硬化引起的杯状/暗带效应（增强版）"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from .base import BaseArtifactGenerator
from typing import Dict, Any, Tuple


class BeamHardeningGenerator(BaseArtifactGenerator):
    """射束硬化伪影生成器 — 杯状伪影 + 暗带效应

    增强特性:
    - HU依赖硬化：高密度物体（骨骼）引起更强的杯状效应
    - 多级暗带：骨骼之间的暗带有梯度过渡
    - 杯状多项式模型：使用二次+四次项更真实地模拟硬化曲线
    - 散射贡献：低频背景变化模拟散射影响
    """

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "cupping_strength": 0.5,
            "dark_band_strength": 0.4,
            "dark_band_positions": None,
            "density_threshold": 200.0,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        for k in ("cupping_strength", "dark_band_strength"):
            if k in params and not (0 <= params[k] <= 1):
                return False
        return True

    def generate(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        params: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        p = {**self.get_default_params(), **params}
        cupping = p["cupping_strength"]
        dark_band = p["dark_band_strength"]
        threshold = p["density_threshold"]
        nz, ny, nx = volume.shape

        result = volume.copy().astype(np.float32)

        for z in range(nz):
            slice_2d = result[z]

            # 非空气区域
            object_mask = slice_2d > (-1024 + 100)
            if not np.any(object_mask):
                continue

            dist_to_edge = distance_transform_edt(object_mask)
            max_dist = np.max(dist_to_edge)
            if max_dist > 0:
                norm_dist = dist_to_edge / max_dist
            else:
                norm_dist = np.zeros_like(dist_to_edge)

            # === 杯状效应：二次+四次多项式模型 ===
            # 低强度时线性，高强度时加速（更真实的硬化曲线）
            hu_reduction = cupping * (
                120.0 * (norm_dist ** 2) + 80.0 * (norm_dist ** 4)
            )
            slice_2d[object_mask] -= hu_reduction[object_mask]

            # === HU依赖硬化：高密度区域（骨骼）引起更强效应 ===
            bone_mask = slice_2d > threshold
            if np.any(bone_mask):
                bone_dist = distance_transform_edt(~bone_mask).astype(np.float32)
                bone_bh_radius = 20
                bone_bh_mask = (bone_dist > 0) & (bone_dist <= bone_bh_radius)
                if np.any(bone_bh_mask):
                    bone_weight = (1 - bone_dist[bone_bh_mask] / bone_bh_radius) ** 1.5
                    bone_reduction = bone_weight * dark_band * 180.0
                    slice_2d[bone_bh_mask] -= bone_reduction.astype(np.float32)

            # === 多级暗带效应：高密度区域之间的梯度暗带 ===
            high_density = slice_2d > threshold
            if np.any(high_density):
                hd_dist = distance_transform_edt(~high_density).astype(np.float32)
                # 多级暗带：近处强，远处弱
                band_radius = 18
                band_mask = (hd_dist > 0) & (hd_dist <= band_radius)

                if np.any(band_mask):
                    # 非线性衰减（近骨骼端更强）
                    band_weight = ((1 - hd_dist[band_mask] / band_radius) ** 1.3) * dark_band
                    band_reduction = band_weight * 120.0
                    slice_2d[band_mask] -= band_reduction.astype(np.float32)

            # === 散射贡献：低频背景变化 ===
            scatter = gaussian_filter(slice_2d, sigma=max(ny, nx) * 0.15)
            scatter_offset = (scatter - slice_2d) * 0.02 * cupping
            slice_2d += scatter_offset.astype(np.float32)

            result[z] = slice_2d

        # 全局轻微平滑使效果更自然
        result = gaussian_filter(result, sigma=0.3)
        result = np.clip(result, -1024, 3071).astype(np.float32)

        # 掩码
        artifact_mask = np.zeros((nz, ny, nx), dtype=np.float32)
        diff = np.abs(result - volume)
        artifact_mask[diff > 1] = 1.0

        metadata = {
            "artifact_type": self.get_artifact_type(),
            "params": p,
            "cupping_strength": cupping,
            "dark_band_strength": dark_band,
        }

        return result, artifact_mask, metadata
