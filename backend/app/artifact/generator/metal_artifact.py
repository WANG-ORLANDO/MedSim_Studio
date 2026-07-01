"""金属伪影生成器 — 模拟高密度金属物体引起的射束硬化+条纹伪影（增强版）"""

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from .base import BaseArtifactGenerator
from typing import Dict, Any, Tuple


class MetalArtifactGenerator(BaseArtifactGenerator):
    """金属伪影生成器 — 模拟高密度金属物体引起的射束硬化+条纹伪影

    增强特性:
    - 亮环效应：金属边缘产生高HU亮环
    - 光子饥饿噪声：金属阴影区域噪声显著增大
    - 变化条纹：暗带和亮带交替，强度沿径向衰减
    - 射束硬化：EDT距离场驱动的杯状效应 + 骨骼间暗带
    """

    METAL_HU = {
        "titanium": 2500.0,
        "stainless_steel": 3071.0,
        "dental_amalgam": 3071.0,
        "gold": 3071.0,
        "copper": 2800.0,
    }

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "metal_type": "titanium",
            "metal_hu": 2500.0,
            "center": [0.5, 0.5, 0.5],
            "radius_mm": [5.0, 5.0, 5.0],
            "streak_intensity": 0.7,
            "beam_hardening_strength": 0.5,
            "photon_starvation_noise": 0.3,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        required = ["metal_type", "center", "radius_mm"]
        return all(k in params for k in required)

    def generate(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        params: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        p = {**self.get_default_params(), **params}
        if p["metal_type"] in self.METAL_HU:
            p["metal_hu"] = self.METAL_HU[p["metal_type"]]
        nz, ny, nx = volume.shape

        metal_hu = p["metal_hu"]
        center_voxel = (
            int(p["center"][0] * nz),
            int(p["center"][1] * ny),
            int(p["center"][2] * nx),
        )
        radius_voxel = (
            p["radius_mm"][0] / spacing[0],
            p["radius_mm"][1] / spacing[1],
            p["radius_mm"][2] / spacing[2],
        )

        z_idx, y_idx, x_idx = np.indices(volume.shape, dtype=np.float32)
        metal_dist = np.sqrt(
            ((z_idx - center_voxel[0]) / max(radius_voxel[0], 1)) ** 2
            + ((y_idx - center_voxel[1]) / max(radius_voxel[1], 1)) ** 2
            + ((x_idx - center_voxel[2]) / max(radius_voxel[2], 1)) ** 2
        )
        metal_mask = metal_dist <= 1.0

        result = volume.copy().astype(np.float32)

        dist_to_metal = distance_transform_edt(~metal_mask)

        # === Step 1: 变化条纹伪影（暗带+亮带交替） ===
        n_streaks = 14
        for z in range(nz):
            slice_2d = result[z]
            cy_z, cx_z = center_voxel[1], center_voxel[2]
            yy, xx = np.indices(slice_2d.shape, dtype=np.float32)

            theta = np.arctan2(yy - cy_z, xx - cx_z)
            r = np.sqrt((yy - cy_z) ** 2 + (xx - cx_z) ** 2)

            streak_pattern = np.zeros_like(slice_2d)
            bright_pattern = np.zeros_like(slice_2d)

            for i in range(n_streaks):
                angle = 2 * np.pi * i / n_streaks + self.rng.uniform(-0.15, 0.15)
                angular_dist = np.abs(np.sin(theta - angle))
                # 锐化条纹核心，增加旁瓣
                streak_core = np.exp(-(angular_dist ** 2) / 0.01)
                streak_side = 0.3 * np.exp(-(angular_dist ** 2) / 0.08)
                streak_weight = streak_core + streak_side

                # 径向衰减 + 近金属端更亮
                max_r = max(ny, nx)
                streak_decay = np.exp(-r / max_r * 2.5)

                # 条纹强度沿角度有随机变化
                intensity_mod = 1.0 + 0.3 * np.sin(theta * 3 + self.rng.uniform(0, 2 * np.pi))

                # 暗带占70%，亮带占30%
                if self.rng.random() < 0.7:
                    streak_pattern += streak_weight * streak_decay * intensity_mod
                else:
                    bright_pattern += streak_weight * streak_decay * intensity_mod * 0.5

            streak_pattern = np.clip(streak_pattern, 0, 1)
            bright_pattern = np.clip(bright_pattern, 0, 1)

            tissue_mask_z = volume[z] > -500
            affect_mask_z = tissue_mask_z & (~metal_mask[z])

            dark_streaks = streak_pattern * p["streak_intensity"] * 600.0
            bright_streaks = bright_pattern * p["streak_intensity"] * 300.0

            slice_2d[affect_mask_z] -= dark_streaks[affect_mask_z]
            slice_2d[affect_mask_z] += bright_streaks[affect_mask_z]
            result[z] = slice_2d

        # === Step 2: 亮环效应 — 金属边缘高HU环 ===
        ring_mask = (dist_to_metal > 0) & (dist_to_metal <= 3.0)
        if np.any(ring_mask):
            ring_intensity = np.exp(-dist_to_metal[ring_mask] / 1.5) * 150.0
            result[ring_mask] += ring_intensity.astype(np.float32)

        # === Step 3: 射束硬化 — EDT距离场驱动的杯状效应 ===
        bh_radius = 35
        bh_mask = (dist_to_metal > 0) & (dist_to_metal <= bh_radius)
        if np.any(bh_mask):
            bh_weight = (1 - dist_to_metal[bh_mask] / bh_radius)
            # 非线性衰减：近金属端更强
            bh_weight = bh_weight ** 1.5
            hu_reduction = bh_weight * p["beam_hardening_strength"] * 250.0
            hu_reduction *= (1 + self.rng.normal(0, 0.05, size=np.sum(bh_mask)))
            result[bh_mask] -= hu_reduction.astype(np.float32)

        # === Step 4: 光子饥饿噪声 — 金属阴影区域噪声剧增 ===
        if p["photon_starvation_noise"] > 0:
            noise_sigma_base = p["photon_starvation_noise"] * 20.0
            # 基础电子噪声（全图）
            base_noise = self.rng.normal(0, noise_sigma_base * 0.3, size=volume.shape).astype(np.float32)

            # 金属附近噪声剧增（光子饥饿效应）
            ps_mask = dist_to_metal <= bh_radius
            if np.any(ps_mask):
                ps_weight = np.exp(-dist_to_metal[ps_mask] / (bh_radius * 0.3))
                ps_noise = self.rng.normal(0, noise_sigma_base * 2.0, size=np.sum(ps_mask)).astype(np.float32)
                base_noise[ps_mask] += ps_noise * ps_weight

            tissue_mask = volume > -200
            result[tissue_mask] += base_noise[tissue_mask]

        # === Step 5: 设置金属区域 ===
        result[metal_mask] = metal_hu

        # 构建伪影掩码
        artifact_mask = (
            metal_mask
            | (dist_to_metal <= bh_radius)
            | (ring_mask if np.any(ring_mask) else metal_mask)
            | (np.abs(result - volume) > 5)
        ).astype(np.float32)

        metadata = {
            "artifact_type": self.get_artifact_type(),
            "params": p,
            "metal_mask_voxel_count": int(np.sum(metal_mask)),
            "artifact_region_voxel_count": int(np.sum(artifact_mask > 0)),
        }

        return np.clip(result, -1024, 3071), artifact_mask, metadata
