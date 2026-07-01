"""环状伪影生成器 — 基于 Radon 变换的探测器通道增益不一致模型（增强版）"""

import numpy as np
from skimage.transform import radon, iradon
from .base import BaseArtifactGenerator
from typing import Dict, Any, Tuple


class RingArtifactGenerator(BaseArtifactGenerator):
    """环状伪影生成器 — 在 sinogram 域模拟缺陷探测器通道

    增强特性:
    - Z轴变化：环伪影沿z轴有渐变强度（模拟通道增益漂移）
    - 部分环：支持只在部分角度出现的环伪影
    - 环宽度变化：不同环有不同宽度
    - 双极性：同时产生亮环和暗环
    """

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "num_rings": 3,
            "intensity": 50.0,
            "ring_positions": None,
            "partial_ring": False,
            "z_variation": True,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        if "num_rings" in params and params["num_rings"] < 1:
            return False
        if "intensity" in params and params["intensity"] < 0:
            return False
        return True

    def generate(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        params: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        p = {**self.get_default_params(), **params}
        num_rings = p["num_rings"]
        intensity = p["intensity"]
        nz, ny, nx = volume.shape
        partial_ring = p.get("partial_ring", False)
        z_variation = p.get("z_variation", True)

        result = volume.copy().astype(np.float32)

        ring_channels = None
        if p["ring_positions"] is not None:
            ring_channels = p["ring_positions"]

        affected_channels = []

        num_angles = min(max(ny, nx), 180)
        theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)

        # 预生成每层的z轴权重
        if z_variation and nz > 1:
            z_weights = np.ones(nz, dtype=np.float64)
            # 每个环有独立的z轴渐变模式
            for r_idx in range(num_rings):
                # 随机选择渐变模式：线性衰减 / 中间强两端弱 / 突变
                mode = self.rng.choice(["linear", "center", "step"])
                if mode == "linear":
                    w = np.linspace(0.3, 1.0, nz)
                    if self.rng.random() < 0.5:
                        w = w[::-1]
                elif mode == "center":
                    mid = nz / 2
                    w = np.exp(-((np.arange(nz) - mid) / (nz * 0.3)) ** 2)
                    w = 0.3 + 0.7 * w
                else:
                    step_pos = self.rng.integers(nz // 4, 3 * nz // 4)
                    w = np.ones(nz)
                    w[:step_pos] = 0.2
                z_weights *= w
        else:
            z_weights = np.ones(nz, dtype=np.float64)

        for z in range(nz):
            slice_2d = volume[z].astype(np.float64)

            sinogram = radon(slice_2d, theta=theta, circle=True)

            if ring_channels is not None:
                bad_channels = np.array(ring_channels)
            else:
                bad_channels = self.rng.choice(
                    sinogram.shape[1], size=min(num_rings, sinogram.shape[1]), replace=False
                )

            if z == 0:
                affected_channels = bad_channels.tolist()

            # 每个环的强度和极性
            for ch_idx, ch in enumerate(bad_channels):
                sign = self.rng.choice([-1, 1])
                # 环宽度变化：有些窄有些宽
                ring_width = self.rng.choice([1, 1, 2, 3])
                z_w = z_weights[z]
                ch_intensity = intensity * sign * z_w

                for w_off in range(-ring_width // 2, ring_width // 2 + 1):
                    target_ch = ch + w_off
                    if 0 <= target_ch < sinogram.shape[1]:
                        sinogram[:, target_ch] += ch_intensity

                # 部分环：只在部分角度范围内生效
                if partial_ring:
                    angle_range = self.rng.integers(num_angles // 4, num_angles // 2)
                    angle_start = self.rng.integers(0, num_angles - angle_range)
                    mask = np.zeros(sinogram.shape[0], dtype=np.float64)
                    mask[angle_start:angle_start + angle_range] = 1.0
                    # 平滑过渡
                    mask = np.convolve(mask, np.ones(5) / 5, mode="same")
                    sinogram[:, ch] *= (1 + mask * 0.5 * sign)

            reconstructed = iradon(sinogram, theta=theta, circle=True, filter_name="ramp")
            result[z] = reconstructed.astype(np.float32)

        result = np.clip(result, -1024, 3071).astype(np.float32)

        # 掩码：标记受影响的环形区域
        artifact_mask = np.zeros_like(volume, dtype=np.float32)
        diff = np.abs(result - volume)
        artifact_mask[diff > 2] = 1.0

        metadata = {
            "artifact_type": self.get_artifact_type(),
            "params": p,
            "num_rings": num_rings,
            "intensity": intensity,
            "affected_channels": affected_channels,
            "z_variation": z_variation,
            "partial_ring": partial_ring,
        }

        return result, artifact_mask, metadata
