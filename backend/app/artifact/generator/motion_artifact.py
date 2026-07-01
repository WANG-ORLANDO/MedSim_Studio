"""运动伪影生成器 — 模拟呼吸/心跳/随机运动引起的伪影（增强版）"""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from .base import BaseArtifactGenerator
from typing import Dict, Any, Tuple


class MotionArtifactGenerator(BaseArtifactGenerator):
    """运动伪影生成器 — 通过位移场+重叠鬼影模拟运动

    增强特性:
    - 鬼影效应：位移后的结构以半透明叠加回原位
    - 多方向运动：呼吸同时有前后+上下分量
    - 运动模糊：沿位移方向的方向性模糊
    - 层间不连续：模拟螺旋CT逐层扫描的运动不一致
    """

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "motion_type": "respiratory",
            "amplitude_mm": 10.0,
            "frequency_hz": 0.25,
            "direction": [0.0, 0.0, 1.0],
            "blur_sigma": 1.5,
            "ghosting_fraction": 0.3,
            "random_seed_shift": 0,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        if "motion_type" not in params:
            return False
        mt = params["motion_type"]
        if mt not in ("respiratory", "cardiac", "random"):
            return False
        if "amplitude_mm" in params and params["amplitude_mm"] < 0:
            return False
        return True

    def generate(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        params: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        p = {**self.get_default_params(), **params}
        nz, ny, nx = volume.shape
        motion_type = p["motion_type"]
        amplitude_mm = p["amplitude_mm"]

        displacement = np.zeros((nz, 3), dtype=np.float64)

        if motion_type == "respiratory":
            t = np.linspace(0, 2 * np.pi * p["frequency_hz"] * nz, nz)
            # 前后运动为主（z方向），叠加少量上下（y方向）
            displacement[:, 2] = amplitude_mm / spacing[2] * np.sin(t)
            displacement[:, 1] = amplitude_mm * 0.3 / spacing[1] * np.sin(t + 0.5)
            # 偶尔的突跳运动
            jump_mask = self.rng.random(nz) < 0.05
            displacement[jump_mask, 2] += self.rng.normal(0, amplitude_mm * 0.5 / spacing[2], size=np.sum(jump_mask))

        elif motion_type == "cardiac":
            t = np.linspace(0, 4 * np.pi, nz)
            displacement[:, 2] = (amplitude_mm * 0.3) / spacing[2] * np.sin(t)
            displacement[:, 1] = (amplitude_mm * 0.2) / spacing[1] * np.cos(t * 1.5)
            # 心脏搏动的快速回弹
            systolic = np.exp(-((t % (2 * np.pi)) - 0.5) ** 2 / 0.1)
            displacement[:, 0] = systolic * amplitude_mm * 0.1 / spacing[0]

        elif motion_type == "random":
            displacement[:, 0] = self.rng.normal(0, amplitude_mm / spacing[0], nz)
            displacement[:, 1] = self.rng.normal(0, amplitude_mm / spacing[1], nz)
            displacement[:, 2] = self.rng.normal(0, amplitude_mm / spacing[2], nz)
            # 平滑随机运动（不是纯白噪声）
            for axis in range(3):
                displacement[:, axis] = gaussian_filter(displacement[:, axis], sigma=2.0)

        # 构建坐标网格并应用位移
        z_idx, y_idx, x_idx = np.meshgrid(
            np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
        )

        warped_z = z_idx.copy().astype(np.float64)
        warped_y = y_idx.copy().astype(np.float64)
        warped_x = x_idx.copy().astype(np.float64)

        for z in range(nz):
            warped_z[z] -= displacement[z, 0]
            warped_y[z] -= displacement[z, 1]
            warped_x[z] -= displacement[z, 2]

        coords = np.array([warped_z, warped_y, warped_x])
        result = map_coordinates(
            volume.astype(np.float64), coords, order=1, mode="constant", cval=-1024
        ).astype(np.float32)

        # === 鬼影效应：位移后的结构半透明叠加回原始位置 ===
        ghosting = p.get("ghosting_fraction", 0.3)
        if ghosting > 0 and np.max(np.abs(displacement)) > 0.5:
            # 反向位移坐标
            unwarped_z = z_idx.copy().astype(np.float64)
            unwarped_y = y_idx.copy().astype(np.float64)
            unwarped_x = x_idx.copy().astype(np.float64)
            for z in range(nz):
                unwarped_z[z] += displacement[z, 0]
                unwarped_y[z] += displacement[z, 1]
                unwarped_x[z] += displacement[z, 2]

            ghost_coords = np.array([unwarped_z, unwarped_y, unwarped_x])
            ghost_image = map_coordinates(
                result.astype(np.float64), ghost_coords, order=1, mode="constant", cval=-1024
            ).astype(np.float32)
            # 鬼影与原图混合
            result = result * (1 - ghosting) + ghost_image * ghosting

        # === 方向性运动模糊 ===
        blur_sigma = p["blur_sigma"]
        if blur_sigma > 0 and np.max(np.abs(displacement)) > 1e-6:
            direction = np.array(p["direction"], dtype=np.float64)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            # 沿运动方向的模糊强度与局部位移量成正比
            sigma_per_axis = blur_sigma * np.abs(direction)
            result = gaussian_filter(result, sigma=sigma_per_axis.tolist())

        # 运动影响掩码
        total_disp = np.sqrt(np.sum(displacement ** 2, axis=1))
        artifact_mask = np.zeros((nz, ny, nx), dtype=np.float32)
        for z in range(nz):
            if total_disp[z] > 0.1:
                # 鬼影区域也标记为受影响
                artifact_mask[z] = 1.0

        result = np.clip(result, -1024, 3071).astype(np.float32)

        mean_spacing = float(np.mean(spacing))
        total_disp_mm = total_disp * mean_spacing
        max_disp_mm = float(np.max(total_disp_mm))
        mean_disp_mm = float(np.mean(total_disp_mm))

        metadata = {
            "artifact_type": self.get_artifact_type(),
            "params": p,
            "max_displacement_mm": max_disp_mm,
            "mean_displacement_mm": mean_disp_mm,
            "affected_slices": int(np.sum(total_disp > 0.1)),
            "ghosting_fraction": ghosting,
        }

        return result, artifact_mask, metadata
