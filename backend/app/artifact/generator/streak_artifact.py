"""条状伪影生成器 — 在 sinogram 域模拟高密度区域引起的直线暗带（增强版）"""

import numpy as np
from skimage.transform import radon, iradon
from .base import BaseArtifactGenerator
from typing import Dict, Any, Tuple


class StreakArtifactGenerator(BaseArtifactGenerator):
    """条状伪影生成器 — 穿过高密度区域的直线暗带/亮带

    增强特性:
    - 暗带+亮带交替：真实条纹伪影同时有暗带和亮带
    - 强度渐变：条纹沿长度方向强度不均匀
    - 阴影效应：条纹之间的区域轻微变暗
    - 锥束伪影：可选的z方向渐变
    """

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "num_streaks": 5,
            "intensity": 60.0,
            "streak_positions": None,
            "cone_beam": False,
            "cone_strength": 0.3,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        if "num_streaks" in params and params["num_streaks"] < 1:
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
        num_streaks = p["num_streaks"]
        intensity = p["intensity"]
        nz, ny, nx = volume.shape

        result = volume.copy().astype(np.float32)

        num_angles = min(max(ny, nx), 180)
        theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)

        for z in range(nz):
            slice_2d = volume[z].astype(np.float64)

            sinogram = radon(slice_2d, theta=theta, circle=True)

            if p["streak_positions"] is not None:
                bad_angles = np.array(p["streak_positions"])
            else:
                bad_angles = self.rng.choice(
                    sinogram.shape[1], size=min(num_streaks, sinogram.shape[1]), replace=False
                )

            for ch in bad_angles:
                sign = self.rng.choice([-1, 1])

                # 条纹宽度变化
                streak_width = self.rng.integers(2, max(4, sinogram.shape[0] // 6))
                center = self.rng.integers(streak_width, sinogram.shape[0] - streak_width)
                start = max(0, center - streak_width)
                end = min(sinogram.shape[0], center + streak_width)

                # 沿条纹长度的强度渐变（中间强，两端弱）
                length = end - start
                profile = np.sin(np.linspace(0, np.pi, length))
                # 添加随机不对称性
                asymmetry = self.rng.uniform(0.6, 1.4)
                profile = profile ** asymmetry

                sinogram[start:end, ch] += intensity * sign * profile

                # 亮带：紧邻暗带的两侧有轻微亮条
                if sign < 0:
                    for neighbor in [-2, -1, 1, 2]:
                        nch = ch + neighbor
                        if 0 <= nch < sinogram.shape[1]:
                            bright_profile = np.sin(np.linspace(0, np.pi, length)) * 0.2
                            sinogram[start:end, nch] += intensity * 0.15 * bright_profile

            reconstructed = iradon(sinogram, theta=theta, circle=True, filter_name="ramp")
            result[z] = reconstructed.astype(np.float32)

        # 条纹间阴影效应：在条纹密集区域轻微降低HU
        if num_streaks >= 3:
            diff_map = np.abs(result - volume)
            # 高差异区域（条纹）周围降低
            from scipy.ndimage import gaussian_filter
            streak_density = gaussian_filter((diff_map > intensity * 0.3).astype(np.float32), sigma=3.0)
            shadow = streak_density * intensity * 0.1
            result -= shadow.astype(np.float32)

        # 锥束伪影
        if p["cone_beam"] and nz > 1:
            cone_strength = p["cone_strength"]
            z_profile = np.ones(nz, dtype=np.float32)
            mid = nz // 2
            for z in range(nz):
                z_profile[z] = 1.0 + cone_strength * ((z - mid) / max(mid, 1)) ** 2
            for z in range(nz):
                result[z] *= z_profile[z]

        result = np.clip(result, -1024, 3071).astype(np.float32)

        artifact_mask = np.zeros_like(volume, dtype=np.float32)
        diff = np.abs(result - volume)
        artifact_mask[diff > 2] = 1.0

        metadata = {
            "artifact_type": self.get_artifact_type(),
            "params": p,
            "num_streaks": num_streaks,
            "intensity": intensity,
            "cone_beam": p["cone_beam"],
        }

        return result, artifact_mask, metadata
