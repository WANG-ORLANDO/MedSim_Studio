"""量子噪声伪影生成器 — 基于 Poisson 光子计数模型（增强版）"""

import numpy as np
from scipy.ndimage import gaussian_filter
from .base import BaseArtifactGenerator
from typing import Dict, Any, Tuple


def hu_to_attenuation(hu: np.ndarray, mu_water: float = 0.2) -> np.ndarray:
    """HU -> 线性衰减系数 (cm⁻¹)"""
    return (hu / 1000.0 + 1.0) * mu_water


def attenuation_to_hu(mu: np.ndarray, mu_water: float = 0.2) -> np.ndarray:
    """线性衰减系数 -> HU"""
    return (mu / mu_water - 1.0) * 1000.0


def apply_quantum_noise(
    volume: np.ndarray,
    mAs: float,
    reference_mAs: float = 150.0,
    slice_thickness_mm: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """施加量子噪声到 CT 体积"""
    if rng is None:
        rng = np.random.default_rng()

    N0 = 1e5 * (mAs / reference_mAs) * (slice_thickness_mm / 1.0)
    mu = hu_to_attenuation(volume)
    expected_photons = N0 * np.exp(-mu)
    expected_photons = np.clip(expected_photons, 0.1, None)
    detected_photons = rng.poisson(expected_photons).astype(np.float64)
    mu_noisy = -np.log(np.clip(detected_photons / N0, 1e-6, None))
    return attenuation_to_hu(mu_noisy).astype(np.float32)


class NoiseArtifactGenerator(BaseArtifactGenerator):
    """量子噪声伪影生成器 — 基于 Poisson 光子计数物理模型

    增强特性:
    - 空间变化噪声：高密度区域（骨骼）噪声更大
    - 电子噪声：可叠加高斯加性噪声
    - 相关噪声纹理（mottle）：模拟重建核引起的噪声相关性
    - 杯状噪声：低mAs下中心噪声略低于边缘
    """

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "mAs": 50.0,
            "reference_mAs": 150.0,
            "slice_thickness_mm": 1.0,
            "mu_water": 0.2,
            "electronic_noise_sigma": 5.0,
            "mottle_sigma": 3.0,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        required = ["mAs"]
        return all(k in params for k in required) and params["mAs"] > 0

    def generate(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        params: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        p = {**self.get_default_params(), **params}
        mu_water = p["mu_water"]
        nz, ny, nx = volume.shape

        # Step 1: Poisson 量子噪声（空间变化：高密度区域噪声更大）
        base_result = apply_quantum_noise(
            volume,
            mAs=p["mAs"],
            reference_mAs=p["reference_mAs"],
            slice_thickness_mm=p["slice_thickness_mm"],
            rng=self.rng,
        )

        # 计算空间变化权重：高HU区域光子衰减更多→噪声更大
        hu_range = np.clip(volume, -1024, 3071)
        density_weight = np.clip((hu_range + 200) / 1200, 0.5, 2.0)

        # 基础噪声差异
        noise_diff = base_result - volume
        # 用密度权重调制噪声
        result = volume + noise_diff * density_weight

        # Step 2: 电子噪声（高斯加性噪声，均匀分布）
        if p["electronic_noise_sigma"] > 0:
            elec_noise = self.rng.normal(
                0, p["electronic_noise_sigma"], size=volume.shape
            ).astype(np.float32)
            result += elec_noise

        # Step 3: 相关噪声纹理（mottle）— 模拟重建核引起的噪声空间相关性
        if p.get("mottle_sigma", 0) > 0:
            raw_noise = self.rng.normal(0, p["mottle_sigma"], size=volume.shape).astype(np.float32)
            # 用高斯滤波引入空间相关性（模拟重建核的平滑效应）
            mottle = gaussian_filter(raw_noise, sigma=[0.5, 1.5, 1.5])
            result += mottle

        # Step 4: 杯状噪声 — 低mAs下由于散射，中心区域信噪比略高
        mAs_ratio = p["mAs"] / p["reference_mAs"]
        if mAs_ratio < 0.8:
            # 创建径向距离场（每层独立）
            for z in range(nz):
                slice_2d = result[z]
                cy, cx = ny // 2, nx // 2
                yy, xx = np.mgrid[:ny, :nx]
                r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / max(ny, nx)
                # 边缘噪声略大
                cupping_noise = r * (1 - mAs_ratio) * 15.0
                slice_2d += cupping_noise.astype(np.float32)
                result[z] = slice_2d

        result = np.clip(result, -1024, 3071).astype(np.float32)

        # 掩码：所有体素都受噪声影响
        artifact_mask = np.ones_like(volume, dtype=np.float32)

        noise_std = float(np.std(result - volume))

        metadata = {
            "artifact_type": self.get_artifact_type(),
            "params": p,
            "noise_std": noise_std,
            "expected_noise_ratio": float(1.0 / np.sqrt(p["mAs"] / p["reference_mAs"])),
        }

        return result, artifact_mask, metadata
