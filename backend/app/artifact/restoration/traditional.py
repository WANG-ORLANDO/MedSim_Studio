"""传统方法伪影修复器 — 中值滤波 / NLM / sinogram 校正 / MAR 插值"""

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.transform import radon, iradon
from typing import Dict, Any, Optional


class TraditionalRestorer:
    """传统方法伪影修复器"""

    @staticmethod
    def median_denoise(volume: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """中值滤波去噪 (适用于椒盐噪声/环状伪影)"""
        return median_filter(volume, size=kernel_size).astype(np.float32)

    @staticmethod
    def nlm_denoise(
        volume: np.ndarray,
        patch_size: int = 5,
        patch_distance: int = 11,
        h: Optional[float] = None,
    ) -> np.ndarray:
        """非局部均值去噪 (适用于量子噪声)"""
        if h is None:
            h = 0.8 * estimate_sigma(volume, channel_axis=None)
        result = np.zeros_like(volume)
        for z in range(volume.shape[0]):
            result[z] = denoise_nl_means(
                volume[z],
                patch_size=patch_size,
                patch_distance=patch_distance,
                h=h,
                fast_mode=True,
            )
        return result.astype(np.float32)

    @staticmethod
    def sinogram_ring_correction(
        volume: np.ndarray,
        threshold: float = 0.1,
    ) -> np.ndarray:
        """sinogram 域环状伪影校正"""
        result = np.zeros_like(volume)
        for z in range(volume.shape[0]):
            theta = np.linspace(0., 180., max(volume[z].shape), endpoint=False)
            sino = radon(volume[z], theta=theta, circle=True)
            # 中值滤波去除纵向条纹 (环在sinogram中表现为纵向条纹)
            sino_corrected = sino - median_filter(
                sino, size=(1, 5)
            ) * threshold
            result[z] = iradon(sino_corrected, theta=theta, circle=True,
                                filter_name="ramp")
        return result.astype(np.float32)

    @staticmethod
    def mar_sinogram_interpolation(
        volume: np.ndarray,
        metal_mask: np.ndarray,
    ) -> np.ndarray:
        """
        金属伪影减少 (MAR) — sinogram 域插值方法

        1. 识别金属区域对应的 sinogram 轨迹
        2. 对受影响的投影数据进行插值
        3. 逆 Radon 变换重建
        """
        result = np.zeros_like(volume)
        for z in range(volume.shape[0]):
            theta = np.linspace(0., 180., max(volume[z].shape), endpoint=False)
            sino = radon(volume[z], theta=theta, circle=True)

            # 检测金属影响区域
            metal_sino = radon(metal_mask[z].astype(float), theta=theta, circle=True)
            affected = metal_sino > 0.01

            # 线性插值修复
            sino_corrected = sino.copy()
            for ch in range(sino.shape[1]):
                if affected[:, ch].any():
                    good = ~affected[:, ch]
                    if good.sum() >= 2:
                        sino_corrected[:, ch] = np.interp(
                            np.arange(sino.shape[0]),
                            np.where(good)[0],
                            sino[good, ch],
                        )

            result[z] = iradon(sino_corrected, theta=theta, circle=True,
                                filter_name="ramp")
        return result.astype(np.float32)

    @staticmethod
    def compute_psnr(original: np.ndarray, restored: np.ndarray) -> float:
        """计算 PSNR"""
        mse = np.mean((original.astype(np.float64) - restored.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        return 10.0 * np.log10(np.max(original) ** 2 / mse)

    @staticmethod
    def compute_ssim(original: np.ndarray, restored: np.ndarray) -> float:
        """计算 SSIM (简化版，逐切片计算后取平均)"""
        from skimage.metrics import structural_similarity as ssim
        scores = []
        for z in range(original.shape[0]):
            try:
                s = ssim(original[z], restored[z], data_range=original[z].max() - original[z].min())
                scores.append(s)
            except Exception:
                pass
        return float(np.mean(scores)) if scores else 0.0
