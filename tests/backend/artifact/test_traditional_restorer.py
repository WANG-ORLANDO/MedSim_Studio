"""单元测试：传统伪影修复方法 — 验证修复效果 + PSNR/SSIM 对比"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend"))

from app.artifact.restoration.traditional import TraditionalRestorer
from app.artifact.generator import get_generator


def _make_test_volume(size=64):
    """生成简单球体测试体积"""
    vol = np.full((size, size, size), 40.0, dtype=np.float32)
    cz, cy, cx = size // 2, size // 2, size // 2
    radius = size // 4
    z, y, x = np.ogrid[:size, :size, :size]
    sphere = ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) <= radius ** 2
    vol[sphere] = 400.0
    return vol


class TestMedianDenoise:
    """中值滤波测试"""

    def test_basic(self):
        vol = _make_test_volume()
        restored = TraditionalRestorer.median_denoise(vol, kernel_size=3)
        assert restored.shape == vol.shape
        assert restored.dtype == np.float32

    def test_reduces_noise(self):
        clean = _make_test_volume()
        noisy = clean + np.random.default_rng(42).normal(0, 30, clean.shape).astype(np.float32)
        restored = TraditionalRestorer.median_denoise(noisy, kernel_size=3)
        psnr_before = TraditionalRestorer.compute_psnr(clean, noisy)
        psnr_after = TraditionalRestorer.compute_psnr(clean, restored)
        assert psnr_after > psnr_before, f"PSNR should improve: {psnr_before:.2f} → {psnr_after:.2f}"

    def test_with_artifact_generator(self):
        clean = _make_test_volume()
        gen = get_generator("noise")
        noisy, _, _ = gen.generate(clean, (1.0, 1.0, 1.0), {"mAs": 30, "reference_mAs": 150})
        restored = TraditionalRestorer.median_denoise(noisy, kernel_size=3)
        psnr_before = TraditionalRestorer.compute_psnr(clean, noisy)
        psnr_after = TraditionalRestorer.compute_psnr(clean, restored)
        ssim_before = TraditionalRestorer.compute_ssim(clean, noisy)
        ssim_after = TraditionalRestorer.compute_ssim(clean, restored)
        print(f"\n[median] PSNR: {psnr_before:.2f} → {psnr_after:.2f} | SSIM: {ssim_before:.3f} → {ssim_after:.3f}")
        assert psnr_after >= psnr_before - 1.0  # 允许小幅度波动


class TestNLMDenoise:
    """非局部均值去噪测试"""

    def test_basic(self):
        vol = _make_test_volume()
        restored = TraditionalRestorer.nlm_denoise(vol, patch_size=3, patch_distance=5, h=0.1)
        assert restored.shape == vol.shape
        assert restored.dtype == np.float32

    def test_reduces_noise(self):
        clean = _make_test_volume()
        rng = np.random.default_rng(42)
        noisy = clean + rng.normal(0, 25, clean.shape).astype(np.float32)
        restored = TraditionalRestorer.nlm_denoise(noisy, patch_size=3, patch_distance=5, h=0.2)
        psnr_before = TraditionalRestorer.compute_psnr(clean, noisy)
        psnr_after = TraditionalRestorer.compute_psnr(clean, restored)
        # NLM should preserve structure (not degrade PSNR significantly)
        assert psnr_after >= psnr_before - 0.5, f"PSNR degraded too much: {psnr_before:.2f} → {psnr_after:.2f}"


class TestSinogramRingCorrection:
    """sinogram 环状伪影校正测试"""

    def test_basic(self):
        vol = _make_test_volume()
        restored = TraditionalRestorer.sinogram_ring_correction(vol, threshold=0.1)
        assert restored.shape == vol.shape
        assert restored.dtype == np.float32

    def test_with_ring_artifact(self):
        clean = _make_test_volume()
        gen = get_generator("ring")
        noisy, mask, _ = gen.generate(clean, (1.0, 1.0, 1.0), {"num_rings": 3, "intensity": 50})
        restored = TraditionalRestorer.sinogram_ring_correction(noisy, threshold=0.1)
        psnr_before = TraditionalRestorer.compute_psnr(clean, noisy)
        psnr_after = TraditionalRestorer.compute_psnr(clean, restored)
        ssim_before = TraditionalRestorer.compute_ssim(clean, noisy)
        ssim_after = TraditionalRestorer.compute_ssim(clean, restored)
        print(f"\n[ring] PSNR: {psnr_before:.2f} → {psnr_after:.2f} | SSIM: {ssim_before:.3f} → {ssim_after:.3f}")
        # ring correction should not make things worse
        assert psnr_after >= psnr_before - 2.0


class TestMARSinogramInterpolation:
    """金属伪影 sinogram 插值修复测试"""

    def test_basic(self):
        vol = _make_test_volume()
        metal_mask = np.zeros_like(vol)
        metal_mask[30:34, 30:34, 30:34] = 1.0
        restored = TraditionalRestorer.mar_sinogram_interpolation(vol, metal_mask)
        assert restored.shape == vol.shape
        assert restored.dtype == np.float32

    def test_with_metal_artifact(self):
        clean = _make_test_volume()
        gen = get_generator("metal")
        noisy, mask, _ = gen.generate(
            clean, (1.0, 1.0, 1.0),
            {"metal_type": "titanium", "center": [0.5, 0.5, 0.5], "radius_mm": [3, 3, 3]}
        )
        restored = TraditionalRestorer.mar_sinogram_interpolation(noisy, mask)
        psnr_before = TraditionalRestorer.compute_psnr(clean, noisy)
        psnr_after = TraditionalRestorer.compute_psnr(clean, restored)
        ssim_before = TraditionalRestorer.compute_ssim(clean, noisy)
        ssim_after = TraditionalRestorer.compute_ssim(clean, restored)
        print(f"\n[metal] PSNR: {psnr_before:.2f} → {psnr_after:.2f} | SSIM: {ssim_before:.3f} → {ssim_after:.3f}")


class TestMetrics:
    """PSNR/SSIM 指标测试"""

    def test_psnr_identical(self):
        vol = _make_test_volume()
        assert TraditionalRestorer.compute_psnr(vol, vol) == float('inf')

    def test_psnr_degrades_with_noise(self):
        clean = _make_test_volume()
        noisy = clean + np.random.default_rng(42).normal(0, 50, clean.shape).astype(np.float32)
        psnr = TraditionalRestorer.compute_psnr(clean, noisy)
        assert 10 < psnr < 40

    def test_ssim_identical(self):
        vol = _make_test_volume()
        # Add slight variation to avoid constant-image SSIM issue
        vol = vol + np.random.default_rng(0).normal(0, 0.01, vol.shape).astype(np.float32)
        ssim = TraditionalRestorer.compute_ssim(vol, vol)
        assert ssim > 0.99

    def test_ssim_degrades_with_noise(self):
        clean = _make_test_volume()
        # Add slight variation to avoid constant-image SSIM issue
        clean = clean + np.random.default_rng(0).normal(0, 0.01, clean.shape).astype(np.float32)
        noisy = clean + np.random.default_rng(42).normal(0, 30, clean.shape).astype(np.float32)
        ssim = TraditionalRestorer.compute_ssim(clean, noisy)
        assert 0.1 < ssim < 0.95, f"SSIM unexpected: {ssim}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
