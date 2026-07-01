"""单元测试：质量评估模块 + 混合修复策略"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend"))

from app.artifact.evaluation.metrics import (
    compute_psnr, compute_ssim, compute_nmse, compute_mae_hu,
    compute_all_metrics, format_quality_report,
)
from app.artifact.restoration.hybrid import HybridRestorer, ARTIFACT_PRIORITY, RESTORATION_STRATEGY
from app.artifact.generator import get_generator


def _make_test_volume(size=32):
    """生成简单球体测试体积"""
    vol = np.full((size, size, size), 40.0, dtype=np.float32)
    cz, cy, cx = size // 2, size // 2, size // 2
    radius = size // 4
    z, y, x = np.ogrid[:size, :size, :size]
    sphere = ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) <= radius ** 2
    vol[sphere] = 400.0
    return vol


# ============================================================
# 质量评估模块测试
# ============================================================

class TestPSNR:
    def test_identical(self):
        vol = _make_test_volume()
        assert compute_psnr(vol, vol) >= 999.0

    def test_noisy(self):
        clean = _make_test_volume()
        noisy = clean + np.random.default_rng(42).normal(0, 30, clean.shape).astype(np.float32)
        psnr = compute_psnr(clean, noisy)
        assert 15 < psnr < 40

    def test_2d(self):
        img = np.full((32, 32), 100.0, dtype=np.float32)
        assert compute_psnr(img, img) >= 999.0


class TestSSIM:
    def test_identical(self):
        vol = _make_test_volume()
        vol = vol + np.random.default_rng(0).normal(0, 0.01, vol.shape).astype(np.float32)
        assert compute_ssim(vol, vol) > 0.99

    def test_2d(self):
        img = np.random.default_rng(42).normal(100, 20, (32, 32)).astype(np.float32)
        noisy = img + np.random.default_rng(42).normal(0, 30, img.shape).astype(np.float32)
        s = compute_ssim(img, noisy)
        assert 0.1 < s < 0.95


class TestNMSE:
    def test_identical(self):
        vol = _make_test_volume()
        assert compute_nmse(vol, vol) == 0.0

    def test_noisy(self):
        clean = _make_test_volume()
        noisy = clean + np.random.default_rng(42).normal(0, 30, clean.shape).astype(np.float32)
        nmse = compute_nmse(clean, noisy)
        assert 0 < nmse < 1.0


class TestMAEHU:
    def test_identical(self):
        vol = _make_test_volume()
        assert compute_mae_hu(vol, vol) == 0.0

    def test_offset(self):
        clean = _make_test_volume()
        shifted = clean + 50.0
        assert abs(compute_mae_hu(clean, shifted) - 50.0) < 0.01


class TestComputeAllMetrics:
    def test_without_artifact(self):
        clean = _make_test_volume()
        metrics = compute_all_metrics(clean, clean)
        assert "psnr" in metrics
        assert "ssim" in metrics
        assert "nmse" in metrics
        assert "mae_hu" in metrics

    def test_with_artifact(self):
        clean = _make_test_volume()
        noisy = clean + np.random.default_rng(42).normal(0, 30, clean.shape).astype(np.float32)
        metrics = compute_all_metrics(clean, clean, artifact=noisy)
        assert "psnr_before" in metrics
        assert "ssim_before" in metrics
        assert "psnr_improvement" in metrics

    def test_report_format(self):
        clean = _make_test_volume()
        noisy = clean + np.random.default_rng(42).normal(0, 30, clean.shape).astype(np.float32)
        metrics = compute_all_metrics(clean, clean, artifact=noisy)
        report = format_quality_report(metrics)
        assert "PSNR" in report
        assert "SSIM" in report
        assert "NMSE" in report


# ============================================================
# 混合修复策略测试
# ============================================================

class TestHybridRestorer:
    def test_single_artifact_noise(self):
        clean = _make_test_volume()
        gen = get_generator("noise")
        noisy, _, _ = gen.generate(clean, (1.0, 1.0, 1.0), {"mAs": 30, "reference_mAs": 150})
        restorer = HybridRestorer()
        result = restorer.restore(noisy, artifact_types=["noise"], clean_reference=clean)
        assert result["restored_volume"].shape == clean.shape
        assert len(result["steps"]) >= 1
        assert "psnr" in result["quality_metrics"]
        assert result["steps"][0]["artifact_type"] == "noise"
        assert result["steps"][0]["strategy"] == "redcnn" or result["steps"][0]["strategy"] == "median_filter"

    def test_single_artifact_ring(self):
        clean = _make_test_volume()
        gen = get_generator("ring")
        noisy, mask, _ = gen.generate(clean, (1.0, 1.0, 1.0), {"num_rings": 3, "intensity": 50})
        restorer = HybridRestorer()
        result = restorer.restore(noisy, artifact_types=["ring"], clean_reference=clean)
        assert result["restored_volume"].shape == clean.shape
        assert result["steps"][0]["strategy"] == "sinogram_ring"

    def test_multi_artifact(self):
        clean = _make_test_volume()
        gen_n = get_generator("noise")
        noisy, _, _ = gen_n.generate(clean, (1.0, 1.0, 1.0), {"mAs": 50, "reference_mAs": 150})
        restorer = HybridRestorer()
        result = restorer.restore(noisy, artifact_types=["noise", "ring"], clean_reference=clean)
        assert len(result["steps"]) >= 1
        assert "report" in result

    def test_metal_with_mask(self):
        clean = _make_test_volume()
        gen = get_generator("metal")
        noisy, mask, _ = gen.generate(
            clean, (1.0, 1.0, 1.0),
            {"metal_type": "titanium", "center": [0.5, 0.5, 0.5], "radius_mm": [3, 3, 3]}
        )
        restorer = HybridRestorer()
        result = restorer.restore(noisy, artifact_types=["metal"], artifact_mask=mask, clean_reference=clean)
        assert result["restored_volume"].shape == clean.shape
        assert result["steps"][0]["strategy"] == "sinogram_mar"

    def test_passthrough_clean(self):
        clean = _make_test_volume()
        restorer = HybridRestorer()
        result = restorer.restore(clean, artifact_types=["clean"])
        assert len(result["steps"]) == 0

    def test_hu_clipping(self):
        vol = np.full((4, 8, 8), 5000.0, dtype=np.float32)  # 超出 HU 范围
        restorer = HybridRestorer()
        result = restorer.restore(vol, artifact_types=["noise"])
        assert result["restored_volume"].max() <= 3071.0
        assert result["restored_volume"].min() >= -1024.0

    def test_report_string(self):
        clean = _make_test_volume()
        noisy = clean + np.random.default_rng(42).normal(0, 20, clean.shape).astype(np.float32)
        restorer = HybridRestorer()
        result = restorer.restore(noisy, artifact_types=["noise"], clean_reference=clean)
        assert "Quality Report" in result["report"]
        assert "PSNR" in result["report"]


class TestArtifactPriority:
    def test_metal_before_noise(self):
        assert ARTIFACT_PRIORITY["metal"] < ARTIFACT_PRIORITY["noise"]

    def test_ring_before_noise(self):
        assert ARTIFACT_PRIORITY["ring"] < ARTIFACT_PRIORITY["noise"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
