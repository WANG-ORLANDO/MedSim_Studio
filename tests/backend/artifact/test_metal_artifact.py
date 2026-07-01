"""MetalArtifactGenerator 单元测试"""

import pytest
import numpy as np
import os

# 确保 app 包可导入
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend"))

from app.artifact.generator.metal_artifact import MetalArtifactGenerator


@pytest.fixture
def soft_tissue_volume():
    """64³ 软组织背景体积 (40 HU)"""
    return np.ones((64, 64, 64), dtype=np.float32) * 40


@pytest.fixture
def generator():
    return MetalArtifactGenerator(seed=42)


class TestMetalArtifactBasic:
    def test_output_shapes(self, generator, soft_tissue_volume):
        result, mask, meta = generator.generate(
            soft_tissue_volume, (1.0, 1.0, 1.0), generator.get_default_params()
        )
        assert result.shape == soft_tissue_volume.shape
        assert mask.shape == soft_tissue_volume.shape
        assert result.dtype == np.float32

    def test_metal_region_high_hu(self, generator, soft_tissue_volume):
        result, mask, meta = generator.generate(
            soft_tissue_volume, (1.0, 1.0, 1.0), generator.get_default_params()
        )
        assert 2500 <= np.max(result) <= 3071

    def test_artifact_changes_volume(self, generator, soft_tissue_volume):
        result, _, _ = generator.generate(
            soft_tissue_volume, (1.0, 1.0, 1.0), generator.get_default_params()
        )
        assert np.any(result != soft_tissue_volume)

    def test_metadata_contains_type(self, generator, soft_tissue_volume):
        _, _, meta = generator.generate(
            soft_tissue_volume, (1.0, 1.0, 1.0), generator.get_default_params()
        )
        assert meta["artifact_type"] == "metalartifact"
        assert meta["metal_mask_voxel_count"] > 0


class TestMetalArtifactParametrize:
    @pytest.mark.parametrize("metal_type", ["titanium", "stainless_steel", "dental_amalgam", "gold", "copper"])
    def test_all_metal_types(self, metal_type):
        gen = MetalArtifactGenerator(seed=0)
        vol = np.ones((32, 32, 32), dtype=np.float32) * 40
        params = {**gen.get_default_params(), "metal_type": metal_type}
        result, mask, _ = gen.generate(vol, (1.0, 1.0, 1.0), params)
        assert result.shape == vol.shape
        expected_hu = gen.METAL_HU[metal_type]
        assert np.max(result) >= expected_hu


class TestMetalArtifactEdgeCases:
    def test_metal_at_center(self, generator, soft_tissue_volume):
        params = {**generator.get_default_params(), "center": [0.5, 0.5, 0.5]}
        result, mask, _ = generator.generate(soft_tissue_volume, (1.0, 1.0, 1.0), params)
        cy, cx, cz = 32, 32, 32
        assert result[cy, cx, cz] >= 2500

    def test_metal_near_edge(self, generator, soft_tissue_volume):
        params = {**generator.get_default_params(), "center": [0.1, 0.5, 0.5]}
        result, mask, _ = generator.generate(soft_tissue_volume, (1.0, 1.0, 1.0), params)
        assert result.shape == soft_tissue_volume.shape
        assert np.max(result) >= 2500

    def test_small_metal(self, generator, soft_tissue_volume):
        params = {**generator.get_default_params(), "radius_mm": [1.0, 1.0, 1.0]}
        result, mask, meta = generator.generate(soft_tissue_volume, (1.0, 1.0, 1.0), params)
        assert meta["metal_mask_voxel_count"] > 0

    def test_large_metal(self, generator, soft_tissue_volume):
        params = {**generator.get_default_params(), "radius_mm": [20.0, 20.0, 20.0]}
        result, mask, meta = generator.generate(soft_tissue_volume, (1.0, 1.0, 1.0), params)
        assert meta["metal_mask_voxel_count"] > 100


class TestValidateParams:
    def test_valid_params(self, generator):
        assert generator.validate_params(generator.get_default_params()) is True

    def test_missing_required(self, generator):
        assert generator.validate_params({"metal_type": "titanium"}) is False


def test_seed_reproducibility():
    vol = np.ones((32, 32, 32), dtype=np.float32) * 40
    spacing = (1.0, 1.0, 1.0)
    params = MetalArtifactGenerator(seed=0).get_default_params()

    r1, m1, _ = MetalArtifactGenerator(seed=99).generate(vol, spacing, params)
    r2, m2, _ = MetalArtifactGenerator(seed=99).generate(vol, spacing, params)
    np.testing.assert_array_equal(r1, r2)


def test_save_slice_png(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen = MetalArtifactGenerator(seed=42)
    vol = np.ones((64, 64, 64), dtype=np.float32) * 40
    result, mask, _ = gen.generate(vol, (1.0, 1.0, 1.0), gen.get_default_params())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(result[32], cmap="gray", vmin=-200, vmax=3000)
    axes[0].set_title("After Metal Artifact")
    axes[1].imshow(mask[32], cmap="gray")
    axes[1].set_title("Artifact Mask")
    for ax in axes:
        ax.axis("off")

    out_path = tmp_path / "metal_artifact_slice.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert out_path.exists()
    print(f"✅ 截图已保存: {out_path}")
