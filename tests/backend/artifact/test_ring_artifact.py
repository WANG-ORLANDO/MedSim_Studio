"""RingArtifactGenerator 单元测试"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend"))

from app.artifact.generator.ring_artifact import RingArtifactGenerator


@pytest.fixture
def uniform_volume():
    """64³ 均匀假体"""
    return np.ones((64, 64, 64), dtype=np.float32) * 40


@pytest.fixture
def generator():
    return RingArtifactGenerator(seed=42)


class TestRingArtifactBasic:
    def test_output_shapes(self, generator, uniform_volume):
        result, mask, meta = generator.generate(
            uniform_volume, (1.0, 1.0, 1.0), generator.get_default_params()
        )
        assert result.shape == uniform_volume.shape
        assert mask.shape == uniform_volume.shape
        assert result.dtype == np.float32

    def test_artifact_changes_volume(self, generator, uniform_volume):
        result, _, _ = generator.generate(
            uniform_volume, (1.0, 1.0, 1.0), {"num_rings": 5, "intensity": 100.0}
        )
        assert np.any(result != uniform_volume)

    def test_metadata_fields(self, generator, uniform_volume):
        _, _, meta = generator.generate(
            uniform_volume, (1.0, 1.0, 1.0), {"num_rings": 3}
        )
        assert meta["artifact_type"] == "ringartifact"
        assert meta["num_rings"] == 3
        assert len(meta["affected_channels"]) == 3


class TestRingArtifactParametrized:
    @pytest.mark.parametrize("num_rings", [1, 3, 5, 10])
    def test_different_ring_counts(self, num_rings):
        gen = RingArtifactGenerator(seed=0)
        vol = np.ones((32, 32, 32), dtype=np.float32) * 40
        result, _, meta = gen.generate(vol, (1.0, 1.0, 1.0), {"num_rings": num_rings})
        assert result.shape == vol.shape
        assert meta["num_rings"] == num_rings

    @pytest.mark.parametrize("intensity", [10.0, 50.0, 100.0, 200.0])
    def test_different_intensities(self, intensity):
        gen = RingArtifactGenerator(seed=0)
        vol = np.ones((32, 32, 32), dtype=np.float32) * 40
        result, _, _ = gen.generate(vol, (1.0, 1.0, 1.0), {"intensity": intensity})
        assert result.shape == vol.shape


class TestRingArtifactEdgeCases:
    def test_ring_positions_specified(self, generator, uniform_volume):
        result, _, meta = generator.generate(
            uniform_volume, (1.0, 1.0, 1.0), {"ring_positions": [10, 20, 30]}
        )
        assert meta["affected_channels"] == [10, 20, 30]

    def test_clipping(self, generator, uniform_volume):
        result, _, _ = generator.generate(
            uniform_volume, (1.0, 1.0, 1.0), {"intensity": 5000.0}
        )
        assert np.min(result) >= -1024
        assert np.max(result) <= 3071

    def test_small_volume(self):
        gen = RingArtifactGenerator(seed=0)
        vol = np.ones((4, 8, 8), dtype=np.float32) * 40
        result, _, _ = gen.generate(vol, (1.0, 1.0, 1.0), {"num_rings": 2})
        assert result.shape == vol.shape


class TestValidateParams:
    def test_valid(self, generator):
        assert generator.validate_params({}) is True
        assert generator.validate_params({"num_rings": 3}) is True

    def test_invalid_num_rings(self, generator):
        assert generator.validate_params({"num_rings": 0}) is False

    def test_negative_intensity(self, generator):
        assert generator.validate_params({"intensity": -10}) is False


def test_seed_reproducibility():
    vol = np.ones((16, 16, 16), dtype=np.float32) * 40
    params = {"num_rings": 3, "intensity": 50.0}

    r1, _, _ = RingArtifactGenerator(seed=88).generate(vol, (1.0, 1.0, 1.0), params)
    r2, _, _ = RingArtifactGenerator(seed=88).generate(vol, (1.0, 1.0, 1.0), params)
    np.testing.assert_array_equal(r1, r2)


def test_concentric_rings_visible():
    """在均匀假体上应能检测到环状结构"""
    gen = RingArtifactGenerator(seed=42)
    vol = np.ones((1, 64, 64), dtype=np.float32) * 40
    result, _, _ = gen.generate(vol, (1.0, 1.0, 1.0), {"num_rings": 5, "intensity": 100.0})
    # 结果应有非均匀分布（环的存在）
    assert np.std(result[0]) > np.std(vol[0])


def test_save_slice_png(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen = RingArtifactGenerator(seed=42)
    vol = np.ones((1, 64, 64), dtype=np.float32) * 40
    result, _, _ = gen.generate(vol, (1.0, 1.0, 1.0), {"num_rings": 5, "intensity": 80.0})

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(vol[0], cmap="gray", vmin=-200, vmax=3000)
    axes[0].set_title("Original (uniform 40 HU)")
    axes[1].imshow(result[0], cmap="gray", vmin=-200, vmax=3000)
    axes[1].set_title("Ring Artifact (5 rings)")
    for ax in axes:
        ax.axis("off")

    out_path = tmp_path / "ring_artifact_slices.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert out_path.exists()
    print(f"✅ 截图已保存: {out_path}")
