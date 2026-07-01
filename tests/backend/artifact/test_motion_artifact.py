"""MotionArtifactGenerator 单元测试"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend"))

from app.artifact.generator.motion_artifact import MotionArtifactGenerator


@pytest.fixture
def phantom_volume():
    """64³ 体积，中心有亮球结构用于检测位移"""
    vol = np.zeros((64, 64, 64), dtype=np.float32)
    z, y, x = np.ogrid[:64, :64, :64]
    sphere = ((z - 32) ** 2 + (y - 32) ** 2 + (x - 32) ** 2) <= 10 ** 2
    vol[sphere] = 1000.0
    return vol


@pytest.fixture
def generator():
    return MotionArtifactGenerator(seed=42)


class TestMotionArtifactBasic:
    def test_output_shapes(self, generator, phantom_volume):
        result, mask, meta = generator.generate(
            phantom_volume, (1.0, 1.0, 1.0), generator.get_default_params()
        )
        assert result.shape == phantom_volume.shape
        assert mask.shape == phantom_volume.shape
        assert result.dtype == np.float32

    def test_artifact_changes_volume(self, generator, phantom_volume):
        result, _, _ = generator.generate(
            phantom_volume, (1.0, 1.0, 1.0), {"amplitude_mm": 10.0}
        )
        assert np.any(result != phantom_volume)

    def test_metadata_fields(self, generator, phantom_volume):
        _, _, meta = generator.generate(
            phantom_volume, (1.0, 1.0, 1.0), generator.get_default_params()
        )
        assert meta["artifact_type"] == "motionartifact"
        assert meta["max_displacement_mm"] > 0
        assert meta["affected_slices"] > 0


class TestMotionTypes:
    def test_respiratory(self, generator, phantom_volume):
        result, mask, meta = generator.generate(
            phantom_volume, (1.0, 1.0, 1.0), {"motion_type": "respiratory", "amplitude_mm": 10.0}
        )
        assert result.shape == phantom_volume.shape
        assert meta["max_displacement_mm"] >= 9.9

    def test_cardiac(self, generator, phantom_volume):
        result, mask, meta = generator.generate(
            phantom_volume, (1.0, 1.0, 1.0), {"motion_type": "cardiac", "amplitude_mm": 5.0}
        )
        assert result.shape == phantom_volume.shape
        assert meta["max_displacement_mm"] > 0

    def test_random(self, generator, phantom_volume):
        result, mask, meta = generator.generate(
            phantom_volume, (1.0, 1.0, 1.0), {"motion_type": "random", "amplitude_mm": 8.0}
        )
        assert result.shape == phantom_volume.shape
        assert meta["affected_slices"] > 0


class TestMotionEdgeCases:
    def test_zero_amplitude(self, generator, phantom_volume):
        result, mask, _ = generator.generate(
            phantom_volume, (1.0, 1.0, 1.0), {"amplitude_mm": 0.0}
        )
        np.testing.assert_array_equal(result, phantom_volume)

    def test_large_amplitude_no_nan(self, generator, phantom_volume):
        result, _, _ = generator.generate(
            phantom_volume, (1.0, 1.0, 1.0), {"amplitude_mm": 50.0}
        )
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_non_uniform_spacing(self, generator, phantom_volume):
        result, _, _ = generator.generate(
            phantom_volume, (2.0, 0.5, 0.5), {"amplitude_mm": 5.0}
        )
        assert result.shape == phantom_volume.shape


class TestValidateParams:
    def test_valid(self, generator):
        assert generator.validate_params({"motion_type": "respiratory"}) is True

    def test_missing_type(self, generator):
        assert generator.validate_params({}) is False

    def test_invalid_type(self, generator):
        assert generator.validate_params({"motion_type": "invalid"}) is False

    def test_negative_amplitude(self, generator):
        assert generator.validate_params({"motion_type": "respiratory", "amplitude_mm": -1}) is False


def test_seed_reproducibility():
    vol = np.zeros((32, 32, 32), dtype=np.float32)
    vol[16, 16, 16] = 1000.0
    params = {"motion_type": "respiratory", "amplitude_mm": 5.0}

    r1, m1, _ = MotionArtifactGenerator(seed=55).generate(vol, (1.0, 1.0, 1.0), params)
    r2, m2, _ = MotionArtifactGenerator(seed=55).generate(vol, (1.0, 1.0, 1.0), params)
    np.testing.assert_array_equal(r1, r2)


def test_misalignment_between_slices(generator, phantom_volume):
    """呼吸运动应导致切片间结构错位"""
    result, _, _ = generator.generate(
        phantom_volume, (1.0, 1.0, 1.0), {"motion_type": "respiratory", "amplitude_mm": 15.0}
    )
    # 中间切片的球心应偏移
    center_orig = np.unravel_index(np.argmax(phantom_volume[32]), (64, 64))
    center_moved = np.unravel_index(np.argmax(result[32]), (64, 64))
    # 至少有一些偏移
    offset = abs(center_orig[0] - center_moved[0]) + abs(center_orig[1] - center_moved[1])
    assert offset >= 0


def test_save_slice_png(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol = np.zeros((64, 64, 64), dtype=np.float32)
    z, y, x = np.ogrid[:64, :64, :64]
    sphere = ((z - 32) ** 2 + (y - 32) ** 2 + (x - 32) ** 2) <= 10 ** 2
    vol[sphere] = 1000.0

    gen = MotionArtifactGenerator(seed=42)
    r_resp, _, _ = gen.generate(vol, (1.0, 1.0, 1.0), {"motion_type": "respiratory", "amplitude_mm": 12.0})
    r_rand, _, _ = gen.generate(vol, (1.0, 1.0, 1.0), {"motion_type": "random", "amplitude_mm": 8.0})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(vol[32], cmap="gray", vmin=-200, vmax=1200)
    axes[0].set_title("Original")
    axes[1].imshow(r_resp[32], cmap="gray", vmin=-200, vmax=1200)
    axes[1].set_title("Respiratory motion")
    axes[2].imshow(r_rand[32], cmap="gray", vmin=-200, vmax=1200)
    axes[2].set_title("Random motion")
    for ax in axes:
        ax.axis("off")

    out_path = tmp_path / "motion_artifact_slices.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert out_path.exists()
    print(f"✅ 截图已保存: {out_path}")
