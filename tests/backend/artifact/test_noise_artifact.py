"""NoiseArtifactGenerator 单元测试"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend"))

from app.artifact.generator.noise_artifact import (
    NoiseArtifactGenerator,
    hu_to_attenuation,
    attenuation_to_hu,
    apply_quantum_noise,
)


class TestHUConversion:
    def test_roundtrip(self):
        hu = np.array([-1000, 0, 40, 100, 300, 1000, 3000], dtype=np.float32)
        mu = hu_to_attenuation(hu)
        hu_back = attenuation_to_hu(mu)
        np.testing.assert_allclose(hu, hu_back, atol=1e-4)

    def test_water_is_zero_hu(self):
        mu_water = hu_to_attenuation(np.array([0.0]))
        assert mu_water[0] == pytest.approx(0.2)

    def test_air_is_neg1000_hu(self):
        mu_air = hu_to_attenuation(np.array([-1000.0]))
        assert mu_air[0] == pytest.approx(0.0)


class TestQuantumNoise:
    def test_noise_increases_with_lower_mas(self):
        vol = np.ones((32, 32, 32), dtype=np.float32) * 40
        rng = np.random.default_rng(42)

        r_high = apply_quantum_noise(vol, mAs=150, rng=rng)
        r_low = apply_quantum_noise(vol, mAs=30, rng=rng)

        std_high = np.std(r_high - vol)
        std_low = np.std(r_low - vol)
        assert std_low > std_high

    def test_noise_std_inversely_proportional_to_sqrt_mas(self):
        vol = np.ones((32, 32, 32), dtype=np.float32) * 40

        stds = []
        mas_values = [50, 100, 200, 400]
        for mas in mas_values:
            rng = np.random.default_rng(0)
            noisy = apply_quantum_noise(vol, mAs=mas, rng=rng)
            stds.append(np.std(noisy - vol))

        # σ ∝ 1/√mAs -> σ * √mAs should be roughly constant
        products = [s * np.sqrt(m) for s, m in zip(stds, mas_values)]
        for i in range(1, len(products)):
            ratio = products[i] / products[0]
            assert 0.5 < ratio < 2.0, f"σ*√mAs not stable: {products}"

    def test_output_shape_and_dtype(self):
        vol = np.ones((16, 16, 16), dtype=np.float32) * 40
        rng = np.random.default_rng(0)
        result = apply_quantum_noise(vol, mAs=100, rng=rng)
        assert result.shape == vol.shape
        assert result.dtype == np.float32


class TestNoiseArtifactGenerator:
    def test_basic_generation(self):
        gen = NoiseArtifactGenerator(seed=42)
        vol = np.ones((32, 32, 32), dtype=np.float32) * 40
        result, mask, meta = gen.generate(vol, (1.0, 1.0, 1.0), gen.get_default_params())

        assert result.shape == vol.shape
        assert mask.shape == vol.shape
        assert mask.dtype == np.float32
        assert np.all(mask == 1)
        assert meta["artifact_type"] == "noiseartifact"
        assert meta["noise_std"] > 0

    def test_low_mas_more_noise(self):
        vol = np.ones((32, 32, 32), dtype=np.float32) * 40

        gen_h = NoiseArtifactGenerator(seed=0)
        r_h, _, m_h = gen_h.generate(vol, (1.0, 1.0, 1.0), {"mAs": 200})

        gen_l = NoiseArtifactGenerator(seed=0)
        r_l, _, m_l = gen_l.generate(vol, (1.0, 1.0, 1.0), {"mAs": 30})

        assert m_l["noise_std"] > m_h["noise_std"]

    def test_electronic_noise(self):
        gen = NoiseArtifactGenerator(seed=42)
        vol = np.ones((32, 32, 32), dtype=np.float32) * 40

        r_no_elec, _, _ = gen.generate(vol, (1.0, 1.0, 1.0), {"mAs": 100, "electronic_noise_sigma": 0})
        r_with_elec, _, _ = gen.generate(vol, (1.0, 1.0, 1.0), {"mAs": 100, "electronic_noise_sigma": 10})

        assert np.std(r_with_elec - vol) > np.std(r_no_elec - vol)

    def test_validate_params(self):
        gen = NoiseArtifactGenerator()
        assert gen.validate_params({"mAs": 100}) is True
        assert gen.validate_params({}) is False
        assert gen.validate_params({"mAs": -10}) is False

    def test_seed_reproducibility(self):
        vol = np.ones((32, 32, 32), dtype=np.float32) * 40
        params = {"mAs": 80}

        r1, _, _ = NoiseArtifactGenerator(seed=77).generate(vol, (1.0, 1.0, 1.0), params)
        r2, _, _ = NoiseArtifactGenerator(seed=77).generate(vol, (1.0, 1.0, 1.0), params)
        np.testing.assert_array_equal(r1, r2)

    def test_clipping(self):
        gen = NoiseArtifactGenerator(seed=42)
        vol = np.ones((16, 16, 16), dtype=np.float32) * 40
        result, _, _ = gen.generate(vol, (1.0, 1.0, 1.0), {"mAs": 10})
        assert np.min(result) >= -1024
        assert np.max(result) <= 3071


def test_save_slice_png(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen_h = NoiseArtifactGenerator(seed=42)
    gen_l = NoiseArtifactGenerator(seed=42)
    vol = np.ones((64, 64, 64), dtype=np.float32) * 40

    r_high, _, _ = gen_h.generate(vol, (1.0, 1.0, 1.0), {"mAs": 200})
    r_low, _, _ = gen_l.generate(vol, (1.0, 1.0, 1.0), {"mAs": 30})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(vol[32], cmap="gray", vmin=-200, vmax=3000)
    axes[0].set_title("Original (40 HU)")
    axes[1].imshow(r_high[32], cmap="gray", vmin=-200, vmax=3000)
    axes[1].set_title("High dose (mAs=200)")
    axes[2].imshow(r_low[32], cmap="gray", vmin=-200, vmax=3000)
    axes[2].set_title("Low dose (mAs=30)")
    for ax in axes:
        ax.axis("off")

    out_path = tmp_path / "noise_artifact_slices.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert out_path.exists()
    print(f"✅ 截图已保存: {out_path}")
