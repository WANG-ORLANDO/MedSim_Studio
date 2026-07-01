"""StreakArtifactGenerator + BeamHardeningGenerator + Composite 测试"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend"))

from app.artifact.generator.streak_artifact import StreakArtifactGenerator
from app.artifact.generator.beam_hardening import BeamHardeningGenerator
from app.artifact.generator.composite import CompositeArtifactGenerator
from app.artifact.generator import get_generator, list_artifact_types, ARTIFACT_GENERATORS


class TestStreakArtifact:
    def test_basic(self):
        gen = StreakArtifactGenerator(seed=42)
        vol = np.ones((32, 32, 32), dtype=np.float32) * 40
        result, mask, meta = gen.generate(vol, (1.0, 1.0, 1.0), gen.get_default_params())
        assert result.shape == vol.shape
        assert meta["artifact_type"] == "streakartifact"

    def test_cone_beam(self):
        gen = StreakArtifactGenerator(seed=0)
        vol = np.ones((16, 16, 16), dtype=np.float32) * 40
        result, _, meta = gen.generate(vol, (1.0, 1.0, 1.0), {"cone_beam": True, "cone_strength": 0.5})
        assert meta["cone_beam"] is True

    def test_validate(self):
        gen = StreakArtifactGenerator()
        assert gen.validate_params({}) is True
        assert gen.validate_params({"num_streaks": 0}) is False


class TestBeamHardening:
    def test_cupping(self):
        vol = np.ones((1, 64, 64), dtype=np.float32) * 40
        y, x = np.ogrid[:64, :64]
        sphere_mask = ((y - 32) ** 2 + (x - 32) ** 2) < 25 ** 2
        vol[0, sphere_mask] = 500.0
        gen = BeamHardeningGenerator(seed=0)
        result, mask, meta = gen.generate(vol, (1.0, 1.0, 1.0), {"cupping_strength": 0.8})
        # 球体中心应比球体边缘暗（杯状效应）
        center_val = result[0, 32, 32]
        edge_val = result[0, 32, 50]  # 球体边缘附近
        assert center_val < 500.0  # 中心 HU 被降低
        assert meta["artifact_type"] == "beamhardening"

    def test_validate(self):
        gen = BeamHardeningGenerator()
        assert gen.validate_params({"cupping_strength": 0.5}) is True
        assert gen.validate_params({"cupping_strength": 1.5}) is False

    def test_no_nan(self):
        gen = BeamHardeningGenerator(seed=0)
        vol = np.ones((8, 16, 16), dtype=np.float32) * 40
        result, _, _ = gen.generate(vol, (1.0, 1.0, 1.0), {})
        assert not np.any(np.isnan(result))


class TestComposite:
    def test_two_artifacts(self):
        gen = CompositeArtifactGenerator(seed=42)
        gen.add_artifact("noise", {"mAs": 100})
        gen.add_artifact("ring", {"num_rings": 3, "intensity": 30})
        vol = np.ones((16, 16, 16), dtype=np.float32) * 40
        result, mask, meta = gen.generate(vol, (1.0, 1.0, 1.0), {})
        assert result.shape == vol.shape
        assert meta["num_artifacts"] == 2
        assert "noise" in meta["individual_metadata"]
        assert "ring" in meta["individual_metadata"]

    def test_empty_composite(self):
        gen = CompositeArtifactGenerator(seed=0)
        vol = np.ones((8, 8, 8), dtype=np.float32) * 40
        result, mask, meta = gen.generate(vol, (1.0, 1.0, 1.0), {})
        np.testing.assert_array_equal(result, vol)
        assert meta["num_artifacts"] == 0

    def test_chaining(self):
        gen = CompositeArtifactGenerator(seed=0)
        returned = gen.add_artifact("noise", {}).add_artifact("motion", {})
        assert returned is gen
        assert len(gen._generators) == 2


class TestRegistry:
    def test_list_types(self):
        types = list_artifact_types()
        assert "metal" in types
        assert "motion" in types
        assert "noise" in types
        assert "ring" in types
        assert "streak" in types
        assert "beam_hardening" in types
        assert len(types) == 7  # 6 basic + composite

    def test_get_generator(self):
        gen = get_generator("metal")
        assert gen.__class__.__name__ == "MetalArtifactGenerator"

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown artifact type"):
            get_generator("nonexistent")

    def test_all_generators_instantiate(self):
        for name in list_artifact_types():
            gen = get_generator(name)
            assert hasattr(gen, "generate")
            assert hasattr(gen, "get_default_params")
            assert hasattr(gen, "validate_params")
