"""BaseArtifactGenerator 单元测试"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend"))
from app.artifact.generator.base import BaseArtifactGenerator


def test_base_class_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseArtifactGenerator()


def test_concrete_subclass_instantiation():
    class DummyGenerator(BaseArtifactGenerator):
        def generate(self, volume, spacing, params):
            return volume, np.zeros_like(volume), {"type": "dummy"}

        def get_default_params(self):
            return {"intensity": 100}

        def validate_params(self, params):
            return "intensity" in params

    gen = DummyGenerator(seed=42)
    assert gen.rng is not None
    assert gen.get_artifact_type() == "dummy"


def test_get_artifact_type_strips_generator_suffix():
    class MetalArtifactGenerator(BaseArtifactGenerator):
        def generate(self, volume, spacing, params):
            return volume, np.zeros_like(volume), {}

        def get_default_params(self):
            return {}

        def validate_params(self, params):
            return True

    gen = MetalArtifactGenerator()
    assert gen.get_artifact_type() == "metalartifact"


def test_generate_returns_correct_shapes():
    class DummyGenerator(BaseArtifactGenerator):
        def generate(self, volume, spacing, params):
            mask = np.ones_like(volume, dtype=np.float32)
            return volume, mask, {"shape": list(volume.shape)}

        def get_default_params(self):
            return {}

        def validate_params(self, params):
            return True

    vol = np.random.randn(10, 20, 30).astype(np.float32)
    gen = DummyGenerator()
    artifact_vol, mask, meta = gen.generate(vol, (1.0, 1.0, 1.0), {})

    assert artifact_vol.shape == vol.shape
    assert mask.shape == vol.shape
    assert meta["shape"] == [10, 20, 30]


def test_seed_reproducibility():
    class RngGenerator(BaseArtifactGenerator):
        def generate(self, volume, spacing, params):
            noise = self.rng.standard_normal(volume.shape).astype(np.float32)
            return volume + noise, np.zeros_like(volume), {}

        def get_default_params(self):
            return {}

        def validate_params(self, params):
            return True

    vol = np.zeros((5, 5, 5), dtype=np.float32)
    spacing = (1.0, 1.0, 1.0)

    g1 = RngGenerator(seed=123)
    v1, _, _ = g1.generate(vol, spacing, {})

    g2 = RngGenerator(seed=123)
    v2, _, _ = g2.generate(vol, spacing, {})

    np.testing.assert_array_equal(v1, v2)


def test_validate_params_returns_bool():
    class DummyGenerator(BaseArtifactGenerator):
        def generate(self, volume, spacing, params):
            return volume, np.zeros_like(volume), {}

        def get_default_params(self):
            return {"level": 50}

        def validate_params(self, params):
            return isinstance(params.get("level"), (int, float))

    gen = DummyGenerator()
    assert gen.validate_params({"level": 50}) is True
    assert gen.validate_params({"level": "bad"}) is False
