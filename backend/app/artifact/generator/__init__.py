"""伪影生成器注册表 — 统一管理所有生成器类型"""

from .metal_artifact import MetalArtifactGenerator
from .motion_artifact import MotionArtifactGenerator
from .noise_artifact import NoiseArtifactGenerator
from .ring_artifact import RingArtifactGenerator
from .streak_artifact import StreakArtifactGenerator
from .beam_hardening import BeamHardeningGenerator
from .composite import CompositeArtifactGenerator

ARTIFACT_GENERATORS = {
    "metal": MetalArtifactGenerator,
    "motion": MotionArtifactGenerator,
    "noise": NoiseArtifactGenerator,
    "ring": RingArtifactGenerator,
    "streak": StreakArtifactGenerator,
    "beam_hardening": BeamHardeningGenerator,
    "composite": CompositeArtifactGenerator,
}


def get_generator(artifact_type: str):
    """根据类型名获取生成器实例"""
    if artifact_type not in ARTIFACT_GENERATORS:
        raise ValueError(
            f"Unknown artifact type: {artifact_type}. "
            f"Available: {list(ARTIFACT_GENERATORS.keys())}"
        )
    return ARTIFACT_GENERATORS[artifact_type]()


def list_artifact_types():
    """列出所有可用伪影类型"""
    return list(ARTIFACT_GENERATORS.keys())
