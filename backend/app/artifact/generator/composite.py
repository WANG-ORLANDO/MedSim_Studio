"""组合伪影生成器 — 同时施加多种伪影"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from .base import BaseArtifactGenerator


class CompositeArtifactGenerator(BaseArtifactGenerator):
    """组合生成器：按顺序叠加多种伪影"""

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self._generators: List[Tuple[str, BaseArtifactGenerator, Dict[str, Any]]] = []

    def add_artifact(self, artifact_type: str, params: Optional[Dict[str, Any]] = None):
        from . import get_generator
        gen = get_generator(artifact_type)
        gen.rng = np.random.default_rng(self.rng.integers(0, 2**31))
        self._generators.append((artifact_type, gen, params or {}))
        return self

    def get_default_params(self) -> Dict[str, Any]:
        return {"artifacts": []}

    def validate_params(self, params: Dict[str, Any]) -> bool:
        return True

    def _build_generators_from_params(self, artifacts: list):
        from . import get_generator
        for item in artifacts:
            art_type = item.get("type", "")
            art_params = item.get("params", {})
            if art_type not in ("composite",):
                gen = get_generator(art_type)
                gen.rng = np.random.default_rng(self.rng.integers(0, 2**31))
                self._generators.append((art_type, gen, art_params))

    def generate(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float],
        params: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        if not self._generators and "artifacts" in params:
            self._build_generators_from_params(params["artifacts"])

        if not self._generators:
            return volume.copy().astype(np.float32), np.zeros_like(volume, dtype=np.float32), {"artifact_type": "composite", "applied_artifacts": [], "num_artifacts": 0}

        current = volume.copy().astype(np.float32)
        individual_masks = {}
        individual_metadata = {}

        for artifact_type, gen, art_params in self._generators:
            current, mask, meta = gen.generate(current, spacing, art_params)
            individual_masks[artifact_type] = mask
            individual_metadata[artifact_type] = meta

        combined_mask = np.zeros_like(volume, dtype=np.float32)
        for mask in individual_masks.values():
            combined_mask = np.maximum(combined_mask, mask)

        def _to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_serializable(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        serializable_metadata = _to_serializable(individual_metadata)

        metadata = {
            "artifact_type": "composite",
            "applied_artifacts": [a[0] for a in self._generators],
            "individual_metadata": serializable_metadata,
            "num_artifacts": len(self._generators),
        }

        return np.clip(current, -1024, 3071).astype(np.float32), combined_mask, metadata
