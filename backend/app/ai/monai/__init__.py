"""MONAI integration for medical image AI processing.

Provides:
- SegmentationModelManager: model loading, caching, and lifecycle
- run_segmentation(): full inference pipeline (preprocess → model → postprocess)
- run_lesion_detection(): lesion-specific detection
"""

from app.ai.monai.model_loader import SegmentationModelManager, ModelNotAvailableError, ORGAN_LABEL_MAP
from app.ai.monai.inference import run_segmentation, run_lesion_detection

__all__ = [
    "SegmentationModelManager",
    "ModelNotAvailableError",
    "ORGAN_LABEL_MAP",
    "run_segmentation",
    "run_lesion_detection",
]
