"""TotalSegmentator integration for medical image segmentation.

Provides a pretrained deep learning segmentation model that can
identify 117 anatomical structures in CT volumes.

This module replaces MONAI-based segmentation as the default
segmentation engine, delivering production-quality results
without requiring any training.

Key components:
  - run_totalsegmentator(): Main inference function (numpy in → label map out)
  - TOTAL_SEGMENTATOR_LABEL_MAP: 117-class label name → index mapping
  - get_label_defs(): Build label definitions for API responses
  - is_available(): Check if TotalSegmentator is installed
"""

from app.ai.totalsegmentator.segmenter import (
    run_totalsegmentator,
    is_available,
    TotalSegmentatorNotAvailableError,
)
from app.ai.totalsegmentator.labels import (
    TOTAL_SEGMENTATOR_LABEL_MAP,
    TOTAL_SEGMENTATOR_COLORS,
    ORGAN_CATEGORIES,
    get_label_map,
    get_label_colors,
    get_label_defs,
)

__all__ = [
    "run_totalsegmentator",
    "is_available",
    "TotalSegmentatorNotAvailableError",
    "TOTAL_SEGMENTATOR_LABEL_MAP",
    "TOTAL_SEGMENTATOR_COLORS",
    "ORGAN_CATEGORIES",
    "get_label_map",
    "get_label_colors",
    "get_label_defs",
]
