"""AI-based segmentation pipeline.

Provides the orchestration layer that combines model loading,
inference, and postprocessing into a single segmentation workflow.

Supported backends:
  - TotalSegmentator (pretrained, 117 structures, recommended)
  - nnUNet custom (Dataset701_TotalSegOrgans6, 6 organs)
  - MONAI UNet/SegResNet/SwinUNETR (random weights without training)
"""

from app.segmentation.ai.pipeline import (
    run_full_segmentation,
    get_available_organs,
)

# TotalSegmentator integration
from app.ai.totalsegmentator import (
    run_totalsegmentator,
    is_available as totalsegmentator_available,
    TotalSegmentatorNotAvailableError,
)

# Custom nnUNet integration
from app.ai.nnunet_custom import (
    run_nnunet_custom,
    is_available as nnunet_custom_available,
    CustomModelNotAvailableError,
)

__all__ = [
    "run_full_segmentation",
    "get_available_organs",
    "run_totalsegmentator",
    "totalsegmentator_available",
    "TotalSegmentatorNotAvailableError",
    "run_nnunet_custom",
    "nnunet_custom_available",
    "CustomModelNotAvailableError",
]
