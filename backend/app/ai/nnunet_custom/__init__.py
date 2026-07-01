"""Custom nnUNet model inference — nnUNet_handoff (6-class organ segmentation).

Exposes:
  - run_nnunet_custom(): main inference entry point
  - is_available(): check that the model folder exists
  - CustomModelNotAvailableError
"""

from app.ai.nnunet_custom.segmenter import (
    run_nnunet_custom,
    is_available,
    CustomModelNotAvailableError,
)

__all__ = [
    "run_nnunet_custom",
    "is_available",
    "CustomModelNotAvailableError",
]
