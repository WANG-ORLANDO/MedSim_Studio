"""Custom nnUNet 20-class model inference — Dataset702_TotalSegOrgans20.

Exposes:
  - run_nnunet_custom_20(): main inference entry point
  - merge_to_6_classes(): optional 20→6 class merging for frontend compat
  - is_available(): check that the model folder exists
  - CustomModelNotAvailableError
"""

from app.ai.nnunet_custom_20.segmenter import (
    run_nnunet_custom_20,
    is_available,
    merge_to_6_classes,
    CustomModelNotAvailableError,
)

__all__ = [
    "run_nnunet_custom_20",
    "is_available",
    "merge_to_6_classes",
    "CustomModelNotAvailableError",
]
