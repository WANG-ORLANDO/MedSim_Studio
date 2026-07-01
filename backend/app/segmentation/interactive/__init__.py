"""Interactive segmentation refinement tools.

Provides click-based region growing for adding/removing
segmentation labels with intensity-constrained flood fill.
"""

from app.segmentation.interactive.refiner import (
    refine_mask_on_click,
    extract_slice_mask,
)

__all__ = [
    "refine_mask_on_click",
    "extract_slice_mask",
]
