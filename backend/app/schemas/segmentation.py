"""
Segmentation Pydantic Schemas

Request/response schemas for AI segmentation APIs.
"""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List


class SegmentationJobCreate(BaseModel):
    """Schema for creating a segmentation job."""
    study_id: str = Field(..., description="Study ID to segment")
    series_id: str = Field(..., description="Series ID to segment")
    model_name: str = Field("unet", description="Model name: unet, segresnet, swin_unetr")
    target_organs: List[str] = Field(
        default_factory=list,
        description="Organs to segment; empty = all available",
    )
    detect_lesions: bool = Field(False, description="Enable lesion detection")


class LabelDef(BaseModel):
    """A single segmentation label definition."""
    index: int
    name: str
    color: List[int]  # RGB [r, g, b]


class SegmentationJobResponse(BaseModel):
    """Response schema for a segmentation job."""
    id: str
    study_id: str
    series_id: str
    status: str = "pending"
    model_name: str = "unet"
    target_organs: List[str] = []
    detect_lesions: bool = False
    progress: float = 0.0
    error_message: Optional[str] = None
    mask_path: Optional[str] = None
    label_map_path: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SliceMaskResponse(BaseModel):
    """Response schema for a single 2D slice of the segmentation mask."""
    z_index: int
    rows: int
    cols: int
    labels: List[LabelDef] = []
    mask_data: List[List[int]] = []  # 2D array of label indices (rows of cols)


class InteractiveClickRequest(BaseModel):
    """Request schema for interactive segmentation refinement."""
    job_id: str = Field(..., description="Segmentation job ID")
    z_index: int = Field(..., description="Slice index (z-axis)")
    x: int = Field(..., description="X voxel coordinate of click")
    y: int = Field(..., description="Y voxel coordinate of click")
    label: int = Field(1, description="Label index to assign")
    operation: str = Field("add", description="Operation: add or remove")


class InteractiveClickResponse(BaseModel):
    """Response after an interactive refinement click."""
    z_index: int
    updated_rows: int = 0
    updated_cols: int = 0
    patch_data: List[List[int]] = []  # Updated local mask patch around click


class ModelInfoResponse(BaseModel):
    """Information about an available segmentation model."""
    name: str
    description: str
    organs: List[str] = []
    status: str = "available"  # available, coming_soon
