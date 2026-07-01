"""
DICOM Pydantic Schemas

Request/response schemas for DICOM study and series management APIs.
"""

from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


class DicomStudyResponse(BaseModel):
    """Response schema for DICOM study metadata."""
    id: str
    patient_id: str
    patient_name: str
    patient_birth_date: Optional[str] = None
    patient_sex: Optional[str] = None
    study_instance_uid: str
    study_date: Optional[str] = None
    study_time: Optional[str] = None
    study_description: Optional[str] = None
    accession_number: Optional[str] = None
    referring_physician: Optional[str] = None
    modalities: List[str] = []
    series_count: int = 0
    instance_count: int = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DicomSeriesResponse(BaseModel):
    """Response schema for DICOM series metadata."""
    id: str
    study_id: str
    series_instance_uid: str
    series_number: Optional[int] = None
    series_description: Optional[str] = None
    modality: Optional[str] = None
    manufacturer: Optional[str] = None
    body_part_examined: Optional[str] = None
    laterality: Optional[str] = None
    protocol_name: Optional[str] = None
    image_count: int = 0
    series_date: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    slice_thickness: Optional[float] = None
    pixel_spacing: Optional[List[float]] = None
    window_center: Optional[float] = None
    window_width: Optional[float] = None

    class Config:
        from_attributes = True

    @field_validator("pixel_spacing", mode="before")
    @classmethod
    def clean_pixel_spacing(cls, v):
        if v is None:
            return None
        if isinstance(v, list):
            cleaned = [x for x in v if x is not None]
            return cleaned if cleaned else None
        return v


class DicomInstanceResponse(BaseModel):
    """Response schema for DICOM instance metadata."""
    id: str
    series_id: str
    sop_instance_uid: str
    instance_number: Optional[int] = None
    slice_location: Optional[float] = None
    rows: Optional[int] = None
    columns: Optional[int] = None

    class Config:
        from_attributes = True


class DicomUploadResponse(BaseModel):
    """Response schema for DICOM upload operation."""
    study_id: str
    series_count: int
    instance_count: int
    message: str = "Upload successful"


class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper."""
    items: List
    total: int
    page: int = 1
    page_size: int = 20
    total_pages: int = 0
