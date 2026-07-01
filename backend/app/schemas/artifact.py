"""Artifact Pydantic Schema — 伪影生成/分类/修复请求与响应"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ArtifactGenerateRequest(BaseModel):
    """伪影生成请求"""
    source: Literal["atlas", "procedural", "dicom"] = "atlas"
    case_id: Optional[str] = None
    series_id: Optional[str] = None
    size: int = Field(160, ge=64, le=192)
    artifact_types: List[str]
    artifact_params: Dict[str, Dict[str, Any]] = {}
    ct_params: Optional[Dict[str, Any]] = None


class ArtifactGenerateResponse(BaseModel):
    """伪影生成响应"""
    artifact_volume_base64: str
    clean_volume_base64: str
    artifact_masks_base64: Dict[str, str]
    metadata: Dict[str, Any]
    standardized_case: Dict[str, Any]


class ArtifactClassifyRequest(BaseModel):
    """伪影分类请求"""
    source: str = "phantom"
    series_id: Optional[str] = None
    slice_indices: Optional[List[int]] = None


class SliceClassifyResult(BaseModel):
    scores: Dict[str, float]
    labels: List[str]
    dominant: str
    slice_index: int


class ArtifactClassifyResponse(BaseModel):
    """伪影分类响应"""
    overall_scores: Dict[str, float]
    per_slice_scores: List[SliceClassifyResult]
    dominant_artifact: str
    slice_count: int


class ArtifactRestoreRequest(BaseModel):
    """伪影修复请求"""
    source: str = "phantom"
    series_id: Optional[str] = None
    artifact_types: List[str] = Field(default_factory=list, description="已知伪影类型列表")
    method: Literal["traditional", "deep_learning", "auto"] = "auto"
    shape: Optional[List[int]] = None
    spacing: Optional[List[float]] = None


class ArtifactRestoreResponse(BaseModel):
    """伪影修复响应"""
    restored_slice: List[List[float]]
    original_slice: List[List[float]]
    quality_metrics: Dict[str, float]
    steps: List[Dict[str, Any]]
    report: str
    shape: List[int]
    spacing: List[float]


class ArtifactPipelineRequest(BaseModel):
    """完整流水线: 生成 → 分类 → 修复"""
    source: str = "phantom"
    series_id: Optional[str] = None
    artifact_type: str = "noise"
    params: Dict[str, Any] = Field(default_factory=dict)
    slice_index: Optional[int] = None


class ArtifactPipelineResponse(BaseModel):
    """完整流水线响应"""
    artifact_type: str
    classification: Dict[str, Any]
    restoration: Dict[str, Any]
    original_slice: List[List[float]]
    artifact_slice: List[List[float]]
    restored_slice: List[List[float]]
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any]


class ArtifactJobResponse(BaseModel):
    """作业响应"""
    id: str
    job_type: str
    status: str
    config: Optional[Dict[str, Any]]
    output_path: Optional[str]
    quality_metrics: Optional[Dict[str, Any]]
    created_at: Optional[str]
    completed_at: Optional[str]


class ArtifactTypesResponse(BaseModel):
    types: List[str]


class ArtifactGenerateSliceRequest(BaseModel):
    """单切片生成请求 (兼容现有前端)"""
    artifact_type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    source: str = "phantom"
    series_id: Optional[str] = None
    study_id: Optional[str] = None
    slice_index: Optional[int] = None


class ArtifactGenerateSliceResponse(BaseModel):
    """单切片生成响应 (兼容现有前端)"""
    artifact_type: str
    original_slice: List[List[float]]
    artifact_slice: List[List[float]]
    mask_slice: List[List[float]]
    metadata: Dict[str, Any]
    shape: List[int]
    spacing: List[float]
    source: str
