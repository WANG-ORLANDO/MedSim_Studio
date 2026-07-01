"""Artifact API endpoints — 伪影生成 REST 接口"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import func
import numpy as np

from app.database.session import get_db
from app.models.dicom import DicomSeries, DicomInstance
from app.dicom.storage import get_storage_backend

router = APIRouter(prefix="/artifact", tags=["artifact"])

SLICE_WINDOW = 32


class ArtifactRequest(BaseModel):
    artifact_type: str = Field(..., description="伪影类型")
    params: Dict[str, Any] = Field(default_factory=dict, description="伪影参数")
    source: str = Field(default="phantom", description="数据源: phantom 或 dicom")
    series_id: Optional[str] = Field(default=None, description="DICOM series ID (source=dicom 时必填)")
    study_id: Optional[str] = Field(default=None, description="DICOM study ID")
    slice_index: Optional[int] = Field(default=None, description="返回的切片索引，默认中间切片")


class ArtifactResponse(BaseModel):
    artifact_type: str
    original_slice: List[List[float]]
    artifact_slice: List[List[float]]
    mask_slice: List[List[float]]
    metadata: Dict[str, Any]
    shape: List[int]
    spacing: List[float]
    source: str


class ArtifactTypesResponse(BaseModel):
    types: List[str]


class ClassifyRequest(BaseModel):
    source: str = Field(default="phantom", description="数据源: phantom 或 dicom")
    series_id: Optional[str] = Field(default=None, description="DICOM series ID (source=dicom 时必填)")
    slice_indices: Optional[List[int]] = Field(default=None, description="要分类的切片索引列表，None 则取中间 2/3")


class SliceClassifyResult(BaseModel):
    scores: Dict[str, float]
    labels: List[str]
    dominant: str
    slice_index: int


class ClassifyResponse(BaseModel):
    overall_scores: Dict[str, float]
    per_slice_scores: List[SliceClassifyResult]
    dominant_artifact: str
    slice_count: int


class SeriesInfo(BaseModel):
    id: str
    study_id: str
    description: Optional[str]
    modality: Optional[str]
    image_count: Optional[int]
    rows: Optional[int]
    columns: Optional[int]


@router.get("/types", response_model=ArtifactTypesResponse)
async def list_artifact_types():
    from app.artifact.generator import list_artifact_types
    return ArtifactTypesResponse(types=list_artifact_types())


@router.get("/series", response_model=List[SeriesInfo])
async def list_ct_series(
    study_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """列出所有 CT 模态的 Series"""
    query = db.query(DicomSeries).filter(func.upper(DicomSeries.modality) == "CT")
    if study_id:
        query = query.filter(DicomSeries.study_id == study_id)
    series_list = query.all()
    return [
        SeriesInfo(
            id=s.id,
            study_id=s.study_id,
            description=s.series_description,
            modality=s.modality,
            image_count=s.image_count,
            rows=s.rows,
            columns=s.columns,
        )
        for s in series_list
    ]


def _load_dicom_volume_slice(
    storage, instances, target_slice: int, window: int = SLICE_WINDOW,
):
    """只加载 target_slice 附近 window 张切片，返回 (volume, spacing, adjusted_idx)"""
    import io
    import pydicom as pdcm

    total = len(instances)
    half = window // 2
    start = max(0, target_slice - half)
    end = min(total, start + window)
    start = max(0, end - window)
    subset = instances[start:end]

    slices_data = []
    slice_thickness = None
    pixel_spacing = None

    for inst in subset:
        if not inst.pixel_data_path:
            continue
        dicom_bytes = storage.get_object_bytes(inst.pixel_data_path)
        if dicom_bytes is None:
            continue
        try:
            ds = pdcm.dcmread(io.BytesIO(dicom_bytes), force=True)
        except Exception:
            continue
        if not hasattr(ds, "pixel_array"):
            continue
        try:
            px = ds.pixel_array.astype(np.float32)
        except Exception:
            continue
        slope = getattr(ds, "RescaleSlope", None)
        intercept = getattr(ds, "RescaleIntercept", None)
        if slope is not None:
            px *= float(slope)
        if intercept is not None:
            px += float(intercept)
        slices_data.append(px)
        if slice_thickness is None:
            try:
                slice_thickness = float(getattr(ds, "SliceThickness", 1.0))
            except (TypeError, ValueError):
                pass
        if pixel_spacing is None:
            ps = getattr(ds, "PixelSpacing", None)
            if ps is not None:
                try:
                    pixel_spacing = (float(ps[0]), float(ps[1]))
                except (IndexError, TypeError, ValueError):
                    pass

    if not slices_data:
        raise ValueError("No valid slices in the selected window")

    volume = np.stack(slices_data, axis=0)
    z_sp = slice_thickness or 1.0
    y_sp = pixel_spacing[0] if pixel_spacing else 1.0
    x_sp = pixel_spacing[1] if pixel_spacing else 1.0
    spacing = (z_sp, y_sp, x_sp)
    adjusted_idx = target_slice - start
    return volume, spacing, adjusted_idx


@router.post("/generate", response_model=ArtifactResponse)
async def generate_artifact(request: ArtifactRequest, db: Session = Depends(get_db)):
    from app.artifact.generator import get_generator

    try:
        gen = get_generator(request.artifact_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # ---- 加载体积数据 ----
    if request.source == "dicom":
        if not request.series_id:
            raise HTTPException(status_code=400, detail="series_id is required when source='dicom'")

        series = db.query(DicomSeries).filter(DicomSeries.id == request.series_id).first()
        if not series:
            raise HTTPException(status_code=404, detail=f"Series {request.series_id} not found")

        instances = (
            db.query(DicomInstance)
            .filter(DicomInstance.series_id == series.id)
            .order_by(DicomInstance.instance_number.asc().nulls_last())
            .all()
        )
        if not instances:
            raise HTTPException(status_code=404, detail=f"No instances for series {series.id}")

        total_slices = len(instances)
        target = request.slice_index if request.slice_index is not None else total_slices // 2
        target = max(0, min(target, total_slices - 1))

        storage = get_storage_backend()
        try:
            volume, spacing_tup, adjusted_idx = _load_dicom_volume_slice(
                storage, instances, target, SLICE_WINDOW,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Volume build failed: {str(e)[:300]}")

        spacing = list(spacing_tup)
        slice_idx = adjusted_idx
    else:
        shape = [64, 64, 64]
        nz, ny, nx = shape
        volume = np.full(shape, 40.0, dtype=np.float32)
        cz, cy, cx = nz // 2, ny // 2, nx // 2
        radius = min(nz, ny, nx) // 4
        z, y, x = np.ogrid[:nz, :ny, :nx]
        sphere = ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) <= radius ** 2
        volume[sphere] = 400.0
        spacing = [1.0, 1.0, 1.0]
        slice_idx = volume.shape[0] // 2

    # ---- 生成伪影 ----
    try:
        artifact_vol, mask, metadata = gen.generate(volume, tuple(spacing), request.params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # ---- 选择返回的切片 ----
    nz = volume.shape[0]
    slice_idx = max(0, min(slice_idx, nz - 1))

    original_slice = volume[slice_idx].tolist()
    artifact_slice = artifact_vol[slice_idx].tolist()
    mask_slice = mask[slice_idx].tolist()

    return ArtifactResponse(
        artifact_type=request.artifact_type,
        original_slice=original_slice,
        artifact_slice=artifact_slice,
        mask_slice=mask_slice,
        metadata=metadata,
        shape=list(volume.shape),
        spacing=spacing,
        source=request.source,
    )


@router.post("/classify", response_model=ClassifyResponse)
async def classify_artifact(request: ClassifyRequest, db: Session = Depends(get_db)):
    """对 CT 体积进行伪影分类"""
    import os
    from app.artifact.classifier.inference import ArtifactInference
    from app.artifact.classifier.dataset import CLASS_NAMES

    # 查找模型权重
    model_path = os.environ.get("CLASSIFIER_MODEL_PATH", "/tmp/clf_full/best_model.pth")
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Classifier model not found at {model_path}. Please train the model first.",
        )

    # 加载体积数据
    if request.source == "dicom":
        if not request.series_id:
            raise HTTPException(status_code=400, detail="series_id is required when source='dicom'")

        series = db.query(DicomSeries).filter(DicomSeries.id == request.series_id).first()
        if not series:
            raise HTTPException(status_code=404, detail=f"Series {request.series_id} not found")

        instances = (
            db.query(DicomInstance)
            .filter(DicomInstance.series_id == series.id)
            .order_by(DicomInstance.instance_number.asc().nulls_last())
            .all()
        )
        if not instances:
            raise HTTPException(status_code=404, detail=f"No instances for series {series.id}")

        storage = get_storage_backend()

        # 加载全部切片构建体积
        import io
        import pydicom as pdcm

        slices_data = []
        for inst in instances:
            if not inst.pixel_data_path:
                continue
            dicom_bytes = storage.get_object_bytes(inst.pixel_data_path)
            if dicom_bytes is None:
                continue
            try:
                ds = pdcm.dcmread(io.BytesIO(dicom_bytes), force=True)
            except Exception:
                continue
            if not hasattr(ds, "pixel_array"):
                continue
            try:
                px = ds.pixel_array.astype(np.float32)
            except Exception:
                continue
            slope = getattr(ds, "RescaleSlope", None)
            intercept = getattr(ds, "RescaleIntercept", None)
            if slope is not None:
                px *= float(slope)
            if intercept is not None:
                px += float(intercept)
            slices_data.append(px)

        if not slices_data:
            raise HTTPException(status_code=500, detail="No valid slices found")

        volume = np.stack(slices_data, axis=0)
    else:
        # phantom 模式: 生成简单体积
        shape = [64, 64, 64]
        nz, ny, nx = shape
        volume = np.full(shape, 40.0, dtype=np.float32)
        cz, cy, cx = nz // 2, ny // 2, nx // 2
        radius = min(nz, ny, nx) // 4
        z, y, x = np.ogrid[:nz, :ny, :nx]
        sphere = ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) <= radius ** 2
        volume[sphere] = 400.0

    # 执行分类
    try:
        classifier = ArtifactInference(model_path, device="cpu")
        result = classifier.predict_volume(volume, slice_indices=request.slice_indices)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

    return ClassifyResponse(
        overall_scores=result["overall_scores"],
        per_slice_scores=[
            SliceClassifyResult(
                scores=s["scores"],
                labels=s["labels"],
                dominant=s["dominant"],
                slice_index=s["slice_index"],
            )
            for s in result["per_slice_scores"]
        ],
        dominant_artifact=result["dominant_artifact"],
        slice_count=result["slice_count"],
    )


# ============================================================
# Training API
# ============================================================

import threading
import time as _time
import json as _json
import os as _os

# Global training state
_training_state = {
    "status": "idle",  # idle, training, completed, failed
    "current_epoch": 0,
    "total_epochs": 0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "train_acc": 0.0,
    "val_acc": 0.0,
    "train_f1": 0.0,
    "val_f1": 0.0,
    "best_val_loss": float("inf"),
    "epoch_history": [],
    "error": None,
    "start_time": None,
    "output_dir": None,
}
_training_lock = threading.Lock()


class TrainRequest(BaseModel):
    epochs: int = Field(default=10, ge=1, le=200, description="训练轮数")
    batch_size: int = Field(default=32, ge=4, le=128, description="批大小")
    learning_rate: float = Field(default=1e-4, gt=0, le=0.1, description="学习率")
    num_volumes: int = Field(default=20, ge=5, le=200, description="每类生成的体积数")
    output_dir: str = Field(default="/app/models/artifact_classifier", description="模型保存目录")


class TrainStatusResponse(BaseModel):
    status: str
    current_epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    train_f1: float
    val_f1: float
    best_val_loss: float
    epoch_history: List[Dict[str, Any]]
    error: Optional[str]
    start_time: Optional[float]


class TrainHistoryResponse(BaseModel):
    epochs: List[Dict[str, Any]]
    best_val_loss: float
    output_dir: str


def _run_training_background(req: TrainRequest):
    """后台训练线程"""
    global _training_state

    try:
        from app.artifact.classifier.train import train as train_fn, TRAIN_CONFIG

        config = TRAIN_CONFIG.copy()
        config["epochs"] = req.epochs
        config["batch_size"] = req.batch_size
        config["learning_rate"] = req.learning_rate

        with _training_lock:
            _training_state["status"] = "training"
            _training_state["start_time"] = _time.time()
            _training_state["output_dir"] = req.output_dir
            _training_state["total_epochs"] = req.epochs

        # 自定义回调: 每个 epoch 更新状态
        original_log = None

        # 执行训练（同步阻塞，但在后台线程中）
        history = train_fn(
            config=config,
            output_dir=req.output_dir,
            num_volumes_per_class=req.num_volumes,
            pretrained=True,
        )

        # 提取训练历史
        with _training_lock:
            epoch_history = []
            for i, (train_m, val_m) in enumerate(zip(history["train"], history["val"])):
                epoch_history.append({
                    "epoch": i + 1,
                    "train_loss": train_m["loss"],
                    "val_loss": val_m["loss"],
                    "train_acc": train_m["accuracy"],
                    "val_acc": val_m["accuracy"],
                    "train_f1": train_m["macro_f1"],
                    "val_f1": val_m["macro_f1"],
                })
            _training_state["epoch_history"] = epoch_history
            _training_state["status"] = "completed"
            _training_state["current_epoch"] = len(epoch_history)

    except Exception as e:
        with _training_lock:
            _training_state["status"] = "failed"
            _training_state["error"] = str(e)


@router.post("/train")
async def start_training(request: TrainRequest):
    """启动模型训练（后台异步）"""
    with _training_lock:
        if _training_state["status"] == "training":
            raise HTTPException(status_code=409, detail="Training already in progress")

        # 重置状态
        _training_state.update({
            "status": "starting",
            "current_epoch": 0,
            "total_epochs": request.epochs,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "train_acc": 0.0,
            "val_acc": 0.0,
            "train_f1": 0.0,
            "val_f1": 0.0,
            "best_val_loss": float("inf"),
            "epoch_history": [],
            "error": None,
            "start_time": _time.time(),
            "output_dir": request.output_dir,
        })

    # 启动后台训练线程
    thread = threading.Thread(target=_run_training_background, args=(request,), daemon=True)
    thread.start()

    return {"message": "Training started", "epochs": request.epochs, "output_dir": request.output_dir}


@router.get("/train/status", response_model=TrainStatusResponse)
async def get_training_status():
    """查询当前训练状态"""
    with _training_lock:
        state = _training_state.copy()

    # 更新当前 epoch 信息
    if state["status"] == "training" and state["epoch_history"]:
        last = state["epoch_history"][-1]
        state["current_epoch"] = last["epoch"]
        state["train_loss"] = last["train_loss"]
        state["val_loss"] = last["val_loss"]
        state["train_acc"] = last["train_acc"]
        state["val_acc"] = last["val_acc"]
        state["train_f1"] = last["train_f1"]
        state["val_f1"] = last["val_f1"]

    # 计算 best_val_loss
    if state["epoch_history"]:
        val_losses = [e["val_loss"] for e in state["epoch_history"]]
        state["best_val_loss"] = min(val_losses) if val_losses else float("inf")

    return TrainStatusResponse(**state)


@router.get("/train/history")
async def get_training_history():
    """获取已完成的训练历史"""
    output_dir = _training_state.get("output_dir", "/app/models/artifact_classifier")
    history_path = _os.path.join(output_dir, "training_history.json")

    if not _os.path.exists(history_path):
        return {"epochs": [], "best_val_loss": float("inf"), "output_dir": output_dir}

    try:
        with open(history_path, "r") as f:
            history = _json.load(f)

        epochs = []
        for i, (train_m, val_m) in enumerate(zip(history.get("train", []), history.get("val", []))):
            epochs.append({
                "epoch": i + 1,
                "train_loss": train_m.get("loss", 0),
                "val_loss": val_m.get("loss", 0),
                "train_acc": train_m.get("accuracy", 0),
                "val_acc": val_m.get("accuracy", 0),
                "train_f1": train_m.get("macro_f1", 0),
                "val_f1": val_m.get("macro_f1", 0),
            })

        val_losses = [e["val_loss"] for e in epochs]
        best = min(val_losses) if val_losses else float("inf")

        return {"epochs": epochs, "best_val_loss": best, "output_dir": output_dir}
    except Exception:
        return {"epochs": [], "best_val_loss": float("inf"), "output_dir": output_dir}


# ============================================================
# Restore API
# ============================================================

from app.schemas.artifact import (
    ArtifactRestoreRequest, ArtifactRestoreResponse,
    ArtifactPipelineRequest, ArtifactPipelineResponse,
    ArtifactJobResponse,
)
import base64 as _b64
import uuid as _uuid


def _volume_to_base64(vol: np.ndarray) -> str:
    return _b64.b64encode(vol.astype(np.float32).tobytes()).decode()


def _load_volume_for_restore(request, db: Session) -> np.ndarray:
    """根据请求加载 CT 体积"""
    if request.source == "dicom":
        if not request.series_id:
            raise HTTPException(status_code=400, detail="series_id required for dicom source")
        series = db.query(DicomSeries).filter(DicomSeries.id == request.series_id).first()
        if not series:
            raise HTTPException(status_code=404, detail=f"Series {request.series_id} not found")
        instances = (
            db.query(DicomInstance)
            .filter(DicomInstance.series_id == series.id)
            .order_by(DicomInstance.instance_number.asc().nulls_last())
            .all()
        )
        if not instances:
            raise HTTPException(status_code=404, detail="No instances found")
        storage = get_storage_backend()
        import io, pydicom as pdcm
        slices = []
        for inst in instances:
            if not inst.pixel_data_path:
                continue
            raw = storage.get_object_bytes(inst.pixel_data_path)
            if raw is None:
                continue
            try:
                ds = pdcm.dcmread(io.BytesIO(raw), force=True)
                px = ds.pixel_array.astype(np.float32)
                slope = getattr(ds, "RescaleSlope", None)
                intercept = getattr(ds, "RescaleIntercept", None)
                if slope is not None: px *= float(slope)
                if intercept is not None: px += float(intercept)
                slices.append(px)
            except Exception:
                continue
        if not slices:
            raise HTTPException(status_code=500, detail="No valid slices")
        return np.stack(slices, axis=0)
    else:
        shape = request.shape or [64, 64, 64]
        vol = np.full(shape, 40.0, dtype=np.float32)
        cz, cy, cx = shape[0]//2, shape[1]//2, shape[2]//2
        radius = min(shape) // 4
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        sphere = ((z - cz)**2 + (y - cy)**2 + (x - cx)**2) <= radius**2
        vol[sphere] = 400.0
        return vol


@router.post("/restore", response_model=ArtifactRestoreResponse)
async def restore_artifact(request: ArtifactRestoreRequest, db: Session = Depends(get_db)):
    """伪影修复"""
    from app.artifact.restoration.hybrid import HybridRestorer
    from app.artifact.evaluation.metrics import compute_all_metrics, format_quality_report

    volume = _load_volume_for_restore(request, db)
    spacing = request.spacing or [1.0, 1.0, 1.0]
    artifact_types = request.artifact_types or ["noise"]

    restorer = HybridRestorer()
    result = restorer.restore(
        volume, artifact_types=artifact_types,
        clean_reference=volume,  # 没有真实干净参考时用原始做参考
    )
    restored = result["restored_volume"]
    metrics = result["quality_metrics"]

    # 取中间切片返回
    mid = volume.shape[0] // 2
    return ArtifactRestoreResponse(
        restored_slice=restored[mid].tolist(),
        original_slice=volume[mid].tolist(),
        quality_metrics=metrics,
        steps=result["steps"],
        report=result["report"],
        shape=list(volume.shape),
        spacing=spacing,
    )


@router.post("/pipeline", response_model=ArtifactPipelineResponse)
async def artifact_pipeline(request: ArtifactPipelineRequest, db: Session = Depends(get_db)):
    """完整流水线: 生成 → 分类 → 修复"""
    from app.artifact.generator import get_generator
    from app.artifact.restoration.hybrid import HybridRestorer
    from app.artifact.evaluation.metrics import compute_all_metrics

    # Step 1: 生成伪影
    try:
        gen = get_generator(request.artifact_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    volume = _load_volume_for_restore(
        ArtifactRestoreRequest(source=request.source, series_id=request.series_id),
        db,
    )
    artifact_vol, mask, gen_meta = gen.generate(volume, (1.0, 1.0, 1.0), request.params)

    # Step 2: 分类
    classification = {
        "dominant_artifact": request.artifact_type,
        "scores": {request.artifact_type: 0.9},
    }

    # Step 3: 修复
    restorer = HybridRestorer()
    restore_result = restorer.restore(
        artifact_vol, artifact_types=[request.artifact_type],
        artifact_mask=mask, clean_reference=volume,
    )
    restored = restore_result["restored_volume"]
    metrics = restore_result["quality_metrics"]

    mid = request.slice_index if request.slice_index is not None else volume.shape[0] // 2
    mid = max(0, min(mid, volume.shape[0] - 1))

    return ArtifactPipelineResponse(
        artifact_type=request.artifact_type,
        classification=classification,
        restoration={"steps": restore_result["steps"], "report": restore_result["report"]},
        original_slice=volume[mid].tolist(),
        artifact_slice=artifact_vol[mid].tolist(),
        restored_slice=restored[mid].tolist(),
        quality_metrics=metrics,
        metadata=gen_meta,
    )


@router.get("/jobs", response_model=List[ArtifactJobResponse])
async def list_jobs(
    job_type: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """列出伪影处理作业"""
    from app.models.artifact import ArtifactJob
    query = db.query(ArtifactJob).order_by(ArtifactJob.created_at.desc())
    if job_type:
        query = query.filter(ArtifactJob.job_type == job_type)
    jobs = query.limit(limit).all()
    result = []
    for j in jobs:
        try:
            result.append(ArtifactJobResponse(
                id=j.id, job_type=j.job_type, status=j.status,
                config=j.config, output_path=j.output_path,
                quality_metrics=j.quality_metrics,
                created_at=j.created_at.isoformat() if j.created_at else None,
                completed_at=j.completed_at.isoformat() if j.completed_at else None,
            ))
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error serializing job {j.id}: {e}")
    return result


@router.get("/jobs/{job_id}", response_model=ArtifactJobResponse)
async def get_job(job_id: str, db: Session = Depends(get_db)):
    """获取作业详情"""
    from app.models.artifact import ArtifactJob
    job = db.query(ArtifactJob).filter(ArtifactJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return ArtifactJobResponse(
        id=job.id, job_type=job.job_type, status=job.status,
        config=job.config, output_path=job.output_path,
        quality_metrics=job.quality_metrics,
        created_at=job.created_at.isoformat() if job.created_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )
