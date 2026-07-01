"""API v1 route handlers."""
from app.api.v1.health import router as health_router
from app.api.v1.dicom import router as dicom_router
from app.api.v1.simulation import router as simulation_router
from app.api.v1.segmentation import router as segmentation_router

__all__ = ["health_router", "dicom_router", "simulation_router", "segmentation_router"]
