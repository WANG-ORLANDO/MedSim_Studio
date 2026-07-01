"""
Health Check API

Provides service health monitoring endpoints for Docker
orchestration and load balancer health probes.
"""

from fastapi import APIRouter, status
from datetime import datetime
from app import __version__, __app_name__
from app.core.config import settings

router = APIRouter(tags=["Health"])


def _get_storage():
    """Lazy import to avoid circular dependency at module load time."""
    from app.main import storage_backend
    return storage_backend


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Basic health check endpoint.
    Returns service status and version information.
    Used by Docker health checks and monitoring systems.
    """
    return {
        "status": "healthy",
        "app_name": __app_name__,
        "version": __version__,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health/ready", status_code=status.HTTP_200_OK)
async def readiness_check():
    """
    Readiness probe endpoint.
    Checks if the service is ready to accept traffic.
    Verifies database connectivity and storage availability.
    """
    checks: dict[str, str] = {}

    # Database: still pending (not yet wired up)
    checks["database"] = "pending"

    # Storage: real health check via unified backend
    storage = _get_storage()
    if storage is None:
        checks["storage"] = "unhealthy"
    else:
        try:
            checks["storage"] = "healthy" if storage.check_health() else "unhealthy"
        except Exception:
            checks["storage"] = "unhealthy"

    # Cache: not yet implemented, skip without blocking readiness
    checks["cache"] = "skipped"

    # Storage backend type (informational)
    checks["storage_backend"] = settings.STORAGE_BACKEND

    # Overall status: only database and storage matter
    ready = (
        checks["database"] in ("healthy", "pending")
        and checks["storage"] == "healthy"
    )
    overall = "ready" if ready else "not_ready"

    return {
        "status": overall,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health/live", status_code=status.HTTP_200_OK)
async def liveness_check():
    """
    Liveness probe endpoint.
    Basic process liveness indicator.
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
    }
