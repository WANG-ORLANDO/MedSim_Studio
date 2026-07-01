"""
Segmentation ORM Models

SQLAlchemy models for AI-powered segmentation jobs and results.
Stores job configurations, status tracking, and references to
segmentation mask data in the storage backend.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, JSON, Boolean
from app.database.session import Base


class SegmentationJob(Base):
    """Represents an AI segmentation job using MONAI models."""
    __tablename__ = "segmentation_jobs"

    id = Column(String, primary_key=True, index=True)
    study_id = Column(String, nullable=False, index=True)
    series_id = Column(String, nullable=False, index=True)
    status = Column(String, default="pending")  # pending, running, completed, failed

    # Configuration
    model_name = Column(String, default="unet")  # unet, segresnet, swin_unetr, totalsegmentator, nnunet_handoff
    target_organs = Column(JSON, default=list)
    detect_lesions = Column(Boolean, default=False)

    # Progress tracking
    progress = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)

    # Output paths (storage object keys)
    mask_path = Column(String, nullable=True)
    label_map_path = Column(String, nullable=True)

    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<SegmentationJob {self.id} [{self.status}] model={self.model_name}>"
