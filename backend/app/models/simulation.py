"""
Simulation ORM Models

SQLAlchemy models for lesion simulation jobs, configurations,
and results. Stores simulation parameters and generated data references.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text, JSON, Boolean
from sqlalchemy.orm import relationship
from app.database.session import Base


class SimulationJob(Base):
    """Represents a lesion/organ simulation job."""
    __tablename__ = "simulation_jobs"

    id = Column(String, primary_key=True, index=True)
    study_id = Column(String, ForeignKey("dicom_studies.id"), nullable=True)
    series_id = Column(String, ForeignKey("dicom_series.id"), nullable=True)
    status = Column(String, default="pending")  # pending, running, completed, failed

    # Configuration
    lesion_count = Column(Integer, default=0)
    organ_count = Column(Integer, default=0)
    has_deformation = Column(Boolean, default=False)
    output_format = Column(String, default="dicom")

    # Progress tracking
    progress = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)

    # Output
    output_path = Column(String, nullable=True)

    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    lesions = relationship("LesionConfig", back_populates="job", cascade="all, delete-orphan")
    organs = relationship("OrganConfig", back_populates="job", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<SimulationJob {self.id} [{self.status}]>"


class LesionConfig(Base):
    """Configuration for a single simulated lesion."""
    __tablename__ = "lesion_configs"

    id = Column(String, primary_key=True, index=True)
    job_id = Column(String, ForeignKey("simulation_jobs.id"), nullable=False)

    # Lesion parameters
    lesion_type = Column(String, nullable=False)  # tumor, nodule, cyst, etc.
    shape = Column(String, default="spherical")  # spherical, ellipsoidal, irregular
    center_x = Column(Float, default=0.0)
    center_y = Column(Float, default=0.0)
    center_z = Column(Float, default=0.0)
    radius_x = Column(Float, default=10.0)  # mm
    radius_y = Column(Float, default=10.0)  # mm
    radius_z = Column(Float, default=10.0)  # mm
    hu_mean = Column(Float, default=40.0)
    hu_std = Column(Float, default=20.0)
    margin_sharpness = Column(Float, default=0.8)
    calcification_fraction = Column(Float, default=0.0)
    necrosis_fraction = Column(Float, default=0.0)
    spiculation_degree = Column(Float, default=0.0)

    # Relationships
    job = relationship("SimulationJob", back_populates="lesions")

    def __repr__(self):
        return f"<LesionConfig {self.lesion_type} ({self.hu_mean} HU)>"


class OrganConfig(Base):
    """Configuration for simulated organ tissue."""
    __tablename__ = "organ_configs"

    id = Column(String, primary_key=True, index=True)
    job_id = Column(String, ForeignKey("simulation_jobs.id"), nullable=False)

    organ_type = Column(String, nullable=False)  # liver, kidney, lung, etc.
    hu_mean = Column(Float, default=40.0)
    hu_std = Column(Float, default=10.0)
    enable_noise = Column(Boolean, default=True)
    noise_level = Column(Float, default=0.1)
    enable_enhancement = Column(Boolean, default=False)
    enhancement_pattern = Column(String, default="none")

    # Relationships
    job = relationship("SimulationJob", back_populates="organs")

    def __repr__(self):
        return f"<OrganConfig {self.organ_type}>"
