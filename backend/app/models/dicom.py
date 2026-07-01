"""
DICOM ORM Models

SQLAlchemy models for DICOM study, series, and instance metadata.
Stores parsed DICOM tags for efficient querying and retrieval.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from app.database.session import Base


class DicomStudy(Base):
    """Represents a DICOM study (typically one patient exam)."""
    __tablename__ = "dicom_studies"

    id = Column(String, primary_key=True, index=True)
    patient_id = Column(String, index=True, nullable=False)
    patient_name = Column(String, nullable=False)
    patient_birth_date = Column(String, nullable=True)
    patient_sex = Column(String, nullable=True)
    study_instance_uid = Column(String, unique=True, nullable=False, index=True)
    study_date = Column(String, nullable=True)
    study_time = Column(String, nullable=True)
    study_description = Column(String, nullable=True)
    accession_number = Column(String, nullable=True)
    referring_physician = Column(String, nullable=True)
    modalities = Column(JSON, default=list)
    series_count = Column(Integer, default=0)
    instance_count = Column(Integer, default=0)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    series = relationship("DicomSeries", back_populates="study", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<DicomStudy {self.patient_name} ({self.study_instance_uid[:12]}...)>"


class DicomSeries(Base):
    """Represents a DICOM series within a study."""
    __tablename__ = "dicom_series"

    id = Column(String, primary_key=True, index=True)
    study_id = Column(String, ForeignKey("dicom_studies.id"), nullable=False, index=True)
    series_instance_uid = Column(String, unique=True, nullable=False, index=True)
    series_number = Column(Integer, nullable=True)
    series_description = Column(String, nullable=True)
    modality = Column(String, nullable=True)
    manufacturer = Column(String, nullable=True)
    body_part_examined = Column(String, nullable=True)
    laterality = Column(String, nullable=True)
    protocol_name = Column(String, nullable=True)
    image_count = Column(Integer, default=0)
    series_date = Column(String, nullable=True)

    # Image parameters
    rows = Column(Integer, nullable=True)
    columns = Column(Integer, nullable=True)
    slice_thickness = Column(Float, nullable=True)
    pixel_spacing = Column(JSON, nullable=True)
    window_center = Column(Float, nullable=True)
    window_width = Column(Float, nullable=True)

    # Storage
    storage_path = Column(String, nullable=True)
    file_count = Column(Integer, default=0)

    # Relationships
    study = relationship("DicomStudy", back_populates="series")
    instances = relationship("DicomInstance", back_populates="series", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<DicomSeries #{self.series_number} ({self.modality})>"


class DicomInstance(Base):
    """Represents a single DICOM image instance."""
    __tablename__ = "dicom_instances"

    id = Column(String, primary_key=True, index=True)
    series_id = Column(String, ForeignKey("dicom_series.id"), nullable=False, index=True)
    sop_instance_uid = Column(String, unique=True, nullable=False)
    instance_number = Column(Integer, nullable=True)
    image_position = Column(JSON, nullable=True)  # (x, y, z) in patient space
    image_orientation = Column(JSON, nullable=True)  # direction cosines
    slice_location = Column(Float, nullable=True)
    rows = Column(Integer, nullable=True)
    columns = Column(Integer, nullable=True)
    pixel_data_path = Column(String, nullable=True)  # Path to stored pixel data
    file_size = Column(Integer, default=0)

    # Relationships
    series = relationship("DicomSeries", back_populates="instances")

    def __repr__(self):
        return f"<DicomInstance #{self.instance_number}>"
