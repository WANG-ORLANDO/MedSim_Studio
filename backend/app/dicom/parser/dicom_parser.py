"""
DICOM File Parser

Handles parsing of DICOM files, metadata extraction, and database storage.
Supports standard DICOM files and extracts key medical imaging parameters
for visualization and processing pipelines.
"""

import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from sqlalchemy.orm import Session

from app.models.dicom import DicomStudy, DicomSeries, DicomInstance

logger = logging.getLogger(__name__)


def safe_str(value, default=None):
    try:
        if value is None:
            return default
        return str(value)
    except Exception:
        return default


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, (list, tuple)) or (hasattr(value, "__iter__") and not isinstance(value, (str, bytes))):
            items = list(value)
            value = items[0] if items else None
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, (list, tuple)) or (hasattr(value, "__iter__") and not isinstance(value, (str, bytes))):
            items = list(value)
            value = items[0] if items else None
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def safe_list(value, converter=None, default=None):
    try:
        if value is None:
            return default if default is not None else []
        if isinstance(value, (str, bytes)):
            return [converter(value) if converter else value]
        items = list(value)
        if converter:
            return [converter(v) for v in items if v is not None]
        return items
    except Exception:
        return default if default is not None else []


class DicomParserError(Exception):
    """Custom exception for DICOM parsing errors."""
    pass


class DicomParser:
    """
    Parser for DICOM medical imaging files.

    Extracts metadata from DICOM files and creates corresponding
    database records for studies, series, and instances.
    """

    # Required DICOM tags for study level
    STUDY_TAGS = [
        "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
        "StudyInstanceUID", "StudyDate", "StudyTime", "StudyDescription",
        "AccessionNumber", "ReferringPhysicianName",
    ]

    # Required DICOM tags for series level
    SERIES_TAGS = [
        "SeriesInstanceUID", "SeriesNumber", "SeriesDescription",
        "Modality", "Manufacturer", "BodyPartExamined",
        "Laterality", "ProtocolName",
    ]

    # Required DICOM tags for instance level
    INSTANCE_TAGS = [
        "SOPInstanceUID", "InstanceNumber",
        "Rows", "Columns", "SliceLocation",
    ]

    def parse_files(
        self,
        file_paths: List[str],
        db: Session,
        storage_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Parse a list of DICOM files and store metadata in the database.

        Args:
            file_paths: List of paths to DICOM files
            db: SQLAlchemy database session

        Returns:
            Dictionary with study_id, series_count, and instance_count

        Raises:
            DicomParserError: If parsing fails for all files
        """
        if not file_paths:
            raise DicomParserError("No files provided for parsing")

        # Group files by StudyInstanceUID
        studies_data: Dict[str, Dict] = {}
        series_data: Dict[str, Dict] = {}
        instances: List[Dict] = []

        for file_path in file_paths:
            try:
                ds = pydicom.dcmread(file_path, force=True)
                self._validate_dataset(ds)

                study_uid = ds.StudyInstanceUID
                series_uid = ds.SeriesInstanceUID

                storage_object_key = storage_map.get(file_path) if storage_map else None

                study_meta = self._extract_study_metadata(ds)
                series_info = self._extract_series_metadata(ds, file_path)
                if storage_object_key:
                    series_info["storage_path"] = storage_object_key.rsplit("/", 1)[0] + "/"
                instance_info = self._extract_instance_metadata(ds, file_path)
                if storage_object_key:
                    instance_info["pixel_data_path"] = storage_object_key

                if study_uid not in studies_data:
                    studies_data[study_uid] = study_meta
                if series_uid not in series_data:
                    series_data[series_uid] = series_info
                instances.append(instance_info)

                logger.info(
                    "Parsed %s: study=%s series=%s sop=%s",
                    os.path.basename(file_path), study_uid, series_uid,
                    instance_info.get("sop_instance_uid"),
                )

            except Exception as e:
                logger.warning("Failed to parse %s: %s", file_path, e, exc_info=True)
                continue

        logger.info(
            "Parse complete: studies=%d series=%d instances=%d",
            len(studies_data), len(series_data), len(instances),
        )

        if not studies_data:
            raise DicomParserError("No valid DICOM files found")

        # Store in database
        study_id = self._store_studies(studies_data, series_data, instances, db)

        return {
            "study_id": study_id,
            "series_count": len(series_data),
            "instance_count": len(instances),
        }

    def _validate_dataset(self, ds: Dataset) -> None:
        """Validate that the dataset contains required DICOM tags."""
        required_tags = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"]
        missing = [tag for tag in required_tags if not hasattr(ds, tag)]
        if missing:
            raise DicomParserError(f"Missing required tags: {missing}")

    def _extract_study_metadata(self, ds: Dataset) -> Dict[str, Any]:
        """Extract study-level metadata from a DICOM dataset."""
        modality = self._get_tag(ds, "Modality")
        return {
            "patient_id": self._get_tag(ds, "PatientID", "UNKNOWN"),
            "patient_name": str(getattr(ds, "PatientName", "UNKNOWN")),
            "patient_birth_date": self._get_tag(ds, "PatientBirthDate"),
            "patient_sex": self._get_tag(ds, "PatientSex"),
            "study_instance_uid": ds.StudyInstanceUID,
            "study_date": self._get_tag(ds, "StudyDate"),
            "study_time": self._get_tag(ds, "StudyTime"),
            "study_description": self._get_tag(ds, "StudyDescription"),
            "accession_number": self._get_tag(ds, "AccessionNumber"),
            "referring_physician": str(getattr(ds, "ReferringPhysicianName", "")),
            "modalities": [modality] if modality else [],
        }

    def _extract_series_metadata(self, ds: Dataset, file_path: str) -> Dict[str, Any]:
        """Extract series-level metadata from a DICOM dataset."""
        pixel_spacing = safe_list(getattr(ds, "PixelSpacing", None), converter=safe_float) or None
        if pixel_spacing is not None and any(v is None for v in pixel_spacing):
            pixel_spacing = None

        window_center = safe_float(getattr(ds, "WindowCenter", None))
        window_width = safe_float(getattr(ds, "WindowWidth", None))

        return {
            "series_instance_uid": ds.SeriesInstanceUID,
            "series_number": safe_int(getattr(ds, "SeriesNumber", None)),
            "series_description": safe_str(getattr(ds, "SeriesDescription", None)),
            "modality": safe_str(getattr(ds, "Modality", None)),
            "manufacturer": safe_str(getattr(ds, "Manufacturer", None)),
            "body_part_examined": safe_str(getattr(ds, "BodyPartExamined", None)),
            "laterality": safe_str(getattr(ds, "Laterality", None)),
            "protocol_name": safe_str(getattr(ds, "ProtocolName", None)),
            "rows": safe_int(getattr(ds, "Rows", None)),
            "columns": safe_int(getattr(ds, "Columns", None)),
            "slice_thickness": safe_float(getattr(ds, "SliceThickness", None)),
            "pixel_spacing": pixel_spacing,
            "window_center": window_center,
            "window_width": window_width,
            "storage_path": file_path,
        }

    def _extract_instance_metadata(self, ds: Dataset, file_path: str) -> Dict[str, Any]:
        """Extract instance-level metadata from a DICOM dataset."""
        image_position = safe_list(getattr(ds, "ImagePositionPatient", None), converter=safe_float) or None
        image_orientation = safe_list(getattr(ds, "ImageOrientationPatient", None), converter=safe_float) or None

        return {
            "sop_instance_uid": ds.SOPInstanceUID,
            "series_instance_uid": ds.SeriesInstanceUID,
            "instance_number": safe_int(getattr(ds, "InstanceNumber", None)),
            "image_position": image_position,
            "image_orientation": image_orientation,
            "slice_location": safe_float(getattr(ds, "SliceLocation", None)),
            "rows": safe_int(getattr(ds, "Rows", None)),
            "columns": safe_int(getattr(ds, "Columns", None)),
            "pixel_data_path": file_path,
            "file_size": os.path.getsize(file_path),
        }

    def _store_studies(
        self,
        studies_data: Dict[str, Dict],
        series_data: Dict[str, Dict],
        instances: List[Dict],
        db: Session,
    ) -> str:
        """Store parsed DICOM data in the database."""
        first_study_uid = list(studies_data.keys())[0]
        study_info = studies_data[first_study_uid]

        # Check if study already exists
        existing = db.query(DicomStudy).filter(
            DicomStudy.study_instance_uid == first_study_uid
        ).first()

        if existing:
            study_id = existing.id
            existing.series_count = len(series_data)
            existing.instance_count = len(instances)
            logger.info(f"Appending to existing study {study_id}")
        else:
            study_id = str(uuid.uuid4())
            study = DicomStudy(
                id=study_id,
                **study_info,
                series_count=len(series_data),
                instance_count=len(instances),
            )
            db.add(study)

        # Store series
        for series_uid, series_info in series_data.items():
            existing_series = db.query(DicomSeries).filter(
                DicomSeries.series_instance_uid == series_uid
            ).first()

            if not existing_series:
                series_instance_count = sum(
                    1 for inst in instances
                    if inst.get("series_instance_uid") == series_uid
                )
                series = DicomSeries(
                    id=str(uuid.uuid4()),
                    study_id=study_id,
                    **series_info,
                    image_count=series_instance_count,
                    file_count=series_instance_count,
                )
                db.add(series)
                logger.info(f"Created series {series_uid} with {series_instance_count} instances")

        # Flush so instance queries can find the newly created series
        db.flush()

        # Store instances
        instances_created = 0
        for inst_info in instances:
            existing_instance = db.query(DicomInstance).filter(
                DicomInstance.sop_instance_uid == inst_info["sop_instance_uid"]
            ).first()

            if not existing_instance:
                series = db.query(DicomSeries).filter(
                    DicomSeries.series_instance_uid == inst_info.get("series_instance_uid", "")
                ).first()

                if series:
                    instance = DicomInstance(
                        id=str(uuid.uuid4()),
                        series_id=series.id,
                        **{k: v for k, v in inst_info.items() if k != "series_instance_uid"},
                    )
                    db.add(instance)
                    instances_created += 1
                else:
                    logger.warning(
                        "No series found for instance %s with series_uid %s",
                        inst_info.get("sop_instance_uid"),
                        inst_info.get("series_instance_uid"),
                    )

        db.commit()
        logger.info(f"Stored study {study_id}: {len(series_data)} series, {instances_created} instances created")
        return study_id

    @staticmethod
    def _get_tag(ds: Dataset, tag_name: str, default: Optional[str] = None) -> Optional[str]:
        """Safely get a DICOM tag value, returning default if not present."""
        value = getattr(ds, tag_name, default)
        if value is None:
            return default
        return str(value)

    @staticmethod
    def _get_int_tag(ds: Dataset, tag_name: str) -> Optional[int]:
        """Safely get an integer DICOM tag value."""
        value = getattr(ds, tag_name, None)
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _get_float_tag(ds: Dataset, tag_name: str) -> Optional[float]:
        """Safely get a float DICOM tag value."""
        value = getattr(ds, tag_name, None)
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            try:
                return float(value[0])  # handle MultiValue
            except (ValueError, TypeError, IndexError):
                return None
