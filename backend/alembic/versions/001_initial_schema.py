"""Initial schema: DICOM, simulation, and segmentation tables

Revision ID: 001
Revises: None
Create Date: 2026-06-11
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def table_exists(table_name: str) -> bool:
    """Check if a database table already exists."""
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    """Create all initial tables (idempotent — skips existing tables)."""

    # ── DicomStudy ──────────────────────────────────────────────
    if not table_exists("dicom_studies"):
        op.create_table(
            "dicom_studies",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("patient_id", sa.String(), nullable=False, index=True),
            sa.Column("patient_name", sa.String(), nullable=False),
            sa.Column("patient_birth_date", sa.String(), nullable=True),
            sa.Column("patient_sex", sa.String(), nullable=True),
            sa.Column("study_instance_uid", sa.String(), nullable=False, unique=True, index=True),
            sa.Column("study_date", sa.String(), nullable=True),
            sa.Column("study_time", sa.String(), nullable=True),
            sa.Column("study_description", sa.String(), nullable=True),
            sa.Column("accession_number", sa.String(), nullable=True),
            sa.Column("referring_physician", sa.String(), nullable=True),
            sa.Column("modalities", sa.JSON(), nullable=True),
            sa.Column("series_count", sa.Integer(), nullable=True),
            sa.Column("instance_count", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_dicom_studies_id", "dicom_studies", ["id"])

    # ── DicomSeries ─────────────────────────────────────────────
    if not table_exists("dicom_series"):
        op.create_table(
            "dicom_series",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("study_id", sa.String(), sa.ForeignKey("dicom_studies.id"), nullable=False, index=True),
            sa.Column("series_instance_uid", sa.String(), nullable=False, unique=True, index=True),
            sa.Column("series_number", sa.Integer(), nullable=True),
            sa.Column("series_description", sa.String(), nullable=True),
            sa.Column("modality", sa.String(), nullable=True),
            sa.Column("manufacturer", sa.String(), nullable=True),
            sa.Column("body_part_examined", sa.String(), nullable=True),
            sa.Column("laterality", sa.String(), nullable=True),
            sa.Column("protocol_name", sa.String(), nullable=True),
            sa.Column("image_count", sa.Integer(), nullable=True),
            sa.Column("series_date", sa.String(), nullable=True),
            sa.Column("rows", sa.Integer(), nullable=True),
            sa.Column("columns", sa.Integer(), nullable=True),
            sa.Column("slice_thickness", sa.Float(), nullable=True),
            sa.Column("pixel_spacing", sa.JSON(), nullable=True),
            sa.Column("window_center", sa.Float(), nullable=True),
            sa.Column("window_width", sa.Float(), nullable=True),
            sa.Column("storage_path", sa.String(), nullable=True),
            sa.Column("file_count", sa.Integer(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_dicom_series_id", "dicom_series", ["id"])
        op.create_index("ix_dicom_series_study_id", "dicom_series", ["study_id"])

    # ── DicomInstance ───────────────────────────────────────────
    if not table_exists("dicom_instances"):
        op.create_table(
            "dicom_instances",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("series_id", sa.String(), sa.ForeignKey("dicom_series.id"), nullable=False, index=True),
            sa.Column("sop_instance_uid", sa.String(), nullable=False, unique=True),
            sa.Column("instance_number", sa.Integer(), nullable=True),
            sa.Column("image_position", sa.JSON(), nullable=True),
            sa.Column("image_orientation", sa.JSON(), nullable=True),
            sa.Column("slice_location", sa.Float(), nullable=True),
            sa.Column("rows", sa.Integer(), nullable=True),
            sa.Column("columns", sa.Integer(), nullable=True),
            sa.Column("pixel_data_path", sa.String(), nullable=True),
            sa.Column("file_size", sa.Integer(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_dicom_instances_id", "dicom_instances", ["id"])
        op.create_index("ix_dicom_instances_series_id", "dicom_instances", ["series_id"])

    # ── SimulationJob ───────────────────────────────────────────
    if not table_exists("simulation_jobs"):
        op.create_table(
            "simulation_jobs",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("study_id", sa.String(), sa.ForeignKey("dicom_studies.id"), nullable=True),
            sa.Column("series_id", sa.String(), sa.ForeignKey("dicom_series.id"), nullable=True),
            sa.Column("status", sa.String(), nullable=True),
            sa.Column("lesion_count", sa.Integer(), nullable=True),
            sa.Column("organ_count", sa.Integer(), nullable=True),
            sa.Column("has_deformation", sa.Boolean(), nullable=True),
            sa.Column("output_format", sa.String(), nullable=True),
            sa.Column("progress", sa.Float(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("output_path", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_simulation_jobs_id", "simulation_jobs", ["id"])

    # ── LesionConfig ────────────────────────────────────────────
    if not table_exists("lesion_configs"):
        op.create_table(
            "lesion_configs",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("job_id", sa.String(), sa.ForeignKey("simulation_jobs.id"), nullable=False),
            sa.Column("lesion_type", sa.String(), nullable=False),
            sa.Column("shape", sa.String(), nullable=True),
            sa.Column("center_x", sa.Float(), nullable=True),
            sa.Column("center_y", sa.Float(), nullable=True),
            sa.Column("center_z", sa.Float(), nullable=True),
            sa.Column("radius_x", sa.Float(), nullable=True),
            sa.Column("radius_y", sa.Float(), nullable=True),
            sa.Column("radius_z", sa.Float(), nullable=True),
            sa.Column("hu_mean", sa.Float(), nullable=True),
            sa.Column("hu_std", sa.Float(), nullable=True),
            sa.Column("margin_sharpness", sa.Float(), nullable=True),
            sa.Column("calcification_fraction", sa.Float(), nullable=True),
            sa.Column("necrosis_fraction", sa.Float(), nullable=True),
            sa.Column("spiculation_degree", sa.Float(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_lesion_configs_id", "lesion_configs", ["id"])

    # ── OrganConfig ─────────────────────────────────────────────
    if not table_exists("organ_configs"):
        op.create_table(
            "organ_configs",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("job_id", sa.String(), sa.ForeignKey("simulation_jobs.id"), nullable=False),
            sa.Column("organ_type", sa.String(), nullable=False),
            sa.Column("hu_mean", sa.Float(), nullable=True),
            sa.Column("hu_std", sa.Float(), nullable=True),
            sa.Column("enable_noise", sa.Boolean(), nullable=True),
            sa.Column("noise_level", sa.Float(), nullable=True),
            sa.Column("enable_enhancement", sa.Boolean(), nullable=True),
            sa.Column("enhancement_pattern", sa.String(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_organ_configs_id", "organ_configs", ["id"])

    # ── SegmentationJob ─────────────────────────────────────────
    if not table_exists("segmentation_jobs"):
        op.create_table(
            "segmentation_jobs",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("study_id", sa.String(), nullable=False, index=True),
            sa.Column("series_id", sa.String(), nullable=False, index=True),
            sa.Column("status", sa.String(), nullable=True),
            sa.Column("model_name", sa.String(), nullable=True),
            sa.Column("target_organs", sa.JSON(), nullable=True),
            sa.Column("detect_lesions", sa.Boolean(), nullable=True),
            sa.Column("progress", sa.Float(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("mask_path", sa.String(), nullable=True),
            sa.Column("label_map_path", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_segmentation_jobs_id", "segmentation_jobs", ["id"])
        op.create_index("ix_segmentation_jobs_study_id", "segmentation_jobs", ["study_id"])
        op.create_index("ix_segmentation_jobs_series_id", "segmentation_jobs", ["series_id"])


def downgrade() -> None:
    """Drop all tables in reverse dependency order."""
    op.drop_table("lesion_configs")
    op.drop_table("organ_configs")
    op.drop_table("segmentation_jobs")
    op.drop_table("simulation_jobs")
    op.drop_table("dicom_instances")
    op.drop_table("dicom_series")
    op.drop_table("dicom_studies")
