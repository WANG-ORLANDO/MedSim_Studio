/**
 * Segmentation Type Definitions
 *
 * Types for AI-powered organ/lesion segmentation and interactive
 * refinement within the CT Simulator platform.
 */

/** Status of a segmentation job */
export type SegmentationJobStatus = 'pending' | 'running' | 'completed' | 'failed';

/** Full segmentation job returned from the backend */
export interface SegmentationJob {
  id: string;
  studyId: string;
  seriesId: string;
  status: SegmentationJobStatus;
  modelName: string;
  targetOrgans: string[];
  detectLesions: boolean;
  progress: number;
  errorMessage: string | null;
  maskPath: string | null;
  labelMapPath: string | null;
  createdAt: string;
  updatedAt: string | null;
  startedAt: string | null;
  completedAt: string | null;
}

/** Request body for creating a new segmentation job */
export interface SegmentationJobCreate {
  studyId: string;
  seriesId: string;
  modelName: string;
  targetOrgans: string[];
  detectLesions: boolean;
}

/** Segmentation model info from the backend */
export interface SegmentationModel {
  name: string;
  description: string;
  organs: string[];
  status: 'available' | 'coming_soon';
}

/** A single segmentation label (organ/lesion type) */
export interface SegmentationLabel {
  index: number;
  name: string;
  color: [number, number, number]; // RGB values 0-255
  /** Optional category grouping (e.g., "abdomen", "bones") for TotalSegmentator */
  category?: string;
  /** Optional human-readable category label */
  category_label?: string;
}

/** 2D slice mask for overlay rendering */
export interface SliceMask {
  zIndex: number;
  rows: number;
  cols: number;
  labels: SegmentationLabel[];
  maskData: number[][]; // 2D array of label indices (rows of cols)
}

/** Request for interactive click refinement */
export interface InteractiveClickRequest {
  jobId: string;
  zIndex: number;
  x: number;
  y: number;
  label: number;
  operation: 'add' | 'remove';
}

/** Response from interactive click refinement */
export interface InteractiveClickResponse {
  zIndex: number;
  updatedRows: number;
  updatedCols: number;
  patchData: number[][]; // Updated local mask patch
}

/** Export format options */
export type ExportFormat = 'nifti' | 'nrrd' | 'dicom_seg';
