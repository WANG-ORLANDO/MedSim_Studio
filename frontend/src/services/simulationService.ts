import { api } from './api';
import type {
  CtParamsPreviewRequest,
  CtParamsPreviewResponse,
  LesionConfig,
  SimulationJob,
} from '@/types/simulation';
import type { AxiosProgressEvent, AxiosResponse } from 'axios';

/**
 * Simulation Service
 *
 * Service layer for lesion and organ simulation operations.
 * Interfaces with the backend simulation engine for:
 * - Lesion generation with configurable HU values
 * - Organ simulation with realistic tissue properties
 * - Deformation field computation
 * - NIfTI/DICOM export of simulated data
 * - CT phantom generation (synthetic upper-body CT)
 *
 * Field name mapping (frontend ↔ backend):
 *   type          ↔ lesion_type
 *   center        ↔ center_x, center_y, center_z  (tuple → separate fields)
 *   radiusMm      ↔ radius_x, radius_y, radius_z  (tuple → separate fields)
 */

// ---------------------------------------------------------------------------
// CT Phantom types
// ---------------------------------------------------------------------------

/** Metadata returned alongside the CT phantom volume */
export interface PhantomMetadata {
  width: number;
  height: number;
  depth: number;
  spacing: [number, number, number]; // (z, y, x) in mm
  windowPresets: Record<string, { windowLevel: number; windowWidth: number }>;
  bodyThresholdHU?: number; // voxels >= this HU belong to the body (≈ -500)
  source?: 'procedural' | 'atlas';
  caseId?: string;
  labelMap?: Record<number, string>; // organ index → name (atlas only)
  description: string;

  // ---- Debug / Orientation fields (atlas mode) ----
  originalShape?: [number, number, number];    // [z, y, x] before resampling
  outputShape?: [number, number, number];      // [z, y, x] after resampling
  originalSpacing?: [number, number, number];  // [z, y, x] in mm, NIfTI native
  outputSpacing?: [number, number, number];    // [z, y, x] after resampling
  scanAxis?: string;                           // always "z"
  scanDirection?: string;                      // "head_to_feet" or "feet_to_head"
  flippedZ?: boolean;                          // true if z-axis was flipped
  niftiRz?: number;                            // raw S-component of NIfTI z-axis (debug)

  // ---- Label statistics (atlas mode, when labels available) ----
  labelNonzeroCounts?: Record<number, number>; // label_id → voxel count
  sliceLabelPresence?: Record<string, [number, number]>; // organ_name → [z_min, z_max]
}

/** Response from GET /simulation/phantom */
export interface PhantomResponse {
  volumeBase64: string;
  labelBase64?: string | null; // optional uint8 label map (atlas only)
  metadata: PhantomMetadata;
}

// ---------------------------------------------------------------------------
// Field name conversion helpers
// ---------------------------------------------------------------------------

/**
 * Convert frontend LesionConfig to backend API format.
 *
 * Frontend uses camelCase with tuples for multi-value fields;
 * backend uses snake_case with individual fields.
 */
function toBackendLesion(lesion: LesionConfig): Record<string, unknown> {
  return {
    lesion_type: lesion.type,
    shape: lesion.shape,
    center_x: lesion.center[0],
    center_y: lesion.center[1],
    center_z: lesion.center[2],
    radius_x: lesion.radiusMm[0],
    radius_y: lesion.radiusMm[1],
    radius_z: lesion.radiusMm[2],
    hu_mean: lesion.huMean,
    hu_std: lesion.huStd,
    margin_sharpness: lesion.marginSharpness,
    calcification_fraction: lesion.calcificationFraction,
    necrosis_fraction: lesion.necrosisFraction,
    spiculation_degree: lesion.spiculationDegree,
  };
}

/**
 * Convert backend-format lesion response back to frontend LesionConfig.
 */
function toFrontendLesion(backend: Record<string, unknown>): LesionConfig {
  return {
    type: (backend.lesion_type as LesionConfig['type']) ?? 'tumor',
    shape: (backend.shape as LesionConfig['shape']) ?? 'spherical',
    center: [
      (backend.center_x as number) ?? 0,
      (backend.center_y as number) ?? 0,
      (backend.center_z as number) ?? 0,
    ],
    radiusMm: [
      (backend.radius_x as number) ?? 10,
      (backend.radius_y as number) ?? 10,
      (backend.radius_z as number) ?? 10,
    ],
    huMean: (backend.hu_mean as number) ?? 40,
    huStd: (backend.hu_std as number) ?? 20,
    marginSharpness: (backend.margin_sharpness as number) ?? 0.8,
    calcificationFraction: (backend.calcification_fraction as number) ?? 0,
    necrosisFraction: (backend.necrosis_fraction as number) ?? 0,
    spiculationDegree: (backend.spiculation_degree as number) ?? 0,
  };
}

/**
 * Convert frontend SimulationJob to backend create-job payload.
 * Handles nested lesion and organ config conversion.
 */
function toBackendJob(config: Partial<SimulationJob>): Record<string, unknown> {
  const payload: Record<string, unknown> = {};

  if (config.studyId !== undefined) payload.study_id = config.studyId;
  if (config.seriesId !== undefined) payload.series_id = config.seriesId;
  if (config.outputFormat !== undefined) payload.output_format = config.outputFormat;
  if (config.lesions !== undefined) {
    payload.lesions = config.lesions.map(toBackendLesion);
  }
  // Organs — send as-is (backend accepts List[dict])
  if (config.organs !== undefined) {
    payload.organs = config.organs.map((o) => ({
      organ_type: o.organType,
      hu_mean: o.huMean,
      hu_std: o.huStd,
      enable_noise: o.enableNoise,
      noise_level: o.noiseLevel,
      enable_enhancement: o.enableEnhancement,
      enhancement_pattern: o.enhancementPattern,
    }));
  }

  return payload;
}

// ---------------------------------------------------------------------------
// Typed backend response shape for preview endpoints
// ---------------------------------------------------------------------------

interface PreviewResponse {
  jobId: string;
  previewData: Record<string, unknown>;
  voxelCount: number;
  huRange: [number, number];
}

/** Response from previewing a lesion on a real DICOM series */
export interface DicomLesionPreviewResponse {
  imageBase64: string;
  sliceIndex: number;
  totalSlices: number;
  lesionCenterVoxel: number[];
  huMin: number;
  huMax: number;
  huMean: number;
  huStd: number;
  voxelCount: number;
  volumeMm3: number;
}

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

export const simulationService = {
  /** Create a new simulation job */
  createJob: (config: Partial<SimulationJob>) =>
    api.post<SimulationJob>('/simulation/jobs', toBackendJob(config)),

  /** Get simulation job status */
  getJobStatus: (jobId: string) =>
    api.get<SimulationJob>(`/simulation/jobs/${jobId}`),

  /** List all simulation jobs for a study */
  getStudyJobs: (studyId: string) =>
    api.get<SimulationJob[]>(`/simulation/jobs?study_id=${studyId}`),

  /** Cancel a running simulation job */
  cancelJob: (jobId: string) =>
    api.post(`/simulation/jobs/${jobId}/cancel`),

  /** Generate a lesion preview (fast, synchronous) */
  previewLesion: (config: LesionConfig) =>
    api.post<PreviewResponse>('/simulation/preview/lesion', toBackendLesion(config)),

  /** Preview a lesion overlaid on a real DICOM series */
  previewLesionOnDicom: (
    seriesId: string,
    lesion: LesionConfig,
    windowCenter = 40,
    windowWidth = 400,
  ) =>
    api.post<DicomLesionPreviewResponse>('/simulation/preview/lesion-on-dicom', {
      series_id: seriesId,
      lesion: toBackendLesion(lesion),
      window_center: windowCenter,
      window_width: windowWidth,
    }),

  /** Generate an organ preview */
  previewOrgan: (config: { organType: string; huMean: number; huStd: number; enableNoise?: boolean; noiseLevel?: number; enableEnhancement?: boolean; enhancementPattern?: string }) =>
    api.post<PreviewResponse>('/simulation/preview/organ', {
      organ_type: config.organType,
      hu_mean: config.huMean,
      hu_std: config.huStd,
      enable_noise: config.enableNoise ?? true,
      noise_level: config.noiseLevel ?? 0.1,
      enable_enhancement: config.enableEnhancement ?? false,
      enhancement_pattern: config.enhancementPattern ?? 'none',
    }),

  /** Export simulation results */
  exportResults: (
    jobId: string,
    format: 'dicom' | 'nifti' | 'nrrd',
    onDownloadProgress?: (progressEvent: AxiosProgressEvent) => void,
  ): Promise<AxiosResponse<Blob>> =>
    api.download(`/simulation/jobs/${jobId}/export?format=${format}`, {
      onDownloadProgress,
    }),

  /**
   * Generate a CT phantom — either procedural (geometric) or atlas-based
   * (real CT from disk). Returns base64-encoded Float32 volume + optional
   * uint8 label map + metadata.
   *
   * @param source        'procedural' (default) or 'atlas'
   * @param size          Volume max edge size in voxels (64–320, default 192)
   * @param caseId        Atlas case ID, e.g. 's0001' (only used with source='atlas')
   * @param scanDirection Z-axis scan direction: 'head_to_feet' or 'feet_to_head'
   */
  getPhantom: (
    source: 'procedural' | 'atlas' = 'procedural',
    size = 192,
    caseId = 's0001',
    scanDirection: 'head_to_feet' | 'feet_to_head' = 'head_to_feet',
  ): Promise<PhantomResponse> =>
    api.get<PhantomResponse>(
      `/simulation/phantom?source=${source}&size=${size}&case_id=${caseId}&scan_direction=${scanDirection}`,
    ),

  /** Run CT scan parameter simulation preview for the current phantom */
  runCtParamsPreview: (request: CtParamsPreviewRequest): Promise<CtParamsPreviewResponse> =>
    api.post<CtParamsPreviewResponse>('/simulation/ct-params/preview', request),
};

export type { PreviewResponse };
