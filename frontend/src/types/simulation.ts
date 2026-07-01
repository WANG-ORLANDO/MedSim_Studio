/**
 * Simulation Type Definitions
 *
 * Types for lesion/organ simulation and Hounsfield Unit (HU)
 * manipulation within the CT Simulator platform.
 */

/** Lesion types available for simulation */
export type LesionType = 'tumor' | 'nodule' | 'cyst' | 'calcification' | 'metastasis';

/** Geometric shape for simulated lesions */
export type LesionShape = 'spherical' | 'ellipsoidal' | 'irregular' | 'lobulated' | 'spiculated';

/** Lesion parameters for simulation generation */
export interface LesionConfig {
  type: LesionType;
  shape: LesionShape;
  center: [number, number, number]; // voxel coordinates
  radiusMm: [number, number, number]; // radii in mm (x, y, z)
  huMean: number; // mean Hounsfield Unit value
  huStd: number; // HU standard deviation (heterogeneity)
  marginSharpness: number; // 0 = diffuse, 1 = sharp
  calcificationFraction: number; // 0-1, internal calcification
  necrosisFraction: number; // 0-1, internal necrosis
  spiculationDegree: number; // 0-1, spiculation for malignant appearance
}

/** Organ simulation parameters */
export interface OrganConfig {
  organType: OrganType;
  huMean: number;
  huStd: number;
  enableNoise: boolean;
  noiseLevel: number;
  enableEnhancement: boolean;
  enhancementPattern: EnhancementPattern;
}

export type OrganType =
  | 'liver'
  | 'kidney'
  | 'lung'
  | 'brain'
  | 'bone'
  | 'heart'
  | 'spleen'
  | 'pancreas'
  | 'bladder';

export type EnhancementPattern = 'homogeneous' | 'heterogeneous' | 'rim' | 'septal' | 'none';

/** Deformation field parameters */
export interface DeformationConfig {
  deformationType: 'rigid' | 'affine' | 'bspline' | 'demons';
  magnitude: number;
  controlPoints: [number, number, number][];
  smoothingSigma: number;
}

/** HU modifier for adjusting tissue densities */
export interface HUModifier {
  regionId: string;
  originalHU: number;
  modifiedHU: number;
  maskVolume: number; // cm鲁
  operation: 'add' | 'subtract' | 'replace' | 'scale';
}

/** Full simulation job configuration */
export interface SimulationJob {
  id: string;
  studyId: string;
  seriesId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  lesions: LesionConfig[];
  organs: OrganConfig[];
  deformation: DeformationConfig | null;
  outputFormat: 'dicom' | 'nifti' | 'nrrd';
  createdAt: string;
  progress: number;
}

export type CtDoseLevel = 'low' | 'standard' | 'high';
export type CtKernel = 'smooth' | 'soft' | 'standard' | 'lung' | 'bone' | 'sharp';
export type CtContrastPhase = 'noncontrast' | 'arterial' | 'venous' | 'delayed';
export type CtScanDirection = 'head_to_feet' | 'feet_to_head';
export type CtPhantomSource = 'procedural' | 'atlas';
export type CtSliceThickness = 0.625 | 1.0 | 2.5 | 5.0 | 10.0;
export type CtKvp = 80 | 100 | 120 | 140;
export type CtPitch = 0.5 | 0.8 | 1.0 | 1.2 | 1.5;
export type CtFovMm = 150 | 250 | 350 | 500;
export type CtMatrixSize = 256 | 512 | 1024;

export interface CtParamsPreviewParams {
  sliceThicknessMm: CtSliceThickness;
  doseLevel: CtDoseLevel;
  mAs: number;
  kVp: CtKvp;
  pitch: CtPitch;
  fovMm: CtFovMm;
  matrixSize: CtMatrixSize;
  kernel: CtKernel;
  contrastPhase: CtContrastPhase;
}

export interface CtParamsPreviewRequest {
  source: CtPhantomSource;
  caseId?: string | null;
  size: number;
  scanDirection: CtScanDirection;
  params: CtParamsPreviewParams;
}

export interface CtCenterSliceStats {
  sliceIndex: number;
  min: number;
  max: number;
  mean: number;
  std: number;
}

export interface CtParamsPreviewMetadata {
  shape: number[];
  spacing: number[];
  huRange: [number, number];
  effectiveSliceThicknessMm: number;
  algorithmNotes?: string[];
  warnings?: string[];
  source?: CtPhantomSource;
  caseId?: string;
  scanDirection?: CtScanDirection;
  previewStats?: {
    originalCenterSliceStats?: CtCenterSliceStats;
    simulatedCenterSliceStats?: CtCenterSliceStats;
  };
  phantomMetadata?: {
    originalShape?: number[];
    outputShape?: number[];
    originalSpacing?: number[];
    outputSpacing?: number[];
    flippedZ?: boolean;
  };
}

export interface CtParamsJson {
  requestedParams: Record<string, unknown>;
  resolvedParams: Record<string, unknown>;
  algorithmSteps: Array<Record<string, unknown>>;
  approximationNotes?: string[];
  warnings?: string[];
  inputShape?: number[];
  outputShape?: number[];
  inputSpacing?: number[];
  outputSpacing?: number[];
  huRangeBefore?: [number, number];
  huRangeAfter?: [number, number];
}

export interface StandardizedCtCaseVolume {
  encoding: 'base64';
  dtype: 'float32';
  byteOrder: 'little_endian';
  axisOrder: 'zyx';
  shape: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
  direction: [
    [number, number, number],
    [number, number, number],
    [number, number, number],
  ];
  huRange: [number, number];
  sliceCount: number;
  modality: 'CT';
  bodyPart: string;
  imageKind: 'simulated_ct';
  imageDataField: 'simulatedVolumeBase64';
}
export interface StandardizedCtCaseSimulation {
  type: 'ct_scan_params';
  paramsJson: CtParamsJson;
  algorithm: 'image_domain_approximation';
  approximationWarning: string;
}
export interface StandardizedCtCase {
  caseId?: string | null;
  source: CtPhantomSource;
  sourceCaseId: string | null;
  volume: StandardizedCtCaseVolume;
  simulation: StandardizedCtCaseSimulation;
  metadata: CtParamsPreviewMetadata;
}
export interface CtParamsPreviewResponse {
  simulatedVolumeBase64: string;
  metadata: CtParamsPreviewMetadata;
  paramsJson: CtParamsJson;
  standardizedCase: StandardizedCtCase;
}

