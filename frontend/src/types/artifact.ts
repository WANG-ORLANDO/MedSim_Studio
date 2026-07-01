/** 伪影类型 */
export type ArtifactType =
  | 'metal'
  | 'motion'
  | 'noise'
  | 'ring'
  | 'streak'
  | 'beam_hardening'
  | 'composite';

/** 伪影生成参数 */
export interface ArtifactParams {
  metal?: MetalArtifactParams;
  motion?: MotionArtifactParams;
  noise?: NoiseArtifactParams;
  ring?: RingArtifactParams;
  streak?: StreakArtifactParams;
  beamHardening?: BeamHardeningParams;
}

export interface MetalArtifactParams {
  metalType: 'titanium' | 'stainless_steel' | 'dental_amalgam' | 'gold' | 'copper';
  center: [number, number, number];
  radiusMm: [number, number, number];
  streakIntensity: number;
  beamHardeningStrength: number;
}

export interface MotionArtifactParams {
  motionType: 'respiratory' | 'cardiac' | 'random';
  amplitudeMm: number;
  blurSigma: number;
}

export interface NoiseArtifactParams {
  mAs: number;
  referenceMAs: number;
}

export interface RingArtifactParams {
  numRings: number;
  intensity: number;
}

export interface StreakArtifactParams {
  numStreaks: number;
  intensity: number;
}

export interface BeamHardeningParams {
  cuppingStrength: number;
  darkBandStrength: number;
}

/** 伪影生成请求/响应 */
export interface ArtifactGenerateRequest {
  source: 'phantom' | 'dicom';
  seriesId?: string;
  artifactType: string;
  params: Record<string, unknown>;
  sliceIndex?: number;
}

export interface ArtifactGenerateResponse {
  artifactType: string;
  originalSlice: number[][];
  artifactSlice: number[][];
  maskSlice: number[][];
  metadata: Record<string, unknown>;
  shape: number[];
  spacing: number[];
  source: string;
}

/** 分类请求 */
export interface ClassifyRequest {
  source?: 'phantom' | 'dicom';
  seriesId?: string;
  sliceIndices?: number[];
}

/** 分类响应 (alias) */
export type ClassifyResponse = ArtifactClassificationResult;

/** 伪影分类结果 */
export interface SliceClassifyResult {
  scores: Record<string, number>;
  labels: string[];
  dominant: string;
  sliceIndex: number;
}

export interface ArtifactClassificationResult {
  overallScores: Record<string, number>;
  perSliceScores: SliceClassifyResult[];
  dominantArtifact: string;
  sliceCount: number;
}

/** 伪影修复结果 */
export interface ArtifactRestoreResponse {
  restoredSlice: number[][];
  originalSlice: number[][];
  qualityMetrics: QualityMetrics;
  steps: RestoreStep[];
  report: string;
  shape: number[];
  spacing: number[];
}

export interface QualityMetrics {
  psnr: number;
  ssim: number;
  nmse: number;
  maeHu: number;
  psnrImprovement?: number;
  ssimImprovement?: number;
}

export interface RestoreStep {
  artifactType: string;
  strategy: string;
  psnrDelta: number;
}

/** 完整流水线结果 */
export interface ArtifactPipelineResponse {
  artifactType: string;
  classification: Record<string, unknown>;
  restoration: { steps: RestoreStep[]; report: string };
  originalSlice: number[][];
  artifactSlice: number[][];
  restoredSlice: number[][];
  qualityMetrics: QualityMetrics;
  metadata: Record<string, unknown>;
}

/** 训练请求/响应 */
export interface TrainRequest {
  epochs: number;
  batchSize: number;
  learningRate: number;
  numVolumes: number;
  outputDir: string;
}

export interface TrainEpochResult {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  trainAcc: number;
  valAcc: number;
  trainF1: number;
  valF1: number;
}

export interface TrainStatusResponse {
  status: string;
  currentEpoch: number;
  totalEpochs: number;
  trainLoss: number;
  valLoss: number;
  trainAcc: number;
  valAcc: number;
  trainF1: number;
  valF1: number;
  bestValLoss: number;
  epochHistory: TrainEpochResult[];
  error: string | null;
  startTime: number | null;
}

export interface TrainHistoryResponse {
  epochs: TrainEpochResult[];
  bestValLoss: number;
  outputDir: string;
}

/** CT 序列信息 */
export interface SeriesInfo {
  id: string;
  studyId: string;
  description: string | null;
  modality: string | null;
  imageCount: number | null;
  rows: number | null;
  columns: number | null;
}

/** 作业信息 */
export interface ArtifactJob {
  id: string;
  jobType: string;
  status: string;
  config: Record<string, unknown> | null;
  outputPath: string | null;
  qualityMetrics: Record<string, unknown> | null;
  createdAt: string | null;
  completedAt: string | null;
}
