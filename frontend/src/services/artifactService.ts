import { api } from './api';
import type {
  ArtifactGenerateRequest,
  ArtifactGenerateResponse,
  ArtifactClassificationResult,
  ArtifactRestoreResponse,
  ArtifactPipelineResponse,
  ClassifyRequest,
  ClassifyResponse,
  SliceClassifyResult,
  SeriesInfo,
  TrainRequest,
  TrainEpochResult,
  TrainStatusResponse,
  TrainHistoryResponse,
  ArtifactJob,
} from '@/types/artifact';

export type {
  ArtifactGenerateRequest,
  ArtifactGenerateResponse,
  ArtifactClassificationResult,
  ArtifactRestoreResponse,
  ArtifactPipelineResponse,
  ClassifyRequest,
  ClassifyResponse,
  SliceClassifyResult,
  SeriesInfo,
  TrainRequest,
  TrainEpochResult,
  TrainStatusResponse,
  TrainHistoryResponse,
  ArtifactJob,
};

export const artifactService = {
  // ---- Types ----
  async getTypes(): Promise<string[]> {
    const res = await api.get<{ types: string[] }>('/artifact/types');
    return res.types;
  },

  // ---- Series ----
  async getSeries(studyId?: string): Promise<SeriesInfo[]> {
    const params = studyId ? { study_id: studyId } : {};
    return api.get<SeriesInfo[]>('/artifact/series', { params });
  },

  // ---- Generate ----
  async generate(req: ArtifactGenerateRequest): Promise<ArtifactGenerateResponse> {
    return api.generateArtifact<ArtifactGenerateResponse>('/artifact/generate', {
      artifact_type: req.artifactType,
      params: req.params,
      source: req.source ?? 'phantom',
      series_id: req.seriesId,
      slice_index: req.sliceIndex,
    });
  },

  // ---- Classify ----
  async classify(req: ClassifyRequest): Promise<ArtifactClassificationResult> {
    return api.generateArtifact<ArtifactClassificationResult>('/artifact/classify', {
      source: req.source ?? 'phantom',
      series_id: req.seriesId,
      slice_indices: req.sliceIndices,
    });
  },

  // ---- Restore ----
  async restore(req: {
    source?: 'phantom' | 'dicom';
    seriesId?: string;
    artifactTypes: string[];
    method?: 'traditional' | 'deep_learning' | 'auto';
  }): Promise<ArtifactRestoreResponse> {
    return api.generateArtifact<ArtifactRestoreResponse>('/artifact/restore', {
      source: req.source ?? 'phantom',
      series_id: req.seriesId,
      artifact_types: req.artifactTypes,
      method: req.method ?? 'auto',
    });
  },

  // ---- Pipeline (generate → classify → restore) ----
  async pipeline(req: {
    source?: 'phantom' | 'dicom';
    seriesId?: string;
    artifactType: string;
    params?: Record<string, unknown>;
    sliceIndex?: number;
  }): Promise<ArtifactPipelineResponse> {
    return api.generateArtifact<ArtifactPipelineResponse>('/artifact/pipeline', {
      source: req.source ?? 'phantom',
      series_id: req.seriesId,
      artifact_type: req.artifactType,
      params: req.params ?? {},
      slice_index: req.sliceIndex,
    });
  },

  // ---- Jobs ----
  async getJobs(jobType?: string): Promise<ArtifactJob[]> {
    const params = jobType ? { job_type: jobType } : {};
    return api.get<ArtifactJob[]>('/artifact/jobs', { params });
  },

  async getJob(jobId: string): Promise<ArtifactJob> {
    return api.get<ArtifactJob>(`/artifact/jobs/${jobId}`);
  },

  // ---- Training ----
  async startTraining(req: TrainRequest): Promise<{ message: string; epochs: number; outputDir: string }> {
    return api.post('/artifact/train', {
      epochs: req.epochs,
      batch_size: req.batchSize,
      learning_rate: req.learningRate,
      num_volumes: req.numVolumes,
      output_dir: req.outputDir,
    });
  },

  async getTrainStatus(): Promise<TrainStatusResponse> {
    return api.get('/artifact/train/status');
  },

  async getTrainHistory(): Promise<TrainHistoryResponse> {
    return api.get('/artifact/train/history');
  },
};
