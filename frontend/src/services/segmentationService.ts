import { api } from './api';
import type {
  SegmentationJob,
  SegmentationJobCreate,
  SegmentationModel,
  SegmentationLabel,
  SliceMask,
  InteractiveClickRequest,
  InteractiveClickResponse,
  ExportFormat,
} from '@/types/segmentation';
import type { AxiosResponse } from 'axios';

/**
 * Segmentation Service
 *
 * Service layer for AI-powered organ/lesion segmentation operations.
 * Interfaces with the MONAI backend for:
 * - Automatic organ segmentation
 * - Lesion detection
 * - Interactive click-based refinement
 * - Mask overlay data retrieval
 * - NRRD/NIfTI/DICOM SEG export
 */

export const segmentationService = {
  /** Create a new segmentation job */
  createJob: (config: SegmentationJobCreate) =>
    api.post<SegmentationJob>('/segment/jobs', config),

  /** Get segmentation job status */
  getJobStatus: (jobId: string) =>
    api.get<SegmentationJob>(`/segment/jobs/${jobId}`),

  /** List all segmentation jobs with optional filters */
  listJobs: (studyId?: string, status?: string) => {
    const params: Record<string, string> = {};
    if (studyId) params.study_id = studyId;
    if (status) params.status = status;
    return api.get<SegmentationJob[]>('/segment/jobs', { params });
  },

  /** Cancel a running segmentation job */
  cancelJob: (jobId: string) =>
    api.post<{ status: string; jobId: string }>(`/segment/jobs/${jobId}/cancel`),

  /** Download full 3D mask as NRRD */
  downloadMask: (jobId: string): Promise<AxiosResponse<Blob>> =>
    api.download(`/segment/jobs/${jobId}/mask`),

  /** Get a single 2D slice mask for overlay rendering */
  getSliceMask: (jobId: string, zIndex: number) =>
    api.get<SliceMask>(`/segment/jobs/${jobId}/mask/slice/${zIndex}`),

  /** Export segmentation mask in specified format */
  exportMask: (
    jobId: string,
    format: ExportFormat,
  ): Promise<AxiosResponse<Blob>> =>
    api.download(`/segment/jobs/${jobId}/export?format=${format}`),

  /** Send an interactive click refinement */
  interactiveClick: (request: InteractiveClickRequest) =>
    api.post<InteractiveClickResponse>('/segment/interactive/click', request),

  /** List available segmentation models */
  getModels: () =>
    api.get<SegmentationModel[]>('/segment/models'),

  /** Get label definitions with colors */
  getLabels: (modelName?: string) => {
    const params: Record<string, string> = {};
    if (modelName) params.model_name = modelName;
    return api.get<{ labels: SegmentationLabel[] }>('/segment/labels', { params });
  },
};
