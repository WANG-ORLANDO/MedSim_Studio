import { api } from './api';
import type { DicomStudy, DicomSeries, DicomInstance, PaginatedResponse } from '@/types/index';

/**
 * DICOM Service
 *
 * Service layer for DICOM study/series management operations.
 * All methods return the backend response body directly.
 */

export const dicomService = {
  /** Fetch paginated list of all studies */
  getStudies: (page = 1, pageSize = 20) =>
    api.get<PaginatedResponse<DicomStudy>>(
      `/dicom/studies?page=${page}&page_size=${pageSize}`
    ),

  /** Get single study details by ID */
  getStudy: (studyId: string) =>
    api.get<DicomStudy>(`/dicom/studies/${studyId}`),

  /** Get series within a study */
  getSeries: (studyId: string) =>
    api.get<DicomSeries[]>(`/dicom/studies/${studyId}/series`),

  /** Get single series details */
  getSeriesDetail: (studyId: string, seriesId: string) =>
    api.get<DicomSeries>(`/dicom/studies/${studyId}/series/${seriesId}`),

  /** Get instances within a series (sorted by instance number) */
  getInstances: (seriesId: string) =>
    api.get<DicomInstance[]>(`/dicom/series/${seriesId}/instances`),

  /** Upload DICOM files */
  uploadDicom: (files: File[], studyId?: string) =>
    api.uploadDicom(files, studyId),

  /** Delete a study */
  deleteStudy: (studyId: string) =>
    api.delete<{ message: string }>(`/dicom/studies/${studyId}`),
};
