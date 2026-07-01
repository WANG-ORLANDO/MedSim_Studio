import { useState, useCallback } from 'react';
import { dicomService } from '@services/index';
import type { DicomStudy, DicomSeries } from '@/types/index';

/**
 * Hook for DICOM data loading
 *
 * Manages async loading of DICOM studies and series
 * with loading/error states for UI feedback.
 */

interface UseDicomLoaderResult {
  studies: DicomStudy[];
  series: DicomSeries[];
  isLoading: boolean;
  error: string | null;
  loadStudies: () => Promise<void>;
  loadSeries: (studyId: string) => Promise<void>;
}

export function useDicomLoader(): UseDicomLoaderResult {
  const [studies, setStudies] = useState<DicomStudy[]>([]);
  const [series, setSeries] = useState<DicomSeries[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadStudies = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await dicomService.getStudies();
      setStudies(response.items);
    } catch (err: any) {
      setError(err.message || 'Failed to load studies');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const loadSeries = useCallback(async (studyId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await dicomService.getSeries(studyId);
      setSeries(response);
    } catch (err: any) {
      setError(err.message || 'Failed to load series');
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { studies, series, isLoading, error, loadStudies, loadSeries };
}
