import { create } from 'zustand';
import type {
  ArtifactGenerateResponse,
  ArtifactClassificationResult,
  ArtifactRestoreResponse,
  QualityMetrics,
  RestoreStep,
  SeriesInfo,
} from '@/types/artifact';

type ViewMode = 'original' | 'artifact' | 'restored' | 'compare';
type DataSource = 'phantom' | 'dicom';

interface ArtifactState {
  // ---- Source ----
  dataSource: DataSource;
  setDataSource: (source: DataSource) => void;
  seriesList: SeriesInfo[];
  setSeriesList: (list: SeriesInfo[]) => void;
  selectedSeries: string;
  setSelectedSeries: (id: string) => void;

  // ---- Generation ----
  selectedArtifactType: string;
  setSelectedArtifactType: (type: string) => void;
  generateParams: Record<string, unknown>;
  setGenerateParams: (params: Record<string, unknown>) => void;
  generationResult: ArtifactGenerateResponse | null;
  setGenerationResult: (result: ArtifactGenerateResponse | null) => void;

  // ---- Classification ----
  classificationResult: ArtifactClassificationResult | null;
  setClassificationResult: (result: ArtifactClassificationResult | null) => void;

  // ---- Restoration ----
  restoreResult: ArtifactRestoreResponse | null;
  setRestoreResult: (result: ArtifactRestoreResponse | null) => void;

  // ---- View ----
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  windowLevel: number;
  setWindowLevel: (wl: number) => void;
  windowWidth: number;
  setWindowWidth: (ww: number) => void;
  sliceIndex: number;
  setSliceIndex: (idx: number) => void;

  // ---- Loading / Error ----
  loading: boolean;
  setLoading: (v: boolean) => void;
  error: string | null;
  setError: (msg: string | null) => void;

  // ---- Actions ----
  reset: () => void;
}

const initialState = {
  dataSource: 'dicom' as DataSource,
  seriesList: [],
  selectedSeries: '',
  selectedArtifactType: 'metal',
  generateParams: {},
  generationResult: null,
  classificationResult: null,
  restoreResult: null,
  viewMode: 'compare' as ViewMode,
  windowLevel: 40,
  windowWidth: 800,
  sliceIndex: 0,
  loading: false,
  error: null,
};

export const useArtifactStore = create<ArtifactState>((set) => ({
  ...initialState,

  setDataSource: (dataSource) => set({ dataSource, generationResult: null, classificationResult: null, restoreResult: null }),
  setSeriesList: (seriesList) => set({ seriesList }),
  setSelectedSeries: (selectedSeries) => set({ selectedSeries, generationResult: null }),

  setSelectedArtifactType: (selectedArtifactType) => set({ selectedArtifactType, generationResult: null }),
  setGenerateParams: (generateParams) => set({ generateParams }),
  setGenerationResult: (generationResult) => set({ generationResult, error: null }),

  setClassificationResult: (classificationResult) => set({ classificationResult }),

  setRestoreResult: (restoreResult) => set({ restoreResult }),

  setViewMode: (viewMode) => set({ viewMode }),
  setWindowLevel: (windowLevel) => set({ windowLevel }),
  setWindowWidth: (windowWidth) => set({ windowWidth }),
  setSliceIndex: (sliceIndex) => set({ sliceIndex }),

  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),

  reset: () => set(initialState),
}));
