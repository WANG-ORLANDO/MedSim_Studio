import { create } from 'zustand';
import type {
  SegmentationJob,
  SegmentationModel,
  SegmentationLabel,
} from '@/types/segmentation';
import { segmentationService } from '@/services/segmentationService';

/**
 * Persisted mask data that survives page navigation.
 */
interface PersistedMaskData {
  jobId: string;
  seriesId: string;
  maskData: Float32Array | null;
  labels: SegmentationLabel[];
}

/**
 * Segmentation Store
 *
 * Global state for AI-powered segmentation module.
 * Manages active segmentation jobs, model/label lists,
 * and organ selection for targeting.
 */

interface SegmentationStore {
  // Active segmentation study/series
  activeStudyId: string | null;
  activeSeriesId: string | null;

  // Available models and labels (fetched from backend)
  models: SegmentationModel[];
  labels: SegmentationLabel[];
  modelsLoading: boolean;
  labelsLoading: boolean;

  // Organ/lesion selection for segmentation targeting
  selectedOrgans: string[];
  detectLesions: boolean;

  // Segmentation jobs
  activeJobs: SegmentationJob[];
  completedJobs: SegmentationJob[];

  // Polling
  pollIntervalId: number | null;

  // Persisted state (survives page navigation)
  persistedMask: PersistedMaskData | null;
  persistedSelectedModel: string;

  // Actions
  setActiveStudy: (studyId: string) => void;
  setActiveSeries: (seriesId: string) => void;
  fetchModels: () => Promise<void>;
  fetchLabels: (modelName?: string) => Promise<void>;
  toggleOrgan: (organ: string) => void;
  setDetectLesions: (enabled: boolean) => void;
  clearTargets: () => void;
  createJob: (config: {
    studyId: string;
    seriesId: string;
    modelName: string;
    targetOrgans: string[];
    detectLesions: boolean;
  }) => Promise<SegmentationJob | null>;
  updateJobStatus: (jobId: string, status: SegmentationJob['status'], progress?: number, errorMessage?: string) => void;
  startPolling: (jobId: string, onUpdate?: (job: SegmentationJob) => void) => void;
  stopPolling: () => void;
  cancelJob: (jobId: string) => Promise<void>;
  removeJob: (jobId: string) => void;
  setPersistedMask: (data: PersistedMaskData) => void;
  clearPersistedMask: () => void;
  setPersistedSelectedModel: (model: string) => void;
  reset: () => void;
}

const POLL_INTERVAL_MS = 2000;

export const useSegmentationStore = create<SegmentationStore>((set, get) => ({
  activeStudyId: null,
  activeSeriesId: null,
  models: [],
  labels: [],
  modelsLoading: false,
  labelsLoading: false,
  selectedOrgans: [],
  detectLesions: false,
  activeJobs: [],
  completedJobs: [],
  pollIntervalId: null,
  persistedMask: null,
  persistedSelectedModel: 'unet',

  setActiveStudy: (studyId) => set({ activeStudyId: studyId }),
  setActiveSeries: (seriesId) => set({ activeSeriesId: seriesId }),

  fetchModels: async () => {
    set({ modelsLoading: true });
    try {
      const models = await segmentationService.getModels();
      set({ models, modelsLoading: false });
    } catch {
      set({ modelsLoading: false });
    }
  },

  fetchLabels: async (modelName?: string) => {
    set({ labelsLoading: true });
    try {
      const response = await segmentationService.getLabels(modelName);
      set({ labels: response.labels, labelsLoading: false });
    } catch {
      set({ labelsLoading: false });
    }
  },

  toggleOrgan: (organ) =>
    set((state) => {
      const exists = state.selectedOrgans.includes(organ);
      return {
        selectedOrgans: exists
          ? state.selectedOrgans.filter((o) => o !== organ)
          : [...state.selectedOrgans, organ],
      };
    }),

  setDetectLesions: (enabled) => set({ detectLesions: enabled }),

  clearTargets: () => set({ selectedOrgans: [], detectLesions: false }),

  createJob: async (config) => {
    try {
      const createPayload = {
        studyId: config.studyId,
        seriesId: config.seriesId,
        modelName: config.modelName,
        targetOrgans: config.targetOrgans,
        detectLesions: config.detectLesions,
      };
      const job = await segmentationService.createJob(createPayload);
      set((state) => ({
        activeJobs: [...state.activeJobs, job],
      }));
      return job;
    } catch {
      return null;
    }
  },

  updateJobStatus: (jobId, status, progress, errorMessage) =>
    set((state) => {
      const updateJob = (job: SegmentationJob): SegmentationJob => ({
        ...job,
        status,
        progress: progress ?? job.progress,
        errorMessage: errorMessage ?? job.errorMessage,
      });

      const activeJobs = state.activeJobs.map((j) =>
        j.id === jobId ? updateJob(j) : j,
      );

      // Move completed/failed jobs to completedJobs list
      const finishedJobs = activeJobs.filter(
        (j) => j.status === 'completed' || j.status === 'failed',
      );

      return {
        activeJobs: activeJobs.filter(
          (j) => j.status === 'pending' || j.status === 'running',
        ),
        completedJobs: [
          ...state.completedJobs.filter((j) => j.id !== jobId),
          ...finishedJobs.filter((j) => j.id === jobId),
        ],
      };
    }),

  startPolling: (jobId, onUpdate) => {
    // Clear any existing polling
    const existing = get().pollIntervalId;
    if (existing !== null) {
      clearInterval(existing);
    }

    const id = window.setInterval(async () => {
      try {
        const job = await segmentationService.getJobStatus(jobId);
        get().updateJobStatus(job.id, job.status, job.progress, job.errorMessage ?? undefined);
        onUpdate?.(job);

        // Stop polling if job reached terminal state
        if (job.status === 'completed' || job.status === 'failed') {
          get().stopPolling();
        }
      } catch {
        // Silently retry on next interval
      }
    }, POLL_INTERVAL_MS);

    set({ pollIntervalId: id });
  },

  stopPolling: () => {
    const id = get().pollIntervalId;
    if (id !== null) {
      clearInterval(id);
      set({ pollIntervalId: null });
    }
  },

  cancelJob: async (jobId) => {
    try {
      await segmentationService.cancelJob(jobId);
      get().updateJobStatus(jobId, 'failed', undefined, 'Cancelled by user');
    } catch {
      // Swallow error — job may have already completed
    }
  },

  removeJob: (jobId) =>
    set((state) => ({
      activeJobs: state.activeJobs.filter((j) => j.id !== jobId),
      completedJobs: state.completedJobs.filter((j) => j.id !== jobId),
    })),

  setPersistedMask: (data) => set({ persistedMask: data }),

  clearPersistedMask: () => set({ persistedMask: null }),

  setPersistedSelectedModel: (model) => set({ persistedSelectedModel: model }),

  reset: () => {
    get().stopPolling();
    set({
      activeStudyId: null,
      activeSeriesId: null,
      selectedOrgans: [],
      detectLesions: false,
      activeJobs: [],
      completedJobs: [],
    });
  },
}));
