import { create } from 'zustand';
import type { LesionConfig, OrganConfig, SimulationJob, DeformationConfig } from '@/types/simulation';

/**
 * Simulation Store
 *
 * Global state for lesion and organ simulation modules.
 * Manages active simulation configurations, job status,
 * and generated results.
 */

interface SimulationStore {
  // Active simulation configuration
  activeStudyId: string | null;
  activeSeriesId: string | null;

  // Lesion configurations
  lesions: LesionConfig[];

  // Organ configurations
  organs: OrganConfig[];

  // Deformation
  deformation: DeformationConfig | null;

  // Simulation jobs
  activeJobs: SimulationJob[];
  completedJobs: SimulationJob[];

  // Selection
  selectedLesionId: string | null;
  selectedOrganId: string | null;

  // Actions
  setActiveStudy: (studyId: string) => void;
  setActiveSeries: (seriesId: string) => void;
  addLesion: (lesion: LesionConfig) => void;
  updateLesion: (index: number, lesion: Partial<LesionConfig>) => void;
  removeLesion: (index: number) => void;
  addOrgan: (organ: OrganConfig) => void;
  removeOrgan: (index: number) => void;
  setDeformation: (def: DeformationConfig | null) => void;
  addJob: (job: SimulationJob) => void;
  updateJobStatus: (jobId: string, status: SimulationJob['status'], progress?: number) => void;
  resetSimulation: () => void;
}

export const useSimulationStore = create<SimulationStore>((set) => ({
  activeStudyId: null,
  activeSeriesId: null,
  lesions: [],
  organs: [],
  deformation: null,
  activeJobs: [],
  completedJobs: [],
  selectedLesionId: null,
  selectedOrganId: null,

  setActiveStudy: (studyId) => set({ activeStudyId: studyId }),
  setActiveSeries: (seriesId) => set({ activeSeriesId: seriesId }),

  addLesion: (lesion) =>
    set((state) => ({ lesions: [...state.lesions, lesion] })),

  updateLesion: (index, partial) =>
    set((state) => {
      const updated = [...state.lesions];
      if (updated[index]) {
        updated[index] = { ...updated[index], ...partial };
      }
      return { lesions: updated };
    }),

  removeLesion: (index) =>
    set((state) => ({
      lesions: state.lesions.filter((_, i) => i !== index),
    })),

  addOrgan: (organ) =>
    set((state) => ({ organs: [...state.organs, organ] })),

  removeOrgan: (index) =>
    set((state) => ({
      organs: state.organs.filter((_, i) => i !== index),
    })),

  setDeformation: (deformation) => set({ deformation }),

  addJob: (job) =>
    set((state) => ({ activeJobs: [...state.activeJobs, job] })),

  updateJobStatus: (jobId, status, progress) =>
    set((state) => {
      const updateJob = (job: SimulationJob) =>
        job.id === jobId ? { ...job, status, progress: progress ?? job.progress } : job;

      const activeJobs = state.activeJobs.map(updateJob);
      const completedJobs = status === 'completed' || status === 'failed'
        ? [...state.completedJobs, ...activeJobs.filter((j) => j.status === 'completed' || j.status === 'failed')]
        : state.completedJobs;

      return {
        activeJobs: activeJobs.filter((j) => j.status === 'pending' || j.status === 'running'),
        completedJobs,
      };
    }),

  resetSimulation: () =>
    set({
      lesions: [],
      organs: [],
      deformation: null,
      activeJobs: [],
      selectedLesionId: null,
      selectedOrganId: null,
    }),
}));
