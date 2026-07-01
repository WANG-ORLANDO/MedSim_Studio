import { create } from 'zustand';
import type { ViewportLayout, MPRViewState, ViewportState, WindowLevelPreset } from '@/types/dicom';

/**
 * Viewer Store
 *
 * Global state management for the DICOM viewer module.
 * Manages viewport layout, MPR state, window/level settings,
 * and active study/series information.
 */

interface ViewerStore {
  // Active study and series
  activeStudyId: string | null;
  activeSeriesId: string | null;

  // Viewport layout
  layout: ViewportLayout[];

  // MPR view states
  mprState: MPRViewState | null;

  // Window/Level
  activePreset: WindowLevelPreset | null;

  // Active tool
  activeTool: string;

  // 3D volume rendering
  volumeRenderingEnabled: boolean;
  volumeOpacity: number;

  // Actions
  setActiveStudy: (studyId: string) => void;
  setActiveSeries: (seriesId: string) => void;
  setLayout: (layout: ViewportLayout[]) => void;
  setMPRState: (state: MPRViewState) => void;
  updateViewport: (viewportId: string, state: Partial<ViewportState>) => void;
  setWindowLevel: (preset: WindowLevelPreset) => void;
  setActiveTool: (tool: string) => void;
  toggleVolumeRendering: () => void;
  setVolumeOpacity: (opacity: number) => void;
  resetViewer: () => void;
}

const initialMPRState: MPRViewState = {
  axial: { windowCenter: 40, windowWidth: 400, rotation: 0, zoom: 1, pan: [0, 0], sliceIndex: 0 },
  sagittal: { windowCenter: 40, windowWidth: 400, rotation: 0, zoom: 1, pan: [0, 0], sliceIndex: 0 },
  coronal: { windowCenter: 40, windowWidth: 400, rotation: 0, zoom: 1, pan: [0, 0], sliceIndex: 0 },
  crosshairPosition: [0, 0, 0],
  linked: true,
};

export const useViewerStore = create<ViewerStore>((set) => ({
  // Initial state
  activeStudyId: null,
  activeSeriesId: null,
  layout: [
    { id: 'axial', type: 'axial', rows: 0, columns: 0 },
    { id: 'sagittal', type: 'sagittal', rows: 0, columns: 0 },
    { id: 'coronal', type: 'coronal', rows: 0, columns: 0 },
    { id: '3d', type: '3d', rows: 0, columns: 0 },
  ],
  mprState: null,
  activePreset: null,
  activeTool: 'windowLevel',
  volumeRenderingEnabled: false,
  volumeOpacity: 0.5,

  // Actions
  setActiveStudy: (studyId) => set({ activeStudyId: studyId }),

  setActiveSeries: (seriesId) => set({ activeSeriesId: seriesId }),

  setLayout: (layout) => set({ layout }),

  setMPRState: (mprState) => set({ mprState }),

  updateViewport: (viewportId, partialState) =>
    set((state) => {
      if (!state.mprState) return state;
      const key = viewportId as keyof MPRViewState;
      if (!(key in state.mprState)) return state;
      return {
        mprState: {
          ...state.mprState,
          [key]: { ...(state.mprState as any)[key], ...partialState },
        },
      };
    }),

  setWindowLevel: (preset) => {
    set({ activePreset: preset });
    // Apply window/level to all active viewports
    set((state) => {
      if (!state.mprState) return state;
      return {
        mprState: {
          ...state.mprState,
          axial: { ...state.mprState.axial, windowCenter: preset.windowCenter, windowWidth: preset.windowWidth },
          sagittal: { ...state.mprState.sagittal, windowCenter: preset.windowCenter, windowWidth: preset.windowWidth },
          coronal: { ...state.mprState.coronal, windowCenter: preset.windowCenter, windowWidth: preset.windowWidth },
        },
      };
    });
  },

  toggleVolumeRendering: () =>
    set((state) => ({ volumeRenderingEnabled: !state.volumeRenderingEnabled })),

  setVolumeOpacity: (volumeOpacity) => set({ volumeOpacity }),

  setActiveTool: (tool) => set({ activeTool: tool }),

  resetViewer: () =>
    set({
      activeStudyId: null,
      activeSeriesId: null,
      mprState: null,
      activePreset: null,
      activeTool: 'windowLevel',
      volumeRenderingEnabled: false,
      volumeOpacity: 0.5,
    }),
}));
