import { useEffect, useState, useCallback, useRef } from 'react';
import { Button } from '@components/ui/button';
import { dicomService } from '@services/index';
import { useSegmentationStore } from '@store/useSegmentationStore';
import { LabelToggleList } from '@segmentation/components/LabelToggleList';
import { JobProgressPanel } from '@segmentation/components/JobProgressPanel';
import { ExportDialog } from '@segmentation/components/ExportDialog';
import { BrushToolToggle } from '@segmentation/components/BrushToolToggle';
import { useSegmentation } from '@segmentation/hooks/useSegmentation';
import { CornerstoneViewport } from '@viewer/cornerstone/viewport/CornerstoneViewport';
import { VolumeRenderer } from '@vtk/volumeRendering/VolumeRenderer';
import {
  createToolGroup,
  setActiveTool as setActiveToolInGroup,
} from '@viewer/cornerstone/toolGroups';
import { initCornerstone } from '@viewer/cornerstone/initCornerstone';
import { fetchImageIdsForSeries } from '@viewer/cornerstone/loadDicomSeries';
import { volumeLoader, imageLoader } from '@cornerstonejs/core';
const { createAndCacheVolumeFromImages } = volumeLoader;
import type { DicomStudy, DicomSeries, SegmentationJob, SegmentationModel, SegmentationLabel } from '@/types/index';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const VOLUME_ID_PREFIX = 'seg-volume';

const VIEWPORTS = [
  { id: 'seg-axial', orientation: 'axial' as const, label: 'A' },
  { id: 'seg-sagittal', orientation: 'sagittal' as const, label: 'S' },
  { id: 'seg-coronal', orientation: 'coronal' as const, label: 'C' },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function SegmentationPage() {
  // ---- Backend data state ----
  const [studies, setStudies] = useState<DicomStudy[]>([]);
  const [seriesList, setSeriesList] = useState<DicomSeries[]>([]);
  const [selectedStudyId, setSelectedStudyId] = useState<string>(
    () => useSegmentationStore.getState().activeStudyId || '',
  );
  const [selectedSeriesId, setSelectedSeriesId] = useState<string>(
    () => useSegmentationStore.getState().activeSeriesId || '',
  );
  const [studiesLoading, setStudiesLoading] = useState(true);

  // ---- Cornerstone state ----
  const [cornerstoneReady, setCornerstoneReady] = useState(false);
  const [volumeReady, setVolumeReady] = useState(false);
  const [viewerMode, setViewerMode] = useState<'mpr' | '3d'>('mpr');
  const [imageIds, setImageIds] = useState<string[]>([]);
  const [, setTotalSlices] = useState(0);

  // ---- Model/label state ----
  const {
    models,
    labels,
    selectedOrgans,
    detectLesions,
    modelsLoading,
    fetchModels,
    fetchLabels,
    toggleOrgan,
    setDetectLesions,
    createJob,
    activeJobs,
    completedJobs,
    updateJobStatus,
    startPolling,
    stopPolling,
    cancelJob,
  } = useSegmentationStore();

  const [selectedModel, setSelectedModel] = useState(
    () => useSegmentationStore.getState().persistedSelectedModel || 'unet'
  );
  const persisted = useSegmentationStore((s) => s.persistedMask);
  const [jobRunning, setJobRunning] = useState(false);
  const [activeJob, setActiveJob] = useState<SegmentationJob | null>(null);

  // ---- Volume ID (shared across MPR viewports) ----
  const volumeId = selectedSeriesId
    ? `${VOLUME_ID_PREFIX}-${selectedSeriesId}`
    : undefined;

  // ---- Cornerstone3D Segmentation ----
  const {
    segState: csSegState,
    loading: segLoading,
    loadMask,
    restoreMaskFromCache,
    syncVisibilityFromSet,
    clearMask,
    maskData,
  } = useSegmentation(volumeId);

  // ---- Organ label visibility ----
  const [visibleLabels, setVisibleLabels] = useState<Set<number>>(new Set([1, 2, 3, 4, 5, 6, 7, 8, 9]));
  const [brushActive, setBrushActive] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  // Track which series we last restored from cache — prevents double-restore
  const cachedSeriesRef = useRef<string | null>(null);

  // ==================================================================
  // Init Cornerstone3D once on mount
  // ==================================================================
  useEffect(() => {
    initCornerstone()
      .then(() => {
        createToolGroup();
        setCornerstoneReady(true);
      })
      .catch((err) => console.error('[SegmentationPage] Cornerstone init failed:', err));
  }, []);

  // ==================================================================
  // Fetch studies list + models/labels on mount
  // ==================================================================
  useEffect(() => {
    const init = async () => {
      try {
        const paginated = await dicomService.getStudies(1, 50);
        setStudies(paginated.items ?? []);
      } catch (err) {
        console.error('[SegmentationPage] Failed to load studies:', err);
      } finally {
        setStudiesLoading(false);
      }
    };
    init();
    fetchModels();
    // Initial label fetch uses the default model ('unet');
    // refetch happens automatically in the model-change effect below.
    fetchLabels();
  }, [fetchModels, fetchLabels]);

  // ==================================================================
  // Persist model selection so it survives page navigation
  // ==================================================================
  useEffect(() => {
    useSegmentationStore.getState().setPersistedSelectedModel(selectedModel);
  }, [selectedModel]);

  // ==================================================================
  // Persist active study/series to store so they survive tab switches
  // ==================================================================
  useEffect(() => {
    if (selectedStudyId) useSegmentationStore.getState().setActiveStudy(selectedStudyId);
  }, [selectedStudyId]);

  useEffect(() => {
    if (selectedSeriesId) useSegmentationStore.getState().setActiveSeries(selectedSeriesId);
  }, [selectedSeriesId]);

  // ==================================================================
  // Refetch labels when model selection changes
  // ==================================================================
  useEffect(() => {
    fetchLabels(selectedModel);
  }, [selectedModel, fetchLabels]);

  // ==================================================================
  // Update visible labels when model changes
  // ==================================================================
  useEffect(() => {
    if (selectedModel === 'totalsegmentator') {
      // Enable key organs by default: lungs, heart, liver, spleen,
      // kidneys, bladder, pancreas, stomach, aorta, spine
      setVisibleLabels(new Set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 30, 31, 34]));
    } else {
      setVisibleLabels(new Set([1, 2, 3, 4, 5, 6, 7, 8, 9]));
    }
  }, [selectedModel]);

  // ==================================================================
  // Sync visibleLabels → Cornerstone3D segment visibility
  // ==================================================================
  useEffect(() => {
    if (csSegState?.loaded) {
      syncVisibilityFromSet(visibleLabels);
    }
  }, [visibleLabels, csSegState?.loaded, syncVisibilityFromSet]);

  // ==================================================================
  // Clear segmentation mask when model or series changes.
  //
  // IMPORTANT: Skips on first mount if there is cached segmentation
  // data (surviving a tab switch), so the restoration effect below
  // can re-create the overlay without a wasteful clear-redraw cycle.
  // ==================================================================
  const prevModelRef = useRef(selectedModel);
  useEffect(() => {
    const store = useSegmentationStore.getState();

    // Skip on mount if we have cached data to restore
    if (store.persistedMask && store.persistedMask.seriesId === selectedSeriesId) {
      return;
    }

    clearMask();

    // Reset the cached-series guard so switching back will re-restore
    if (cachedSeriesRef.current && cachedSeriesRef.current !== selectedSeriesId) {
      cachedSeriesRef.current = null;
    }

    // On model change: also clear persisted mask because a different
    // model produces different label indices / names
    if (prevModelRef.current !== selectedModel) {
      store.clearPersistedMask();
      cachedSeriesRef.current = null;
    }
    prevModelRef.current = selectedModel;
  }, [selectedModel, selectedSeriesId, clearMask]);

  // ==================================================================
  // Load series when study changes
  // ==================================================================
  useEffect(() => {
    if (!selectedStudyId) {
      setSeriesList([]);
      setSelectedSeriesId('');
      useSegmentationStore.getState().clearPersistedMask();
      cachedSeriesRef.current = null;
      return;
    }

    // Clear persisted mask when study changes — different anatomy
    useSegmentationStore.getState().clearPersistedMask();
    cachedSeriesRef.current = null;

    let mounted = true;
    const loadSeries = async () => {
      try {
        const allSeries = await dicomService.getSeries(selectedStudyId);
        const imageModalities = ['CT', 'MR', 'PT', 'NM', 'US'];
        const filtered = allSeries.filter((s) => imageModalities.includes(s.modality));
        if (!mounted) return;
        setSeriesList(filtered);
        if (filtered.length > 0) {
          setSelectedSeriesId(filtered[0].id);
        }
      } catch (err) {
        console.error('[SegmentationPage] Failed to load series:', err);
      }
    };
    loadSeries();
    return () => { mounted = false; };
  }, [selectedStudyId]);

  // ==================================================================
  // Pre-create 3D volume when series is selected
  // ==================================================================
  useEffect(() => {
    if (!cornerstoneReady || !selectedSeriesId) return;

    let cancelled = false;
    const prepareVolume = async () => {
      try {
        setVolumeReady(false);
        const ids = await fetchImageIdsForSeries(selectedSeriesId);
        if (cancelled) return;
        setImageIds(ids);
        setTotalSlices(ids.length);

        await Promise.all(imageLoader.loadAndCacheImages(ids));
        if (cancelled) return;

        const volId = `${VOLUME_ID_PREFIX}-${selectedSeriesId}`;
        await createAndCacheVolumeFromImages(volId, ids);
        if (cancelled) return;
        setVolumeReady(true);
      } catch (err) {
        console.error('[SegmentationPage] Volume creation failed:', err);
      }
    };
    prepareVolume();
    return () => { cancelled = true; };
  }, [cornerstoneReady, selectedSeriesId]);

  // ==================================================================
  // Handle Run Segmentation
  // ==================================================================
  const handleRunSegmentation = useCallback(async () => {
    if (!selectedStudyId || !selectedSeriesId) return;

    setJobRunning(true);
    try {
      const job = await createJob({
        studyId: selectedStudyId,
        seriesId: selectedSeriesId,
        modelName: selectedModel,
        targetOrgans: selectedOrgans,
        detectLesions,
      });

      if (job) {
        setActiveJob(job);
        startPolling(job.id, (updatedJob) => {
          if (updatedJob.status === 'completed' || updatedJob.status === 'failed') {
            setJobRunning(false);
            setActiveJob(updatedJob);
          }
        });
      }
    } catch {
      setJobRunning(false);
    }
  }, [selectedStudyId, selectedSeriesId, selectedModel, selectedOrgans, detectLesions, createJob, startPolling]);

  // ==================================================================
  // Compute which completed job is current
  // ==================================================================
  const completedJob = completedJobs.length > 0 ? completedJobs[completedJobs.length - 1] : null;

  // ==================================================================
  // Load mask into Cornerstone3D when a job completes
  // ==================================================================
  useEffect(() => {
    if (completedJob?.id && completedJob.status === 'completed' && volumeReady && volumeId) {
      const maskLabels = labels.length > 0 ? labels : (persisted?.labels ?? DEFAULT_LABELS);
      loadMask(completedJob.id, maskLabels, completedJob.seriesId);
    }
  }, [completedJob?.id, completedJob?.status, volumeReady, volumeId, loadMask, labels, persisted]);

  // ==================================================================
  // Restore segmentation mask from cache when navigating back
  //
  // When the user switches away from the Segmentation page and comes
  // back, the component re-mounts but the mask data (Float32Array)
  // survives in zustand store. This effect re-creates the Cornerstone3D
  // labelmap volume from that cached data — no backend download needed.
  // ==================================================================
  useEffect(() => {
    if (!volumeReady || !volumeId) return;

    const p = useSegmentationStore.getState().persistedMask;

    // Only restore if the cached mask belongs to the current series
    // and we haven't already restored it this mount cycle
    if (p?.maskData && p.seriesId === selectedSeriesId && cachedSeriesRef.current !== p.seriesId) {
      cachedSeriesRef.current = p.seriesId;
      restoreMaskFromCache({
        jobId: p.jobId,
        maskData: p.maskData,
        labels: p.labels.length > 0 ? p.labels : DEFAULT_LABELS,
      });
    }
  }, [volumeReady, volumeId, selectedSeriesId, restoreMaskFromCache]);

  // ==================================================================
  // Handle cancel
  // ==================================================================
  const handleCancel = useCallback(async (jobId: string) => {
    await cancelJob(jobId);
    setJobRunning(false);
    setActiveJob(null);
    clearMask();
    useSegmentationStore.getState().clearPersistedMask();
  }, [cancelJob, clearMask]);

  // ==================================================================
  // Handle label visibility toggle
  // ==================================================================
  const toggleLabelVisibility = useCallback((labelIndex: number) => {
    setVisibleLabels((prev) => {
      const next = new Set(prev);
      if (next.has(labelIndex)) next.delete(labelIndex);
      else next.add(labelIndex);
      return next;
    });
  }, []);

  // ==================================================================
  // Handle viewport ready
  // ==================================================================
  const handleViewportReady = useCallback((viewportId: string) => {
    console.info(`[SegmentationPage] Viewport ready: ${viewportId}`);
  }, []);

  // ==================================================================
  // Render
  // ==================================================================
  return (
    <div className="flex h-full flex-col">
      {/* ---- Toolbar ---- */}
      <div className="flex flex-wrap items-center gap-3 border-b border-border bg-card px-4 py-2">
        {/* Study selector */}
        <select
          value={selectedStudyId}
          onChange={(e) => setSelectedStudyId(e.target.value)}
          className="rounded border border-border bg-background px-2 py-1 text-xs max-w-[180px]"
          disabled={jobRunning}
        >
          <option value="">Select study...</option>
          {studies.map((s) => (
            <option key={s.id} value={s.id}>
              {s.patientName || s.id.substring(0, 8)}
            </option>
          ))}
        </select>

        {/* Series selector */}
        {seriesList.length > 0 && (
          <select
            value={selectedSeriesId}
            onChange={(e) => setSelectedSeriesId(e.target.value)}
            className="rounded border border-border bg-background px-2 py-1 text-xs max-w-[180px]"
            disabled={jobRunning}
          >
            {seriesList.map((s) => (
              <option key={s.id} value={s.id}>
                {s.seriesDescription || `Series ${s.seriesNumber}`}
              </option>
            ))}
          </select>
        )}

        <div className="h-5 w-px bg-border" />

        {/* Model selector */}
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="rounded border border-border bg-background px-2 py-1 text-xs"
          disabled={jobRunning}
        >
          {models.map((m) => (
            <option key={m.name} value={m.name}>
              {m.name} — {m.description.substring(0, 40)}
            </option>
          ))}
        </select>

        <div className="h-5 w-px bg-border" />

        {/* Lesion detection toggle */}
        <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer">
          <input
            type="checkbox"
            checked={detectLesions}
            onChange={(e) => setDetectLesions(e.target.checked)}
            disabled={jobRunning}
            className="h-3.5 w-3.5 accent-primary"
          />
          Lesions
        </label>

        {/* Run button */}
        <Button
          variant="default"
          size="sm"
          onClick={handleRunSegmentation}
          disabled={!selectedSeriesId || !volumeReady || jobRunning}
        >
          {jobRunning ? 'Segmenting...' : 'Run Segmentation'}
        </Button>

        {/* 2D/3D toggle */}
        <Button
          variant={viewerMode === '3d' ? 'secondary' : 'ghost'}
          size="sm"
          onClick={() => setViewerMode((v) => (v === 'mpr' ? '3d' : 'mpr'))}
          disabled={!volumeReady}
        >
          {viewerMode === 'mpr' ? '3D' : '2D'}
        </Button>

        {/* Brush tool toggle */}
        {completedJob && (
          <>
            <div className="h-5 w-px bg-border" />
            <BrushToolToggle
              active={brushActive}
              onToggle={() => setBrushActive((b) => !b)}
              disabled={!completedJob}
            />
          </>
        )}

        {/* Export button */}
        {completedJob && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => setExportOpen(true)}
          >
            Export
          </Button>
        )}
      </div>

      {/* ---- Main content: sidebar + viewport ---- */}
      <div className="flex flex-1 overflow-hidden">
        {/* ---- Left sidebar: config + labels + status ---- */}
        <div className="w-64 flex-shrink-0 overflow-y-auto border-r border-border bg-card p-4 space-y-4">
          {/* Organ labels */}
          <div>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Target Organs
            </h3>
            {modelsLoading ? (
              <p className="text-xs text-muted-foreground">Loading...</p>
            ) : (
              <LabelToggleList
                labels={labels.length > 0 ? labels : DEFAULT_LABELS}
                enabledLabels={visibleLabels}
                onToggle={(idx) => {
                  toggleOrgan(
                    labels.find((l) => l.index === idx)?.name ?? '',
                  );
                  toggleLabelVisibility(idx);
                }}
              />
            )}
          </div>

          {/* Segmentation loading indicator */}
          {segLoading && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <div className="h-3 w-3 animate-spin rounded-full border border-current border-t-transparent" />
              Loading segmentation mask...
            </div>
          )}

          {/* Job status */}
          {activeJob && (
            <div>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Current Job
              </h3>
              <JobProgressPanel job={activeJob} onCancel={handleCancel} />
            </div>
          )}

          {/* Completed job quick info */}
          {completedJob && !activeJob && (
            <div>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Last Result
              </h3>
              <JobProgressPanel job={completedJob} />
            </div>
          )}
        </div>

        {/* ---- Main viewport area ---- */}
        <div className="flex-1">
          {!cornerstoneReady ? (
            <div className="flex h-full items-center justify-center bg-black">
              <div className="flex flex-col items-center gap-3">
                <div className="h-8 w-8 animate-spin rounded-full border-2 border-white/20 border-t-white" />
                <p className="text-sm text-white/50">Initializing rendering engine...</p>
              </div>
            </div>
          ) : !selectedSeriesId || !volumeReady ? (
            <div className="flex h-full items-center justify-center bg-black">
              <div className="flex flex-col items-center gap-3">
                {studiesLoading ? (
                  <>
                    <div className="h-8 w-8 animate-spin rounded-full border-2 border-white/20 border-t-white" />
                    <p className="text-sm text-white/50">Loading studies...</p>
                  </>
                ) : (
                  <>
                    <svg className="h-12 w-12 text-white/20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1}>
                      <path d="M12 2L2 7l10 5 10-5-10-5z" />
                      <path d="M2 17l10 5 10-5" />
                      <path d="M2 12l10 5 10-5" />
                    </svg>
                    <p className="text-sm text-white/40">
                      {selectedStudyId ? 'Building 3D volume...' : 'Select a study above'}
                    </p>
                  </>
                )}
              </div>
            </div>
          ) : viewerMode === '3d' ? (
            <VolumeRenderer
              mode="dicom"
              seriesId={selectedSeriesId}
              showControls
              opacityPreset="ct-soft-tissue"
              segmentationMask={maskData ?? persisted?.maskData ?? null}
              segmentationLabels={(labels.length > 0 ? labels : persisted?.labels) ?? DEFAULT_LABELS}
            />
          ) : (
            <div className="grid h-full w-full grid-cols-2 grid-rows-[1fr_1fr] gap-0.5 bg-black">
              {VIEWPORTS.map((vp) => (
                <div
                  key={vp.id}
                  className={`relative overflow-hidden ${
                    vp.id === 'seg-coronal' ? 'col-span-2' : ''
                  }`}
                >
                  <CornerstoneViewport
                    viewportId={vp.id}
                    seriesId={selectedSeriesId}
                    orientation={vp.orientation}
                    volumeId={volumeId}
                    imageIds={imageIds}
                    className="h-full w-full"
                    onViewportReady={handleViewportReady}
                  />
                  <span className="pointer-events-none absolute left-2 top-2 z-10 rounded bg-black/60 px-1.5 py-0.5 text-xs text-white/70">
                    {vp.label}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ---- Status bar ---- */}
      <div className="flex items-center justify-between border-t border-border bg-card px-4 py-1 text-xs text-muted-foreground">
        <span>
          Study: {selectedStudyId ? selectedStudyId.substring(0, 8) + '...' : '--'}
        </span>
        <span>
          Series: {selectedSeriesId ? selectedSeriesId.substring(0, 8) + '...' : '--'}
        </span>
        <span>
          Model: {selectedModel}
        </span>
        <span>
          Organs: {selectedOrgans.length > 0 ? selectedOrgans.join(', ') : 'all'}
        </span>
        {brushActive && <span className="text-yellow-400">Brush: Active</span>}
      </div>

      {/* ---- Export dialog ---- */}
      {exportOpen && completedJob && (
        <ExportDialog
          jobId={completedJob.id}
          onClose={() => setExportOpen(false)}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Default labels used when backend labels aren't loaded yet
// ---------------------------------------------------------------------------

const DEFAULT_LABELS: SegmentationLabel[] = [
  { index: 1, name: 'Liver', color: [255, 0, 0] },
  { index: 2, name: 'Kidney', color: [0, 255, 0] },
  { index: 3, name: 'Lung', color: [0, 0, 255] },
  { index: 4, name: 'Spleen', color: [255, 255, 0] },
  { index: 5, name: 'Pancreas', color: [255, 0, 255] },
  { index: 6, name: 'Bladder', color: [0, 255, 255] },
  { index: 7, name: 'Bone', color: [128, 128, 255] },
  { index: 8, name: 'Lesion (Tumor)', color: [255, 128, 0] },
  { index: 9, name: 'Lesion (Metastasis)', color: [255, 0, 128] },
];
