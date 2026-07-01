import { useEffect, useState, useRef, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { volumeLoader, imageLoader } from '@cornerstonejs/core';
const { createAndCacheVolumeFromImages } = volumeLoader;
import { useViewerStore } from '@store/useViewerStore';
import { CornerstoneViewport } from '@viewer/cornerstone/viewport/CornerstoneViewport';
import { WINDOW_LEVEL_PRESETS } from '@/types/dicom';
import { Button } from '@components/ui/button';
import { dicomService } from '@services/index';
import {
  createToolGroup,
  setActiveTool as setActiveToolInGroup,
} from '@viewer/cornerstone/toolGroups';
import { initCornerstone } from '@viewer/cornerstone/initCornerstone';
import { fetchImageIdsForSeries } from '@viewer/cornerstone/loadDicomSeries';
import { VolumeRenderer } from '@vtk';
import type { DicomSeries } from '@/types/index';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const VOLUME_ID_PREFIX = 'mpr-volume';

const VIEWPORTS = [
  { id: 'axial-viewport', orientation: 'axial' as const, label: 'A' },
  { id: 'sagittal-viewport', orientation: 'sagittal' as const, label: 'S' },
  { id: 'coronal-viewport', orientation: 'coronal' as const, label: 'C' },
] as const;

const TOOLS = [
  { id: 'WindowLevel', label: 'W/L' },
  { id: 'Pan', label: 'Pan' },
  { id: 'Zoom', label: 'Zoom' },
  { id: 'StackScroll', label: 'Scroll' },
  { id: 'Length', label: 'Length' },
  { id: 'RectangleROI', label: 'Rect' },
  { id: 'EllipticalROI', label: 'Ellipse' },
  { id: 'Probe', label: 'Probe' },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ViewerPage() {
  const { studyId } = useParams<{ studyId: string }>();
  const { activePreset, activeTool, setWindowLevel, setActiveTool } =
    useViewerStore();

  const [seriesList, setSeriesList] = useState<DicomSeries[]>([]);
  const [activeSeriesId, setActiveSeriesId] = useState<string | null>(null);
  const [viewerMode, setViewerMode] = useState<'mpr' | '3d'>('mpr');
  const [cornerstoneReady, setCornerstoneReady] = useState(false);
  const [volumeReady, setVolumeReady] = useState(false);
  const [viewportReady, setViewportReady] = useState(false);
  const [totalSlices, setTotalSlices] = useState(0);
  const imageIdsRef = useRef<string[]>([]);

  // ------------------------------------------------------------------
  // Init Cornerstone3D once on page mount
  // ------------------------------------------------------------------
  useEffect(() => {
    initCornerstone()
      .then(() => {
        createToolGroup();
        setCornerstoneReady(true);
      })
      .catch((err) => {
        console.error('[ViewerPage] Cornerstone init failed:', err);
      });
  }, []);

  // ------------------------------------------------------------------
  // Load series list for the given study
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!studyId) return;

    let mounted = true;

    const loadSeries = async () => {
      try {
        const allSeries = await dicomService.getSeries(studyId);
        if (!mounted) return;

        // Only imaging modalities (CT, MR, etc.) can be used for volume
        // rendering. Exclude RTSTRUCT, SEG, REG, PR and other non-image types.
        const IMAGE_MODALITIES = ['CT', 'MR', 'PT', 'NM', 'US', 'XA', 'CR'];
        const series = allSeries.filter(
          (s: DicomSeries) => IMAGE_MODALITIES.includes(s.modality)
        );

        setSeriesList(series);

        if (series.length > 0) {
          setActiveSeriesId(series[0].id);
          setTotalSlices(series[0].imageCount);
        }
      } catch (error) {
        console.error('[ViewerPage] Failed to load series:', error);
      }
    };

    loadSeries();

    return () => {
      mounted = false;
    };
  }, [studyId]);

  // ------------------------------------------------------------------
  // Pre-create the 3D volume when a series is selected
  // Volume is created once and shared across all 3 MPR viewports
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!cornerstoneReady || !activeSeriesId) return;

    let cancelled = false;

    const prepareVolume = async () => {
      try {
        setVolumeReady(false);

        // Fetch imageIds for the series
        const imageIds = await fetchImageIdsForSeries(activeSeriesId);
        if (cancelled) return;

        imageIdsRef.current = imageIds;
        setTotalSlices(imageIds.length);

        // Pre-load all images into cache so DICOM metadata is available
        // for volume creation. The wadouri metadata provider reads from the
        // dataSetCacheManager which is populated during image loading.
        await Promise.all(imageLoader.loadAndCacheImages(imageIds));
        if (cancelled) return;

        // Build the 3D volume from the cached images
        const volumeId = `${VOLUME_ID_PREFIX}-${activeSeriesId}`;
        await createAndCacheVolumeFromImages(volumeId, imageIds);
        if (cancelled) return;

        setVolumeReady(true);
      } catch (error) {
        console.error('[ViewerPage] Volume creation failed:', error);
      }
    };

    prepareVolume();

    return () => {
      cancelled = true;
    };
  }, [cornerstoneReady, activeSeriesId]);

  // ------------------------------------------------------------------
  // Tool switching
  // ------------------------------------------------------------------
  const handleToolChange = (tool: string) => {
    setActiveTool(tool);
    setActiveToolInGroup(tool);
  };

  const handleViewportReady = useCallback(() => {
    setViewportReady(true);
  }, []);

  const volumeId = activeSeriesId
    ? `${VOLUME_ID_PREFIX}-${activeSeriesId}`
    : undefined;

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------
  return (
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <div className="flex items-center gap-2 border-b border-border bg-card px-4 py-2">
        {/* Series selector */}
        {seriesList.length > 1 && (
          <>
            <select
              value={activeSeriesId ?? ''}
              onChange={(e) => setActiveSeriesId(e.target.value)}
              className="rounded border border-border bg-background px-2 py-1 text-xs"
            >
              {seriesList.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.seriesDescription || `Series ${s.seriesNumber}`}
                </option>
              ))}
            </select>
            <div className="mx-2 h-6 w-px bg-border" />
          </>
        )}

        {/* Tools */}
        <div className="flex items-center gap-1">
          {TOOLS.map((tool) => (
            <Button
              key={tool.id}
              variant={activeTool === tool.id ? 'secondary' : 'ghost'}
              size="sm"
              onClick={() => handleToolChange(tool.id)}
            >
              {tool.label}
            </Button>
          ))}
        </div>

        <div className="mx-2 h-6 w-px bg-border" />

        {/* 3D toggle */}
        <Button
          variant={viewerMode === '3d' ? 'secondary' : 'ghost'}
          size="sm"
          onClick={() =>
            setViewerMode((v) => (v === 'mpr' ? '3d' : 'mpr'))
          }
          disabled={!activeSeriesId || !volumeReady}
        >
          3D
        </Button>

        <div className="mx-2 h-6 w-px bg-border" />

        {/* Window/Level presets */}
        <div className="flex items-center gap-1">
          {WINDOW_LEVEL_PRESETS.map((preset) => (
            <Button
              key={preset.name}
              variant={activePreset?.name === preset.name ? 'secondary' : 'ghost'}
              size="sm"
              onClick={() => setWindowLevel(preset)}
              title={preset.description}
            >
              {preset.name}
            </Button>
          ))}
        </div>
      </div>

      {/* MPR Viewport Grid */}
      <div className="flex-1">
        {!cornerstoneReady ? (
          <div className="flex h-full items-center justify-center bg-black">
            <div className="flex flex-col items-center gap-3">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-white/20 border-t-white" />
              <p className="text-sm text-white/50">Initializing rendering engine...</p>
            </div>
          </div>
        ) : !studyId ? (
          <div className="flex h-full items-center justify-center bg-black">
            <p className="text-lg text-white/40">No study selected</p>
          </div>
        ) : viewerMode === '3d' && activeSeriesId ? (
          <VolumeRenderer
            mode="dicom"
            seriesId={activeSeriesId}
            showControls
            opacityPreset="ct-soft-tissue"
          />
        ) : !activeSeriesId || !volumeReady ? (
          <div className="flex h-full items-center justify-center bg-black">
            <div className="flex flex-col items-center gap-3">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-white/20 border-t-white" />
              <p className="text-sm text-white/50">Building 3D volume...</p>
            </div>
          </div>
        ) : (
          <div className="grid h-full w-full grid-cols-2 grid-rows-[1fr_1fr] gap-0.5 bg-black">
            {VIEWPORTS.map((vp) => (
              <div
                key={vp.id}
                className={`relative overflow-hidden ${
                  vp.id === 'coronal-viewport' ? 'col-span-2' : ''
                }`}
              >
                <CornerstoneViewport
                  viewportId={vp.id}
                  seriesId={activeSeriesId}
                  orientation={vp.orientation}
                  volumeId={volumeId}
                  imageIds={imageIdsRef.current}
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

      {/* Status bar */}
      <div className="flex items-center justify-between border-t border-border bg-card px-4 py-1 text-xs text-muted-foreground">
        <span>
          Study: {studyId ? studyId.substring(0, 8) + '...' : '--'}
        </span>
        <span>
          Series:{' '}
          {activeSeriesId ? activeSeriesId.substring(0, 8) + '...' : '--'}
        </span>
        <span>Slices: {totalSlices > 0 ? totalSlices : '--'}</span>
        <span>Tool: {activeTool}</span>
        <span>
          {activePreset
            ? `WW: ${activePreset.windowWidth} / WC: ${activePreset.windowCenter}`
            : 'WW: -- / WC: --'}
        </span>
      </div>
    </div>
  );
}
