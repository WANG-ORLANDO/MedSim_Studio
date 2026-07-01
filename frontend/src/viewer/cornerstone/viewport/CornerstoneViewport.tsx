/**
 * CornerstoneViewport Component
 *
 * Phase 4 of the Cornerstone3D rendering pipeline.
 * React component that wraps a Cornerstone3D viewport with:
 * - Proper lifecycle management (init → render → cleanup)
 * - React StrictMode safety (prevents double initialization)
 * - Auto resize on container size changes
 * - DICOM series loading via loadDicomSeries module
 * - Tool group binding
 * - WebGL memory leak prevention on unmount
 *
 * Segmentation overlay rendering is handled automatically by the
 * SegmentationDisplayTool (registered globally) — no custom canvas needed.
 *
 * Usage:
 * ```tsx
 * <CornerstoneViewport
 *   viewportId="ct-viewport"
 *   seriesId="abc-123"
 *   className="h-full w-full"
 *   onViewportReady={(id) => console.log('ready', id)}
 * />
 * ```
 */

import { useRef, useEffect, useState } from 'react';
import { type StackViewport, type VolumeViewport } from '@cornerstonejs/core';
import {
  enableStackViewport,
  enableVolumeViewport,
  disableViewport,
  resizeViewports,
  getStackViewport,
} from '../createRenderingEngine';
import { loadSeriesOnViewport, loadVolumeOnViewport } from '../loadDicomSeries';
import { addViewportToToolGroup } from '../toolGroups';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface CornerstoneViewportProps {
  /** Unique identifier for this viewport (must be stable across renders) */
  viewportId: string;
  /** Backend series ID to load DICOM images from */
  seriesId: string;
  /** Optional pre-computed imageIds (skip fetching) */
  imageIds?: string[];
  /** MPR orientation — enables VolumeViewport (instead of StackViewport) */
  orientation?: 'axial' | 'sagittal' | 'coronal';
  /** Shared volume ID for MPR — all three viewports use the same volume */
  volumeId?: string;
  /** Additional CSS classes */
  className?: string;
  /** Callback when the viewport is ready and rendered */
  onViewportReady?: (viewportId: string) => void;
  /** ToolGroup identifier to bind this viewport to */
  toolGroupId?: string;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function CornerstoneViewport({
  viewportId,
  seriesId,
  imageIds: preloadedImageIds,
  orientation,
  volumeId,
  className = '',
  onViewportReady,
  toolGroupId = 'ct-tool-group',
}: CornerstoneViewportProps) {
  // Ref to the DOM container element
  const elementRef = useRef<HTMLDivElement>(null);

  // Track viewport instance for cleanup
  const viewportRef = useRef<StackViewport | VolumeViewport | null>(null);

  // Stable ref for the onViewportReady callback
  const onReadyRef = useRef(onViewportReady);
  onReadyRef.current = onViewportReady;

  // Loading/error state for UI feedback
  const [loadState, setLoadState] = useState<
    'idle' | 'loading' | 'loaded' | 'error'
  >('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // ------------------------------------------------------------------
  // Resize handler
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!elementRef.current) return;

    const observer = new ResizeObserver(() => {
      resizeViewports();
    });

    observer.observe(elementRef.current);

    return () => {
      observer.disconnect();
    };
  }, []);

  // ------------------------------------------------------------------
  // Main initialization effect
  // ------------------------------------------------------------------
  useEffect(() => {
    const element = elementRef.current;
    if (!element || !seriesId) return;

    let cancelled = false;

    const init = async () => {
      try {
        setLoadState('loading');
        setErrorMessage(null);

        const isVolume = !!orientation;

        if (isVolume) {
          // -----------------------------------------------------------
          // Volume mode (MPR orientation)
          // -----------------------------------------------------------

          // Create and enable the VolumeViewport with the given orientation
          const viewport = await enableVolumeViewport({
            viewportId,
            element,
            orientation: orientation as any,
          });

          if (cancelled) return;
          viewportRef.current = viewport;

          // Load the volume onto this viewport.
          // The volume is shared across viewports via the volumeId.
          const volId = volumeId || `volume-${seriesId}`;
          await loadVolumeOnViewport(viewport, seriesId, volId, {
            imageIds: preloadedImageIds,
          });

          if (cancelled) return;
        } else {
          // -----------------------------------------------------------
          // Stack mode (2D slices — original behavior)
          // -----------------------------------------------------------

          // Create and enable the StackViewport
          const viewport = await enableStackViewport({
            viewportId,
            element,
          });

          if (cancelled) return;
          viewportRef.current = viewport;

          // Load the DICOM series onto the viewport
          const loadedImageIds = await loadSeriesOnViewport(viewportId, seriesId, {
            imageIds: preloadedImageIds,
          });

          if (cancelled) return;
          if (loadedImageIds.length === 0) return;
        }

        // Bind this viewport to the tool group
        addViewportToToolGroup(toolGroupId, viewportId);

        setLoadState('loaded');
        console.info(
          `[CornerstoneViewport] Initialized ${viewportId}` +
            (orientation ? ` (${orientation})` : '')
        );

        // Force resize to ensure canvas & event system are properly connected
        resizeViewports();

        // notify parent
        onReadyRef.current?.(viewportId);
      } catch (error) {
        if (cancelled) return;

        const message =
          error instanceof Error ? error.message : 'Viewport initialization failed';
        console.error(`[CornerstoneViewport] ${viewportId} error:`, message);
        setLoadState('error');
        setErrorMessage(message);
      }
    };

    init();

    // Cleanup
    return () => {
      cancelled = true;

      // Disable the viewport to release WebGL resources and remove
      // Cornerstone3D event listeners (mouse, wheel, touch, etc.)
      disableViewport(viewportId);
      viewportRef.current = null;

      console.info(`[CornerstoneViewport] Cleaned up: ${viewportId}`);
    };
  }, [viewportId, seriesId, preloadedImageIds, orientation, volumeId, toolGroupId]);

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------
  return (
    <div className={`relative overflow-hidden bg-black ${className}`}>
      {/* The Cornerstone3D viewport renders into this div */}
      <div
        ref={elementRef}
        className="h-full w-full"
        data-viewport-id={viewportId}
        style={{ minHeight: '100%', minWidth: '100%' }}
      />

      {/* Loading overlay */}
      {loadState === 'loading' && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/60">
          <div className="flex flex-col items-center gap-2">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-white/40 border-t-white" />
            <span className="text-xs text-white/60">
              {orientation ? 'Loading volume...' : 'Loading DICOM...'}
            </span>
          </div>
        </div>
      )}

      {/* Error overlay */}
      {loadState === 'error' && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80">
          <div className="max-w-xs text-center">
            <p className="mb-1 text-sm font-medium text-red-400">
              Failed to load
            </p>
            <p className="text-xs text-white/50">
              {errorMessage || 'Unknown error'}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
