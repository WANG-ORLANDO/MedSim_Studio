import { useRef, useEffect } from 'react';
import { useViewerStore } from '@store/useViewerStore';

/**
 * MPRViewer Component
 *
 * Multi-Planar Reconstruction viewport for CT images.
 * Renders axial, sagittal, or coronal reconstructions
 * using Cornerstone3D rendering engine.
 */

interface MPRViewerProps {
  orientation: 'axial' | 'sagittal' | 'coronal';
  studyId?: string;
  seriesId?: string;
}

export function MPRViewer({ orientation, studyId, seriesId }: MPRViewerProps) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const { mprState, activePreset } = useViewerStore();

  useEffect(() => {
    // TODO: Initialize Cornerstone3D viewport for this orientation
    // This will be implemented when Cornerstone3D is fully integrated
    const element = viewportRef.current;
    if (!element) return;

    const initViewport = async () => {
      try {
        // const renderingEngine = getRenderingEngine('ctViewer');
        // const viewportInput = {
        //   viewportId: `viewport-${orientation}`,
        //   type: ViewportType.ORTHOGRAPHIC,
        //   element,
        //   defaultOptions: {
        //     orientation: orientationMap[orientation],
        //   },
        // };
        // renderingEngine.enableElement(viewportInput);
      } catch (error) {
        console.error(`Failed to initialize ${orientation} viewport:`, error);
      }
    };

    initViewport();

    return () => {
      // Cleanup Cornerstone3D viewport
    };
  }, [orientation, studyId, seriesId]);

  return (
    <div className="relative h-full w-full overflow-hidden bg-black">
      {/* Orientation label */}
      <div className="pointer-events-none absolute left-2 top-2 z-10">
        <span className="rounded bg-black/60 px-2 py-0.5 text-xs font-medium uppercase text-white">
          {orientation}
        </span>
      </div>

      {/* Viewport element */}
      <div
        ref={viewportRef}
        className="viewport-element h-full w-full"
        data-orientation={orientation}
      />

      {/* Window/Level overlay */}
      {activePreset && (
        <div className="pointer-events-none absolute bottom-2 right-2 z-10">
          <span className="rounded bg-black/60 px-2 py-0.5 text-xs text-white/80">
            W: {activePreset.windowWidth} L: {activePreset.windowCenter}
          </span>
        </div>
      )}

      {/* Slice indicator */}
      {mprState && (
        <div className="pointer-events-none absolute bottom-2 left-2 z-10">
          <span className="rounded bg-black/60 px-2 py-0.5 text-xs text-white/80">
            Slice: {mprState[orientation]?.sliceIndex ?? 0}
          </span>
        </div>
      )}
    </div>
  );
}

const orientationMap = {
  axial: [0, 0, -1],
  sagittal: [-1, 0, 0],
  coronal: [0, -1, 0],
} as const;
