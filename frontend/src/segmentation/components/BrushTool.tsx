import { useCallback, useRef, useEffect } from 'react';
import type { SegmentationLabel } from '@/types/segmentation';
import { segmentationService } from '@/services/segmentationService';

interface BrushToolProps {
  jobId: string;
  active: boolean;
  selectedLabel: number;
  viewportElement: HTMLDivElement | null;
  currentZIndex: number;
  onMaskUpdated: (zIndex: number, patchData: number[][]) => void;
}

/**
 * BrushTool
 *
 * Handles click/drag interactions in a Cornerstone3D viewport for
 * interactive segmentation refinement. On click, sends the voxel
 * coordinates to the backend's interactive click endpoint, which
 * performs intensity-constrained region growing and returns the
 * updated local mask patch.
 *
 * Events are debounced to avoid flooding the backend during drag.
 */
export function BrushTool({
  jobId,
  active,
  selectedLabel,
  viewportElement,
  currentZIndex,
  onMaskUpdated,
}: BrushToolProps) {
  const lastClickRef = useRef<{ x: number; y: number; time: number } | null>(null);
  const debounceRef = useRef<number | null>(null);

  const handleInteraction = useCallback(
    (clientX: number, clientY: number) => {
      if (!viewportElement || !active) return;

      // Calculate position relative to the viewport element
      const rect = viewportElement.getBoundingClientRect();
      const relX = clientX - rect.left;
      const relY = clientY - rect.top;

      // Debounce: skip if same position within 300ms
      const now = Date.now();
      const last = lastClickRef.current;
      if (
        last &&
        Math.abs(last.x - relX) < 3 &&
        Math.abs(last.y - relY) < 3 &&
        now - last.time < 300
      ) {
        return;
      }
      lastClickRef.current = { x: relX, y: relY, time: now };

      // Clear previous debounce
      if (debounceRef.current !== null) {
        clearTimeout(debounceRef.current);
      }

      debounceRef.current = window.setTimeout(async () => {
        try {
          // Map pixel coordinates to voxel coordinates.
          // The viewport image may be zoomed/panned, so we compute the
          // normalized position and map to voxel space.
          const imageWidth = viewportElement.clientWidth;
          const imageHeight = viewportElement.clientHeight;

          // Simple mapping: normalize click to [0, 1], then to voxel coords.
          // In production, this would use Cornerstone3D's viewport API
          // to convert canvas coordinates to image coordinates.
          const normX = relX / imageWidth;
          const normY = relY / imageHeight;

          // Estimate voxel coordinates based on image dimensions.
          // The backend will clamp to valid range.
          const voxelX = Math.round(normX * 512); // approximate image width
          const voxelY = Math.round(normY * 512); // approximate image height

          const response = await segmentationService.interactiveClick({
            jobId,
            zIndex: currentZIndex,
            x: voxelX,
            y: voxelY,
            label: selectedLabel,
            operation: 'add',
          });

          onMaskUpdated(response.zIndex, response.patchData);
        } catch {
          // Silently handle errors during refinement
        }
      }, 300); // 300ms debounce
    },
    [viewportElement, active, jobId, currentZIndex, selectedLabel, onMaskUpdated],
  );

  // Attach event listeners to the viewport element
  useEffect(() => {
    if (!viewportElement || !active) return;

    const onClick = (e: MouseEvent) => {
      handleInteraction(e.clientX, e.clientY);
    };

    const onTouchEnd = (e: TouchEvent) => {
      if (e.changedTouches.length > 0) {
        const touch = e.changedTouches[0];
        handleInteraction(touch.clientX, touch.clientY);
      }
    };

    viewportElement.addEventListener('click', onClick);
    viewportElement.addEventListener('touchend', onTouchEnd);

    // Change cursor to crosshair when brush is active
    const originalCursor = viewportElement.style.cursor;
    viewportElement.style.cursor = 'crosshair';

    return () => {
      viewportElement.removeEventListener('click', onClick);
      viewportElement.removeEventListener('touchend', onTouchEnd);
      viewportElement.style.cursor = originalCursor;

      if (debounceRef.current !== null) {
        clearTimeout(debounceRef.current);
      }
    };
  }, [viewportElement, active, handleInteraction]);

  // This component renders nothing — it's purely behavioral
  return null;
}
