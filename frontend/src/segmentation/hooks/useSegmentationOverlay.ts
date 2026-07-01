import { useState, useCallback, useRef } from 'react';
import type { SliceMask, SegmentationLabel } from '@/types/segmentation';
import { segmentationService } from '@/services/segmentationService';

/**
 * useSegmentationOverlay
 *
 * Hook that manages segmentation mask state for a single viewport.
 * Fetches slice masks from the backend as the user scrolls through slices,
 * caches them locally to minimize network requests, and prefetches
 * adjacent slices for smooth scrolling.
 *
 * Usage in CornerstoneViewport:
 *   const { currentSlice, loadSlice } = useSegmentationOverlay(jobId);
 *   // On slice change: loadSlice(newZIndex)
 *   // Render: currentSlice?.maskData
 */

interface OverlayCache {
  [key: string]: SliceMask; // key = `${jobId}:${zIndex}`
}

/** Shared cache accessible from both the hook and fire-and-forget prefetch */
const globalCache: OverlayCache = {};
const globalCacheKeys: string[] = [];

const MAX_CACHE_SIZE = 50;

function evictFromGlobalCache() {
  while (globalCacheKeys.length > MAX_CACHE_SIZE) {
    const oldest = globalCacheKeys.shift();
    if (oldest) delete globalCache[oldest];
  }
}

export function useSegmentationOverlay(jobId: string | null) {
  const [currentSlice, setCurrentSlice] = useState<SliceMask | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [labels, setLabels] = useState<SegmentationLabel[]>([]);

  const getCacheKey = useCallback(
    (zIndex: number) => `${jobId}:${zIndex}`,
    [jobId],
  );

  const loadSlice = useCallback(
    async (zIndex: number) => {
      if (!jobId) {
        setCurrentSlice(null);
        return;
      }

      const key = getCacheKey(zIndex);

      // Check global cache first
      const cached = globalCache[key];
      if (cached) {
        setCurrentSlice(cached);
        if (labels.length === 0 && cached.labels.length > 0) {
          setLabels(cached.labels);
        }
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const sliceMask = await segmentationService.getSliceMask(jobId, zIndex);

        // Store in global cache
        globalCache[key] = sliceMask;
        globalCacheKeys.push(key);
        evictFromGlobalCache();

        setCurrentSlice(sliceMask);
        if (sliceMask.labels.length > 0) {
          setLabels(sliceMask.labels);
        }
      } catch (err: any) {
        setError(err?.message ?? 'Failed to load slice mask');
        setCurrentSlice(null);
      } finally {
        setLoading(false);
      }

      // Prefetch adjacent slices for smooth scrolling (fire-and-forget)
      prefetchAdjacent(jobId, zIndex, getCacheKey);
    },
    [jobId, getCacheKey, labels.length],
  );

  const clearCache = useCallback(() => {
    // Clear global cache (owned by this hook instance — only one is expected)
    globalCacheKeys.length = 0;
    for (const key of Object.keys(globalCache)) {
      delete globalCache[key];
    }
    setCurrentSlice(null);
    setError(null);
    setLabels([]);
  }, []);

  return {
    currentSlice,
    labels,
    loading,
    error,
    loadSlice,
    clearCache,
  };
}

/**
 * Prefetch adjacent slices without awaiting (fire-and-forget).
 * Skips negative indices to avoid 400 errors from the backend.
 */
function prefetchAdjacent(
  jobId: string,
  zIndex: number,
  getCacheKey: (z: number) => string,
) {
  [zIndex - 1, zIndex + 1].forEach((adjZ) => {
    // Skip negative indices — backend rejects them with 400
    if (adjZ < 0) return;

    const key = getCacheKey(adjZ);
    if (globalCache[key]) return; // already cached

    segmentationService.getSliceMask(jobId, adjZ).then((sliceMask) => {
      globalCache[key] = sliceMask;
      globalCacheKeys.push(key);
      evictFromGlobalCache();
    }).catch(() => {
      // Prefetch failures are silently ignored
    });
  });
}
