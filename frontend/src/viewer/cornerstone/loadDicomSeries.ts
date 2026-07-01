/**
 * DICOM Series Loading Module
 *
 * Phase 3 of the Cornerstone3D rendering pipeline.
 * Provides utilities for:
 * - Fetching DICOM instances for a given series from the backend
 * - Constructing Cornerstone3D imageIds (wadouri scheme)
 * - Setting a stack on a StackViewport
 * - Sorting instances by instance number for correct slice order
 * - Loading and rendering a series
 *
 * ImageId format: wadouri:<url>
 * The wadouri scheme tells Cornerstone3D's image loader to fetch
 * the DICOM file via HTTP using the provided URL.
 */

import { type StackViewport, type VolumeViewport, volumeLoader } from '@cornerstonejs/core';

const { createAndCacheVolumeFromImages } = volumeLoader;
import { dicomService } from '@services/index';
import type { DicomInstance } from '@/types/index';
import { getStackViewport } from './createRenderingEngine';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Base URL for DICOM instance file serving.
 *  The Vite dev server proxies /api -> http://localhost:8000,
 *  so we use the relative path. */
const INSTANCE_FILE_BASE = '/api/v1/dicom/instances';

// ---------------------------------------------------------------------------
// ImageId Construction
// ---------------------------------------------------------------------------

/**
 * Construct a Cornerstone3D imageId for a DICOM instance.
 *
 * The imageId uses the wadouri scheme which tells the dicom-image-loader
 * to fetch the DICOM file via HTTP.
 *
 * @param instanceId - The backend instance ID
 * @returns Full imageId string (e.g., "wadouri:/api/v1/dicom/instances/abc123/file")
 */
export function createImageId(instanceId: string): string {
  return `wadouri:${INSTANCE_FILE_BASE}/${instanceId}/file`;
}

/**
 * Create imageIds for all instances in a series.
 *
 * Instances are sorted by instanceNumber to ensure correct slice order.
 *
 * @param seriesId - The backend series ID
 * @param instances - Array of DicomInstance from the backend
 * @returns Array of sorted imageIds
 */
export function createImageIds(
  instances: DicomInstance[]
): string[] {
  // Sort instances by instance number for correct slice order
  const sorted = [...instances].sort(
    (a, b) => (a.instanceNumber ?? 0) - (b.instanceNumber ?? 0)
  );

  return sorted.map((inst) => createImageId(inst.id));
}

// ---------------------------------------------------------------------------
// Series Loading
// ---------------------------------------------------------------------------

/**
 * Fetch all DICOM instances for a given series and return their imageIds.
 *
 * @param seriesId - The backend series ID
 * @returns Sorted array of imageIds ready for setStack()
 */
export async function fetchImageIdsForSeries(
  seriesId: string
): Promise<string[]> {
  // Fetch instance list from the backend
  const instances = await dicomService.getInstances(seriesId);

  if (!instances || instances.length === 0) {
    throw new Error(`No instances found for series ${seriesId}`);
  }

  // Create sorted imageIds from instances
  const imageIds = createImageIds(instances);

  console.info(
    `[LoadDicomSeries] Created ${imageIds.length} imageIds for series ${seriesId}`
  );
  return imageIds;
}

// ---------------------------------------------------------------------------
// Stack Operations on Viewport
// ---------------------------------------------------------------------------

/**
 * Load a DICOM series onto a StackViewport and render it.
 *
 * Steps:
 * 1. Fetch instances for the given seriesId
 * 2. Construct sorted imageIds
 * 3. Set the image stack on the viewport
 * 4. Trigger render
 *
 * The viewport is looked up from the RenderingEngine by viewportId
 * before each engine operation, ensuring the instance is never stale.
 * This is critical for React StrictMode where viewports may be destroyed
 * and re-created before async operations complete.
 *
 * @param viewportId - The viewport ID to load the series onto
 * @param seriesId - The backend series ID to load
 * @param options - Optional configuration
 * @param options.imageIds - Pre-computed imageIds (skip fetching)
 * @returns The imageIds that were loaded
 */
export async function loadSeriesOnViewport(
  viewportId: string,
  seriesId: string,
  options?: { imageIds?: string[] }
): Promise<string[]> {
  // Get or fetch imageIds
  const imageIds =
    options?.imageIds ?? (await fetchImageIdsForSeries(seriesId));

  if (imageIds.length === 0) {
    throw new Error(`No imageIds available for series ${seriesId}`);
  }

  // Look up the viewport fresh from the engine to avoid using a stale
  // reference (e.g. when React StrictMode destroys/re-creates the
  // viewport while async operations are in-flight)
  const viewport = getStackViewport(viewportId) as StackViewport;
  if (!viewport) {
    // Viewport was cleaned up (e.g. StrictMode unmount) — not an error,
    // just nothing left to do. Return empty array so the caller's
    // catch block doesn't log a misleading error.
    return [];
  }

  // Set the stack — loads images into GPU memory
  await viewport.setStack(imageIds);

  // Render the first image
  viewport.render();

  console.info(
    `[LoadDicomSeries] Loaded ${imageIds.length} slices onto viewport ${viewportId}`
  );

  return imageIds;
}

/**
 * Navigate to a specific slice index in a StackViewport.
 *
 * @param viewport - The StackViewport
 * @param sliceIndex - The target slice index (0-based)
 */
export function goToSlice(
  viewport: StackViewport,
  sliceIndex: number
): void {
  viewport.setImageIdIndex(sliceIndex);
  viewport.render();
}

/**
 * Get the current slice index from a StackViewport.
 *
 * @param viewport - The StackViewport
 * @returns Current slice index
 */
export function getCurrentSliceIndex(
  viewport: StackViewport
): number {
  return viewport.getCurrentImageIdIndex();
}

/**
 * Get the total number of slices in a StackViewport.
 *
 * @param viewport - The StackViewport
 * @returns Total slice count
 */
export function getSliceCount(viewport: StackViewport): number {
  return viewport.getImageIds().length;
}

// ---------------------------------------------------------------------------
// Volume Operations (MPR)
// ---------------------------------------------------------------------------

/**
 * Load a DICOM series as a 3D volume and display it on a VolumeViewport.
 *
 * Creates a 3D volume from the series imageIds and sets it on the given
 * viewport. The volume is cached and shared across multiple viewports,
 * so subsequent calls with the same volumeId are near-instant.
 *
 * @param viewport - The VolumeViewport to display the volume on
 * @param seriesId - The backend series ID to load
 * @param volumeId - Unique ID for the volume cache entry
 * @param options - Optional configuration
 * @param options.imageIds - Pre-computed imageIds (skip fetching)
 * @returns The volumeId that was loaded
 */
export async function loadVolumeOnViewport(
  viewport: VolumeViewport,
  seriesId: string,
  volumeId: string,
  options?: { imageIds?: string[] }
): Promise<string> {
  // Get or fetch imageIds
  const imageIds =
    options?.imageIds ?? (await fetchImageIdsForSeries(seriesId));

  if (imageIds.length === 0) {
    throw new Error(`No imageIds available for series ${seriesId}`);
  }

  // Create a 3D volume from the imageIds (cached — subsequent calls are O(1))
  const volume = await createAndCacheVolumeFromImages(volumeId, imageIds);

  // Set the volume on the viewport
  await viewport.setVolumes([{ volumeId: volume.volumeId }]);

  // Increase maximum ray-cast samples per ray to handle large CT volumes.
  // Default is 1000 in vtk.js; Cornerstone3D sets 4000 on its shared mapper,
  // but the GPU-specific renderable may not inherit that value, producing
  // "The number of steps required NNN is larger than the specified maximum"
  // warnings. We bump it here on each actor to suppress the noise.
  try {
    const actors = (viewport as any).getActors?.() as any[];
    if (actors) {
      for (const actor of actors) {
        const mapper = actor?.getMapper?.();
        if (mapper && typeof mapper.setMaximumSamplesPerRay === 'function') {
          mapper.setMaximumSamplesPerRay(10000);
        }
      }
    }
  } catch {
    // getActors may not exist in all Cornerstone3D versions — safe to ignore
  }

  console.info(
    `[LoadVolumeOnViewport] Loaded volume ${volumeId} ` +
      `(${imageIds.length} slices) onto ${viewport.id}`
  );

  return volume.volumeId;
}
