/**
 * Rendering Engine Factory
 *
 * Phase 2 of the Cornerstone3D rendering pipeline.
 * Manages the singleton RenderingEngine instance and provides
 * utilities for creating and managing StackViewports.
 *
 * Architecture:
 * - One RenderingEngine per application (singleton pattern)
 * - RenderingEngine can manage multiple viewports
 * - Each viewport is identified by a unique viewportId
 * - StackViewports render 2D DICOM slices
 */

import {
  RenderingEngine,
  getRenderingEngine,
  type StackViewport,
  type VolumeViewport,
  Enums,
} from '@cornerstonejs/core';
import type { Types } from '@cornerstonejs/core';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Unique identifier for the application's RenderingEngine */
const ENGINE_ID = 'ctViewerEngine';

// ---------------------------------------------------------------------------
// Rendering Engine Singleton
// ---------------------------------------------------------------------------

/**
 * Get or create the singleton RenderingEngine.
 *
 * Cornerstone3D recommends a single RenderingEngine per application.
 * Multiple viewports are managed within this engine.
 *
 * @returns The singleton RenderingEngine instance
 */
export function createRenderingEngine(): Types.IRenderingEngine {
  // Check if engine already exists
  const existing = getRenderingEngine(ENGINE_ID);
  if (existing) {
    return existing;
  }

  // Create new engine
  const engine = new RenderingEngine(ENGINE_ID);
  console.info(`[RenderingEngine] Created: ${ENGINE_ID}`);
  return engine;
}

/**
 * Get the singleton RenderingEngine.
 *
 * Returns null if the engine has not been created yet.
 * Use `createRenderingEngine()` to lazily create it.
 */
export function getEngine(): Types.IRenderingEngine | null {
  return getRenderingEngine(ENGINE_ID) ?? null;
}

// ---------------------------------------------------------------------------
// Viewport Management
// ---------------------------------------------------------------------------

export interface EnableViewportInput {
  /** Unique ID for this viewport (e.g., 'ct-viewport', 'axial-viewport') */
  viewportId: string;
  /** The DOM element to render into */
  element: HTMLDivElement;
  /** Optional default window center (HU) */
  defaultWindowCenter?: number;
  /** Optional default window width (HU) */
  defaultWindowWidth?: number;
}

/**
 * Create and enable a StackViewport on the given DOM element.
 *
 * This:
 * 1. Gets or creates the RenderingEngine
 * 2. Enables the element as a STACK viewport
 * 3. Returns the StackViewport instance for further operations
 *
 * @param input - Viewport configuration
 * @returns The created StackViewport
 */
export async function enableStackViewport(
  input: EnableViewportInput
): Promise<StackViewport> {
  const { viewportId, element, defaultWindowCenter, defaultWindowWidth } =
    input;

  const engine = createRenderingEngine();

  // If the viewport already exists (e.g. from a previous StrictMode mount
  // that skipped disableElement), reuse it instead of calling enableElement
  // again. This preserves the event system and tool bindings.
  const existing = engine.getViewport(viewportId);
  if (existing) {
    return existing as StackViewport;
  }

  // Configure viewport input for a STACK type viewport
  const viewportInput = {
    viewportId,
    type: Enums.ViewportType.STACK,
    element,
    defaultOptions: {
      background: <[number, number, number]>[0, 0, 0], // black background
      ...(defaultWindowCenter !== undefined && {
        voiRange: {
          upper: defaultWindowCenter + (defaultWindowWidth ?? 400) / 2,
          lower: defaultWindowCenter - (defaultWindowWidth ?? 400) / 2,
        },
      }),
    },
  };

  // Enable the element — creates the viewport and WebGL resources
  engine.enableElement(viewportInput);

  // Get the StackViewport instance
  const viewport = engine.getViewport(viewportId) as StackViewport;

  console.info(`[RenderingEngine] Viewport enabled: ${viewportId}`);
  return viewport;
}

/**
 * Get a specific StackViewport by ID.
 *
 * @param viewportId - The viewport identifier
 * @returns The StackViewport, or null if not found
 */
export function getStackViewport(
  viewportId: string
): StackViewport | null {
  const engine = getRenderingEngine(ENGINE_ID);
  if (!engine) return null;

  try {
    const viewport = engine.getViewport(viewportId);
    // Check if it's a StackViewport
    if (viewport?.type !== Enums.ViewportType.STACK) return null;
    return viewport as StackViewport;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Volume Viewport Management
// ---------------------------------------------------------------------------

export interface EnableVolumeViewportInput {
  /** Unique ID for this viewport */
  viewportId: string;
  /** The DOM element to render into */
  element: HTMLDivElement;
  /** MPR orientation (axial, sagittal, coronal) */
  orientation: Enums.OrientationAxis;
  /** Optional default window center (HU) */
  defaultWindowCenter?: number;
  /** Optional default window width (HU) */
  defaultWindowWidth?: number;
}

/**
 * Create and enable a VolumeViewport for MPR rendering.
 *
 * Creates an ORTHOGRAPHIC type viewport with the specified
 * orientation for multi-planar reconstruction.
 *
 * @param input - Viewport configuration
 * @returns The created VolumeViewport
 */
export async function enableVolumeViewport(
  input: EnableVolumeViewportInput
): Promise<VolumeViewport> {
  const { viewportId, element, orientation, defaultWindowCenter, defaultWindowWidth } =
    input;

  const engine = createRenderingEngine();

  // If viewport already exists, reuse it
  const existing = engine.getViewport(viewportId);
  if (existing) {
    return existing as VolumeViewport;
  }

  const viewportInput = {
    viewportId,
    type: Enums.ViewportType.ORTHOGRAPHIC,
    element,
    defaultOptions: {
      background: <[number, number, number]>[0, 0, 0],
      orientation: orientation as any,
      ...(defaultWindowCenter !== undefined && {
        voiRange: {
          upper: defaultWindowCenter + (defaultWindowWidth ?? 400) / 2,
          lower: defaultWindowCenter - (defaultWindowWidth ?? 400) / 2,
        },
      }),
    },
  };

  engine.enableElement(viewportInput);

  const viewport = engine.getViewport(viewportId) as VolumeViewport;
  console.info(`[RenderingEngine] Volume viewport enabled: ${viewportId} (${orientation})`);
  return viewport;
}

/**
 * Get a specific VolumeViewport by ID.
 *
 * @param viewportId - The viewport identifier
 * @returns The VolumeViewport, or null if not found
 */
export function getVolumeViewport(
  viewportId: string
): VolumeViewport | null {
  const engine = getRenderingEngine(ENGINE_ID);
  if (!engine) return null;

  try {
    const viewport = engine.getViewport(viewportId);
    if (viewport?.type !== Enums.ViewportType.ORTHOGRAPHIC) return null;
    return viewport as VolumeViewport;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

/**
 * Disable a specific viewport and release its WebGL resources.
 *
 * Call this when a viewport's React component unmounts to prevent
 * WebGL memory leaks. Safe to call even if the viewport has already
 * been removed — checks existence first.
 *
 * @param viewportId - The viewport to disable
 */
export function disableViewport(viewportId: string): void {
  const engine = getRenderingEngine(ENGINE_ID);
  if (!engine) return;

  if (!engine.getViewport(viewportId)) {
    return; // already gone — nothing to do
  }

  try {
    engine.disableElement(viewportId);
    console.info(`[RenderingEngine] Viewport disabled: ${viewportId}`);
  } catch (error) {
    console.warn(
      `[RenderingEngine] Error disabling viewport ${viewportId}:`,
      error
    );
  }
}

/**
 * Destroy the entire RenderingEngine and release all WebGL resources.
 *
 * This is a destructive operation — call only when the application
 * is shutting down or the viewer module is fully unmounted.
 */
export function destroyRenderingEngine(): void {
  const engine = getRenderingEngine(ENGINE_ID);
  if (!engine) return;

  try {
    engine.destroy();
    console.info('[RenderingEngine] Destroyed');
  } catch (error) {
    console.warn('[RenderingEngine] Error during destruction:', error);
  }
}

/**
 * Resize all viewports in the RenderingEngine.
 *
 * Call this when the container element's size changes (e.g., window resize,
 * sidebar toggle, panel resize) to ensure correct rendering.
 */
export function resizeViewports(): void {
  const engine = getRenderingEngine(ENGINE_ID);
  if (!engine) return;

  try {
    engine.resize();
  } catch (error) {
    console.warn('[RenderingEngine] Error during resize:', error);
  }
}
