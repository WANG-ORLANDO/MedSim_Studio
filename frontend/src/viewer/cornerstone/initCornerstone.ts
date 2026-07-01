/**
 * Cornerstone3D Initialization Module
 *
 * Phase 1 of the Cornerstone3D rendering pipeline.
 * Provides singleton initialization of:
 * - @cornerstonejs/core rendering engine
 * - @cornerstonejs/tools registration
 * - @cornerstonejs/dicom-image-loader with web worker pool
 * - DICOM image loader registration (wadouri scheme)
 * - Tool type registration (WindowLevel, Pan, Zoom, StackScroll)
 *
 * Must be called once at application startup before creating any viewport.
 * Subsequent calls are safe (no-op after first initialization).
 */

import * as cs from '@cornerstonejs/core';
import {
  init as csToolsInit,
  addTool,
  WindowLevelTool,
  PanTool,
  ZoomTool,
  StackScrollTool,
  StackScrollMouseWheelTool,
  LengthTool,
  RectangleROITool,
  EllipticalROITool,
  ProbeTool,
  SegmentationDisplayTool,
} from '@cornerstonejs/tools';
import {
  configure as configureImageLoader,
  external as dicomImageLoaderExternal,
} from '@cornerstonejs/dicom-image-loader';

// ---------------------------------------------------------------------------
// Module-level state — prevents double initialization
// ---------------------------------------------------------------------------
let initialized = false;
let initPromise: Promise<void> | null = null;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/** GPU tier detection — let Cornerstone auto-detect */
const GPU_TIER = undefined;

/** Core rendering configuration */
const RENDERING_CONFIG = {
  rendering: {
    useCPURendering: false,
    preferSizeOverAccuracy: false,
    useNorm16Texture: true,
    strictZSpacingForVolumeViewport: true,
  },
  gpuTier: GPU_TIER,
  isMobile: false,
  detectGPUConfig: {},
  enableCacheOptimization: true,
};

/** DICOM image loader configuration */
const IMAGE_LOADER_CONFIG = {
  maxWebWorkers: Math.min(navigator.hardwareConcurrency || 1, 4),
  startWebWorkersOnDemand: true,
  taskConfiguration: {
    decodeTask: {
      initializeCodecsOnStartup: true,
      usePDFJS: false,
      strict: false,
    },
  },
};

// ---------------------------------------------------------------------------
// Tools to register globally
// ---------------------------------------------------------------------------
const TOOLS_TO_REGISTER = [
  WindowLevelTool,
  PanTool,
  ZoomTool,
  StackScrollTool,
  StackScrollMouseWheelTool,
  LengthTool,
  RectangleROITool,
  EllipticalROITool,
  ProbeTool,
  SegmentationDisplayTool,
] as const;

// ---------------------------------------------------------------------------
// Init function — call once at app start
// ---------------------------------------------------------------------------

/**
 * Initialize Cornerstone3D core, tools, and DICOM image loading pipeline.
 *
 * This function:
 * 1. Initializes the @cornerstonejs/core rendering engine with GPU config
 * 2. Configures the DICOM image loader with web worker pool
 * 3. Registers the wadouri image loader scheme
 * 4. Registers all tools globally (WindowLevel, Pan, Zoom, StackScroll)
 *
 * Safe to call multiple times — only executes once.
 *
 * @returns Promise that resolves when initialization is complete
 */
export async function initCornerstone(): Promise<void> {
  // Singleton guard — skip if already initialized
  if (initialized) return;

  // Deduplicate concurrent calls
  if (initPromise) return initPromise;

  initPromise = (async () => {
    console.info('[Cornerstone3D] Initializing...');

    try {
      // Disable SharedArrayBuffer to avoid requiring cross-origin isolation
      // headers (Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy).
      // Fall back to regular typed arrays for volume scalar data.
      cs.setUseSharedArrayBuffer(false);

      // ------------------------------------------------------------------
      // Step 1: Initialize @cornerstonejs/core
      // ------------------------------------------------------------------
      // This sets up the WebGL rendering context, GPU tier detection,
      // shared cache, and event system.
      await cs.init(RENDERING_CONFIG);
      console.info('[Cornerstone3D] Core initialized');

      // ------------------------------------------------------------------
      // Step 2: Initialize @cornerstonejs/tools event system
      // ------------------------------------------------------------------
      // This sets up native event listeners (wheel, mouse, touch) and the
      // custom event dispatchers that route events to tools.
      // Without this, tools won't respond to user input.
      csToolsInit();
      console.info('[Cornerstone3D] Tools initialized');

      // ------------------------------------------------------------------
      // Step 3: Configure DICOM image loader
      // Provide core module refs to dicom-image-loader.
      // The setter on `external.cornerstone` auto-registers
      // wadouri/wadors/dicomweb/dicomfile schemes.
      (dicomImageLoaderExternal as any).cornerstone = cs;
      // dicom-parser is required for DICOM image decoding; use dynamic
      // import to bypass CJS/ESM interop issues with the package's
      // __esModule flag.
      (dicomImageLoaderExternal as any).dicomParser = await import('dicom-parser');

      // Configure web worker pool and decoding options
      configureImageLoader(IMAGE_LOADER_CONFIG);
      console.info(
        `[Cornerstone3D] Image loader configured (${IMAGE_LOADER_CONFIG.maxWebWorkers} web workers)`
      );

      // ------------------------------------------------------------------
      // Step 5: Register tools globally
      // ------------------------------------------------------------------
      // Tools must be registered before they can be added to ToolGroups.
      // Registration makes the tool constructor available to the framework.
      for (const Tool of TOOLS_TO_REGISTER) {
        addTool(Tool);
      }
      console.info(
        `[Cornerstone3D] Registered ${TOOLS_TO_REGISTER.length} tools`
      );

      // Mark as initialized
      initialized = true;
      console.info('[Cornerstone3D] Initialization complete');
    } catch (error) {
      // Reset promise so caller can retry
      initPromise = null;
      console.error('[Cornerstone3D] Initialization failed:', error);
      throw error;
    }
  })();

  return initPromise;
}

/**
 * Check whether Cornerstone3D has been fully initialized.
 */
export function isInitialized(): boolean {
  return initialized;
}

/**
 * Reset initialization state (useful for testing / hot reload).
 */
export function resetInit(): void {
  initialized = false;
  initPromise = null;
}
