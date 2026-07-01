import { useRef, useEffect, useState, useCallback } from 'react';

// ---------------------------------------------------------------------------
// vtk.js imports
// Profile must load first to register GPU volume ray cast mapper backends
// ---------------------------------------------------------------------------
import '@kitware/vtk.js/Rendering/Profiles/Volume';

import vtkRenderWindow from '@kitware/vtk.js/Rendering/Core/RenderWindow';
import vtkRenderer from '@kitware/vtk.js/Rendering/Core/Renderer';
import vtkOpenGLRenderWindow from '@kitware/vtk.js/Rendering/OpenGL/RenderWindow';
import vtkRenderWindowInteractor from '@kitware/vtk.js/Rendering/Core/RenderWindowInteractor';
import vtkInteractorStyleTrackballCamera from '@kitware/vtk.js/Interaction/Style/InteractorStyleTrackballCamera';
import vtkVolume from '@kitware/vtk.js/Rendering/Core/Volume';
import vtkVolumeMapper from '@kitware/vtk.js/Rendering/Core/VolumeMapper';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import vtkPiecewiseFunction from '@kitware/vtk.js/Common/DataModel/PiecewiseFunction';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkPlane from '@kitware/vtk.js/Common/DataModel/Plane';

import { loadDicomVolume, type DicomVolumeData } from './dicomVolumeLoader';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type PresetName = 'ct-bone' | 'ct-soft-tissue' | 'ct-lung' | 'ct-angio';

/** Normalised clipping plane positions (0..1). 0 = disabled, 1 = max clip. */
interface ClipState {
  x: number;
  y: number;
  z: number;
}

/** A single segmentation label with RGB color */
interface SegmentationLabelDef {
  index: number;
  name: string;
  color: [number, number, number]; // RGB 0-255
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Default peak opacity for segmentation overlay (0..1).
 *  Lower values make the overlay more transparent so underlying CT
 *  anatomy (vessels, bronchi, texture) remains visible. */
const DEFAULT_SEG_OPACITY = 0.25;

/** Half-band width around integer label values in the opacity transfer
 *  function.  Keeps the function sharp at each label while preventing
 *  floating-point sampling misses. */
const LABEL_HALF_BAND = 0.45;

interface VolumeRendererProps {
  /** Render mode: synthetic phantom (default) or real DICOM series */
  mode?: 'synthetic' | 'dicom';
  /** DICOM series ID — required when mode='dicom' */
  seriesId?: string;
  /** Initial opacity function preset */
  opacityPreset?: PresetName;
  /** Whether to show rendering controls */
  showControls?: boolean;
  /** Optional 3D segmentation label map for overlay */
  segmentationMask?: Float32Array | null;
  /** Label definitions for coloring the segmentation mask */
  segmentationLabels?: SegmentationLabelDef[] | null;
  /**
   * External volume data for synthetic mode.
   * When provided with mode='synthetic', this replaces the built-in
   * 64³ phantom. scalarData must be a Float32Array of HU values
   * in [x, y, z] fast-axis order (x fastest).
   */
  syntheticData?: Float32Array | null;
  /** Dimensions [x, y, z] for syntheticData (required if syntheticData given) */
  syntheticDims?: [number, number, number] | null;
  /** Spacing [x, y, z] in mm for syntheticData */
  syntheticSpacing?: [number, number, number] | null;
  /**
   * Progressive-scan clip index (0-based).
   * When set, only voxels along the scan axis at indices 0..clipIndex
   * are visible; indices beyond are clipped away.  Used to simulate
   * a CT scan that builds up slice-by-slice from top to bottom.
   * Undefined / -1 disables scan clipping (full volume shown).
   */
  syntheticClipIndex?: number;
  /**
   * Axis along which the progressive scan proceeds.
   * 'x', 'y', or 'z' — defaults to 'z' (superior→inferior for CT phantom).
   */
  syntheticScanAxis?: 'x' | 'y' | 'z';
  /**
   * Enable head-to-feet scan view mode.
   * When true, the camera is oriented so the Z axis (head→feet direction)
   * points downward on screen, making the progressive scan appear to build
   * from top (head/chest) to bottom (abdomen/pelvis).
   * Also enables the scan-plane overlay and direction labels.
   */
  scanView?: boolean;
  /**
   * Scan direction label — used for the overlay text.
   * 'head_to_feet' (default) or 'feet_to_head'.
   */
  scanDirection?: 'head_to_feet' | 'feet_to_head';
}

/** Shape of a transfer function preset */
interface TransferFunctionPreset {
  /** [scalarValue, red, green, blue] — 0..1 range */
  colorPoints: Array<[number, number, number, number]>;
  /** [scalarValue, opacity] — 0..1 range */
  opacityPoints: Array<[number, number]>;
}

// ---------------------------------------------------------------------------
// Transfer function presets (scalar range mapped to -1024..3071 HU)
// ---------------------------------------------------------------------------

const PRESETS: Record<PresetName, TransferFunctionPreset> = {
  'ct-bone': {
    colorPoints: [
      [-1024, 0.0, 0.0, 0.0],
      [-400, 0.2, 0.15, 0.1],
      [200, 0.5, 0.35, 0.2],
      [1000, 0.95, 0.95, 0.95],
      [3071, 1.0, 1.0, 1.0],
    ],
    opacityPoints: [
      [-1024, 0.0],
      [-200, 0.0],
      [100, 0.05],
      [500, 0.35],
      [1000, 0.8],
      [3071, 1.0],
    ],
  },

  'ct-soft-tissue': {
    colorPoints: [
      [-1024, 0.0, 0.0, 0.0],
      [-200, 0.1, 0.1, 0.1],
      [40, 0.4, 0.3, 0.2],
      [80, 0.6, 0.5, 0.4],
      [300, 0.9, 0.9, 0.85],
      [3071, 1.0, 1.0, 1.0],
    ],
    opacityPoints: [
      [-1024, 0.0],
      [-200, 0.0],
      [-50, 0.03],
      [40, 0.15],
      [80, 0.4],
      [300, 0.8],
      [3071, 1.0],
    ],
  },

  'ct-lung': {
    colorPoints: [
      [-1024, 0.0, 0.0, 0.0],
      [-900, 0.1, 0.1, 0.1],
      [-700, 0.3, 0.3, 0.3],
      [-500, 0.5, 0.5, 0.5],
      [0, 0.8, 0.75, 0.7],
      [3071, 1.0, 1.0, 1.0],
    ],
    opacityPoints: [
      [-1024, 0.0],
      [-900, 0.05],
      [-750, 0.15],
      [-500, 0.4],
      [0, 0.8],
      [3071, 1.0],
    ],
  },

  'ct-angio': {
    colorPoints: [
      [-1024, 0.0, 0.0, 0.0],
      [-200, 0.1, 0.1, 0.1],
      [50, 0.6, 0.2, 0.15],
      [200, 0.9, 0.3, 0.2],
      [500, 1.0, 0.7, 0.5],
      [3071, 1.0, 1.0, 1.0],
    ],
    opacityPoints: [
      [-1024, 0.0],
      [-50, 0.0],
      [50, 0.05],
      [150, 0.35],
      [300, 0.7],
      [500, 0.95],
      [3071, 1.0],
    ],
  },
};

// ---------------------------------------------------------------------------
// Synthetic volume generation — CT-like phantom (64 × 64 × 64)
// ---------------------------------------------------------------------------

/** Generate a 64³ Float32Array with air (-1000), soft tissue (~40),
 *  bone (~700), and a small calcification (~300).
 *  A bit of sinusoidal texture is added to the soft-tissue region so the
 *  volume does not look perfectly uniform. */
function generateSyntheticVolume(): Float32Array {
  const DIM = 64;
  const size = DIM * DIM * DIM;
  const data = new Float32Array(size);
  const center = DIM / 2; // 32

  for (let z = 0; z < DIM; z++) {
    for (let y = 0; y < DIM; y++) {
      for (let x = 0; x < DIM; x++) {
        const idx = z * DIM * DIM + y * DIM + x;

        const dx = x - center;
        const dy = y - center;
        const dz = z - center;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        // ---- air background ----
        let hu = -1000;

        // ---- soft-tissue sphere (r ≈ 22) ----
        if (dist < 22) {
          // Add subtle sinusoidal texture to break uniformity
          const texture =
            Math.sin(dx * 0.3) * Math.cos(dy * 0.3) * Math.sin(dz * 0.3) * 12;
          hu = 40 + texture;
        }

        // ---- bone-like core (r ≈ 8, slightly off-centre) ----
        const bDx = x - (center - 4);
        const bDy = y - (center + 2);
        const bDz = z - (center - 3);
        const boneDist = Math.sqrt(bDx * bDx + bDy * bDy + bDz * bDz);
        if (boneDist < 8) {
          hu = 700 + (Math.random() - 0.5) * 40;
        }

        // ---- small calcification (r ≈ 3) ----
        const cDx = x - (center + 12);
        const cDy = y - (center - 8);
        const cDz = z - (center + 6);
        const calcDist = Math.sqrt(cDx * cDx + cDy * cDy + cDz * cDz);
        if (calcDist < 3) {
          hu = 300;
        }

        data[idx] = hu;
      }
    }
  }

  return data;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function VolumeRenderer({
  mode = 'synthetic',
  seriesId,
  opacityPreset: initialPreset = 'ct-soft-tissue',
  showControls = false,
  segmentationMask,
  segmentationLabels,
  syntheticData,
  syntheticDims,
  syntheticSpacing,
  syntheticClipIndex,
  syntheticScanAxis = 'z',
  scanView = false,
  scanDirection = 'head_to_feet',
}: VolumeRendererProps) {
  // ---- DOM ref ----
  const containerRef = useRef<HTMLDivElement>(null);

  // ---- VTK object handles (stored in ref to avoid re-render storms) ----
  const vtkRef = useRef<{
    renderWindow: any;
    renderer: any;
    openglRW: any;
    interactor: any;
    volume: any;
    mapper: any;
    imageData: any;
    colorTransfer: any;
    piecewiseFunc: any;
  } | null>(null);

  // ---- Segmentation overlay volume ref (separate from main CT volume) ----
  const segRef = useRef<{
    volume: any;
    mapper: any;
    imageData: any;
    colorTransfer: any;
    piecewiseFunc: any;
  } | null>(null);

  // ---- UI state ----
  const [isReady, setIsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [activePreset, setActivePreset] = useState<PresetName>(initialPreset);
  const [clip, setClip] = useState<ClipState>({ x: 0, y: 0, z: 0 });
  const [segmentationOpacity, setSegmentationOpacity] = useState(DEFAULT_SEG_OPACITY);

  // ------------------------------------------------------------------
  // 1. Initialise vtk.js pipeline
  //    — synthetic: runs once on mount
  //    — dicom: re-runs when seriesId changes
  // ------------------------------------------------------------------
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let cancelled = false;

    const init = async () => {
      setIsLoading(true);
      setLoadError(null);
      setClip({ x: 0, y: 0, z: 0 });

      try {
        // ------ acquire volume data ------
        let volData: {
          scalarData: Float32Array;
          dimensions: [number, number, number];
          spacing: [number, number, number];
          origin: [number, number, number];
        };

        if (mode === 'dicom' && seriesId) {
          // ---- load real DICOM series ----
          const dicomVol: DicomVolumeData = await loadDicomVolume(seriesId);
          if (cancelled) return;
          volData = {
            scalarData: dicomVol.scalarData,
            dimensions: dicomVol.dimensions,
            spacing: dicomVol.spacing,
            origin: dicomVol.origin,
          };
        } else {
          // ---- synthetic phantom ----
          if (syntheticData && syntheticDims) {
            // External phantom data provided (e.g. from backend CT phantom API)
            const sp = syntheticSpacing || [1, 1, 1];
            volData = {
              scalarData: syntheticData,
              dimensions: syntheticDims,
              spacing: sp,
              origin: [
                -(syntheticDims[0] * sp[0]) / 2,
                -(syntheticDims[1] * sp[1]) / 2,
                -(syntheticDims[2] * sp[2]) / 2,
              ],
            };
          } else {
            // Built-in 64³ phantom (legacy / demo)
            volData = {
              scalarData: generateSyntheticVolume(),
              dimensions: [64, 64, 64],
              spacing: [1, 1, 1],
              origin: [-32, -32, -32],
            };
          }
        }

        if (cancelled) return;

        // ------- vtkImageData -------
        const imageData = vtkImageData.newInstance();
        imageData.setDimensions(volData.dimensions);
        imageData.setSpacing(volData.spacing);
        imageData.setOrigin(volData.origin);

        const dataArray = vtkDataArray.newInstance({
          name: 'Scalars',
          values: volData.scalarData,
          numberOfComponents: 1,
        });
        imageData.getPointData().setScalars(dataArray);

        // ------- rendering pipeline -------
        const renderWindow = vtkRenderWindow.newInstance();
        const renderer = vtkRenderer.newInstance();
        renderer.setBackground(0.0, 0.0, 0.0);
        renderWindow.addRenderer(renderer);

        const openglRW = vtkOpenGLRenderWindow.newInstance();
        openglRW.setContainer(container);
        const { width, height } = container.getBoundingClientRect();
        openglRW.setSize(Math.max(width, 1), Math.max(height, 1));
        renderWindow.addView(openglRW);

        // ------- volume mapper -------
        const mapper = vtkVolumeMapper.newInstance();
        mapper.setInputData(imageData);

        // ------- volume actor -------
        const volume = vtkVolume.newInstance();
        volume.setMapper(mapper);

        // ------- transfer functions (initial preset) -------
        const colorTransfer = vtkColorTransferFunction.newInstance();
        const piecewiseFunc = vtkPiecewiseFunction.newInstance();
        applyPreset(colorTransfer, piecewiseFunc, PRESETS[activePreset]);

        const property = volume.getProperty();
        property.setRGBTransferFunction(0, colorTransfer);
        property.setScalarOpacity(0, piecewiseFunc);
        property.setInterpolationTypeToLinear();
        property.setShade(true);
        property.setAmbient(0.1);
        property.setDiffuse(0.7);
        property.setSpecular(0.2);
        property.setSpecularPower(10.0);

        property.setIndependentComponents(true);
        property.setUseGradientOpacity(0, false);

        renderer.addVolume(volume);
        renderer.resetCamera();

        // ------- Scan-view camera orientation -------
        // When scanView is enabled, orient the camera so the progressive
        // scan builds from TOP (head/chest) to BOTTOM (abdomen/pelvis).
        // We position the camera in +Y (anterior) and set viewUp to -Z,
        // which makes +Z (the head→feet axis) point downward on screen.
        if (scanView) {
          const camera = renderer.getActiveCamera();
          const fp = camera.getFocalPoint();
          const pos = camera.getPosition();
          const dist = Math.sqrt(
            (pos[0] - fp[0]) ** 2 +
            (pos[1] - fp[1]) ** 2 +
            (pos[2] - fp[2]) ** 2,
          );
          // Camera from anterior (+Y), looking at volume center
          camera.setPosition(fp[0], fp[1] + dist, fp[2]);
          // -Z is screen-up → +Z (head→feet) is screen-down
          camera.setViewUp(0, 0, -1);
          renderer.resetCameraClippingRange();
        }

        // ------- mouse interaction -------
        const interactor = vtkRenderWindowInteractor.newInstance();
        interactor.setView(openglRW);
        interactor.initialize();
        interactor.setInteractorStyle(
          vtkInteractorStyleTrackballCamera.newInstance(),
        );
        interactor.bindEvents(container);

        // ------- first render -------
        renderWindow.render();

        // ------- store handles for cleanup / preset updates -------
        vtkRef.current = {
          renderWindow,
          renderer,
          openglRW,
          interactor,
          volume,
          mapper,
          imageData,
          colorTransfer,
          piecewiseFunc,
        };

        if (!cancelled) {
          setIsReady(true);
          setIsLoading(false);
          setLoadError(null);
        }
      } catch (error: any) {
        console.error('[VolumeRenderer] Initialization failed:', error);
        if (!cancelled) {
          setIsLoading(false);
          setLoadError(error.message ?? 'Unknown error loading volume');
        }
      }
    };

    // Defer to next microtask so the container has been laid out
    const timer = setTimeout(init, 0);

    // ------- cleanup -------
    return () => {
      cancelled = true;
      clearTimeout(timer);

      // Cleanup segmentation overlay first
      const seg = segRef.current;
      if (seg) {
        try {
          if (vtkRef.current?.renderer) {
            vtkRef.current.renderer.removeVolume(seg.volume);
          }
        } catch (_) { /* ignore */ }
        try { seg.volume.delete(); } catch (_) { /* ignore */ }
        try { seg.mapper.delete(); } catch (_) { /* ignore */ }
        try { seg.imageData.delete(); } catch (_) { /* ignore */ }
        try { seg.colorTransfer.delete(); } catch (_) { /* ignore */ }
        try { seg.piecewiseFunc.delete(); } catch (_) { /* ignore */ }
        segRef.current = null;
      }

      const s = vtkRef.current;
      if (s) {
        try {
          s.interactor.unbindEvents(container);
        } catch (_) {
          /* ignore */
        }
        try {
          s.openglRW.setContainer(null);
        } catch (_) {
          /* ignore */
        }
        try {
          s.renderer.removeVolume(s.volume);
        } catch (_) {
          /* ignore */
        }
        try {
          s.volume.delete();
        } catch (_) {
          /* ignore */
        }
        try {
          s.mapper.delete();
        } catch (_) {
          /* ignore */
        }
        try {
          s.openglRW.delete();
        } catch (_) {
          /* ignore */
        }
        try {
          s.renderer.delete();
        } catch (_) {
          /* ignore */
        }
        try {
          s.renderWindow.delete();
        } catch (_) {
          /* ignore */
        }
        vtkRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, seriesId, syntheticData, syntheticDims, syntheticSpacing]);

  // ------------------------------------------------------------------
  // 2. Handle preset changes (update transfer functions in-place)
  // ------------------------------------------------------------------
  useEffect(() => {
    const state = vtkRef.current;
    if (!state || !isReady) return;

    const newColor = vtkColorTransferFunction.newInstance();
    const newOpacity = vtkPiecewiseFunction.newInstance();
    applyPreset(newColor, newOpacity, PRESETS[activePreset]);

    state.volume.getProperty().setRGBTransferFunction(0, newColor);
    state.volume.getProperty().setScalarOpacity(0, newOpacity);

    // Replace stored refs so we don't leak old TF objects
    state.colorTransfer = newColor;
    state.piecewiseFunc = newOpacity;

    state.renderWindow.render();
  }, [activePreset, isReady]);

  // ------------------------------------------------------------------
  // 3. Clipping plane management — user clips + progressive scan clip
  // ------------------------------------------------------------------
  useEffect(() => {
    const state = vtkRef.current;
    if (!state || !isReady) return;

    const { mapper, imageData, renderWindow } = state;
    if (!mapper || !imageData || !renderWindow) return;

    const bounds = imageData.getBounds();
    // bounds: [xMin, xMax, yMin, yMax, zMin, zMax]
    const activePlanes: any[] = [];

    // ---- User clipping planes (slider-controlled) ----
    // X axis (normal points +X, clip negative side)
    if (clip.x > 0) {
      const xPos = bounds[0] + clip.x * (bounds[1] - bounds[0]);
      activePlanes.push(
        vtkPlane.newInstance({ normal: [1, 0, 0], origin: [xPos, 0, 0] }),
      );
    }

    // Y axis (normal points +Y, clip negative side)
    if (clip.y > 0) {
      const yPos = bounds[2] + clip.y * (bounds[3] - bounds[2]);
      activePlanes.push(
        vtkPlane.newInstance({ normal: [0, 1, 0], origin: [0, yPos, 0] }),
      );
    }

    // Z axis (normal points +Z, clip negative side)
    if (clip.z > 0) {
      const zPos = bounds[4] + clip.z * (bounds[5] - bounds[4]);
      activePlanes.push(
        vtkPlane.newInstance({ normal: [0, 0, 1], origin: [0, 0, zPos] }),
      );
    }

    // ---- Progressive scan clipping ----
    // Hides voxels with data-index > clipIndex along the scan axis.
    // A plane whose normal points NEGATIVE clips the POSITIVE side,
    // so voxels after the clip position are hidden.
    if (
      syntheticClipIndex !== undefined &&
      syntheticClipIndex >= 0 &&
      syntheticScanAxis
    ) {
      const dims = imageData.getDimensions();   // [x, y, z]
      const spacing = imageData.getSpacing();   // [x, y, z]
      const origin = imageData.getOrigin();      // [x, y, z]

      const axisIdx = syntheticScanAxis === 'x' ? 0
        : syntheticScanAxis === 'y' ? 1 : 2;
      const axisDim = dims[axisIdx];

      // Only add plane when we haven't reached the final slice yet
      if (syntheticClipIndex < axisDim - 1) {
        // Position the plane just past the last visible slice
        const clipWorld =
          origin[axisIdx] + (syntheticClipIndex + 0.5) * spacing[axisIdx];

        const normal: [number, number, number] = [0, 0, 0];
        normal[axisIdx] = -1; // negative normal → clips positive side

        const planeOrigin: [number, number, number] = [0, 0, 0];
        planeOrigin[axisIdx] = clipWorld;

        activePlanes.push(
          vtkPlane.newInstance({ normal, origin: planeOrigin }),
        );
      }
    }

    mapper.removeAllClippingPlanes();
    for (const plane of activePlanes) {
      mapper.addClippingPlane(plane);
    }
    renderWindow.render();
     
  }, [clip, syntheticClipIndex, syntheticScanAxis, isReady]);

  // ------------------------------------------------------------------
  // 4. Segmentation overlay — add/remove a second volume for mask
  // ------------------------------------------------------------------
  useEffect(() => {
    const state = vtkRef.current;
    if (!state || !isReady) return;

    const { renderer, renderWindow, imageData: ctImageData } = state;

    // Cleanup previous segmentation overlay
    if (segRef.current) {
      try {
        renderer.removeVolume(segRef.current.volume);
        segRef.current.volume.delete();
        segRef.current.mapper.delete();
        segRef.current.imageData.delete();
        segRef.current.colorTransfer.delete();
        segRef.current.piecewiseFunc.delete();
      } catch (_) { /* ignore */ }
      segRef.current = null;
    }

    // If no mask provided, we're done
    if (!segmentationMask || !segmentationLabels || segmentationLabels.length === 0) {
      renderWindow.render();
      return;
    }

    try {
      // Get CT volume dimensions
      const ctDims = ctImageData.getDimensions(); // [x, y, z]
      const spacing = ctImageData.getSpacing();
      const origin = ctImageData.getOrigin();

      const ctVoxelCount = ctDims[0] * ctDims[1] * ctDims[2];
      const maskVoxelCount = segmentationMask.length;

      console.info(
        '[VolumeRenderer] CT dims=%o (%d voxels) | mask elements=%d',
        ctDims, ctVoxelCount, maskVoxelCount,
      );

      // Validate dimension match — critical for correct rendering
      if (maskVoxelCount !== ctVoxelCount) {
        console.error(
          '[VolumeRenderer] Dimension mismatch! CT=%d voxels, mask=%d voxels. ' +
          'The segmentation mask was generated for a different volume. Skipping overlay.',
          ctVoxelCount, maskVoxelCount,
        );
        renderWindow.render();
        return;
      }

      // ------- segmentation vtkImageData -------
      const segImageData = vtkImageData.newInstance();
      segImageData.setDimensions(ctDims);
      segImageData.setSpacing(spacing);
      segImageData.setOrigin(origin);
      // Set origin to match CT so segmentation aligns in world space.

      const segArray = vtkDataArray.newInstance({
        name: 'Segmentation',
        values: segmentationMask,
        numberOfComponents: 1,
      });
      segImageData.getPointData().setScalars(segArray);

      // Determine the scalar range from the data
      let segMin = Infinity;
      let segMax = -Infinity;
      for (let i = 0; i < segmentationMask.length; i++) {
        const v = segmentationMask[i];
        if (v < segMin) segMin = v;
        if (v > segMax) segMax = v;
      }
      console.info('[VolumeRenderer] Segmentation scalar range: %d – %d', segMin, segMax);

      // ------- color / opacity transfer functions -------
      const segColor = vtkColorTransferFunction.newInstance();
      const segOpacity = vtkPiecewiseFunction.newInstance();

      // Background (0) — fully transparent
      segOpacity.addPoint(-0.5, 0.0);
      segOpacity.addPoint(0.0, 0.0);
      segOpacity.addPoint(0.5, 0.0);

      // Each label gets its color and partial opacity.
      // Use narrow bands around integer label values so that
      // interpolation between labels stays transparent.
      for (const label of segmentationLabels) {
        const [r, g, b] = label.color;
        const idx = label.index;
        if (idx === 0) continue; // background handled above

        segColor.addRGBPoint(idx, r / 255, g / 255, b / 255);

        // Opacity: transparent → visible → transparent
        segOpacity.addPoint(idx - LABEL_HALF_BAND, 0.0);
        segOpacity.addPoint(idx, segmentationOpacity);  // peak opacity
        segOpacity.addPoint(idx + LABEL_HALF_BAND, 0.0);
      }

      // ------- segmentation volume actor -------
      const segMapper = vtkVolumeMapper.newInstance();
      segMapper.setInputData(segImageData);
      segMapper.setBlendModeToComposite();

      // Critical: tell the mapper the exact data range so the transfer
      // function is sampled correctly.
      if (typeof segMapper.setSampleDistance === 'function') {
        // Use the smallest spacing dimension as a reasonable sample distance
        const minSpacing = Math.min(spacing[0], spacing[1], spacing[2]);
        segMapper.setSampleDistance(minSpacing * 0.5);
      }

      const segVolume = vtkVolume.newInstance();
      segVolume.setMapper(segMapper);

      // Higher render priority → rendered later → on top of CT
      if (typeof segVolume.getProperty === 'function') {
        // vtk.js uses Visibility, not RenderPriority for ordering.
        // Setting a higher "layer" via mapper ensures the segmentation
        // is sampled after the CT in the ray-casting pass.
      }

      const segProp = segVolume.getProperty();
      segProp.setRGBTransferFunction(0, segColor);
      segProp.setScalarOpacity(0, segOpacity);
      segProp.setInterpolationTypeToNearest(); // nearest-neighbor for labels
      segProp.setShade(false);                 // no shading for labels
      segProp.setIndependentComponents(true);

      renderer.addVolume(segVolume);
      renderWindow.render();

      segRef.current = {
        volume: segVolume,
        mapper: segMapper,
        imageData: segImageData,
        colorTransfer: segColor,
        piecewiseFunc: segOpacity,
      };

      console.info('[VolumeRenderer] Segmentation overlay registered (labels=%d)', segmentationLabels.length);
    } catch (err) {
      console.error('[VolumeRenderer] Segmentation overlay failed:', err);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [segmentationMask, segmentationLabels, isReady]);

  // ------------------------------------------------------------------
  // 4b. Segmentation opacity live update — when the slider is dragged,
  //     rebuild only the opacity transfer function (no volume recreation).
  // ------------------------------------------------------------------
  useEffect(() => {
    const seg = segRef.current;
    const state = vtkRef.current;
    if (!seg || !state || !isReady) return;

    const labels = segmentationLabels;
    if (!labels || labels.length === 0) return;

    const newOpacity = vtkPiecewiseFunction.newInstance();

    // Background — fully transparent
    newOpacity.addPoint(-0.5, 0.0);
    newOpacity.addPoint(0.0, 0.0);
    newOpacity.addPoint(0.5, 0.0);

    // Each label with the new opacity value
    for (const label of labels) {
      if (label.index === 0) continue;
      newOpacity.addPoint(label.index - LABEL_HALF_BAND, 0.0);
      newOpacity.addPoint(label.index, segmentationOpacity);
      newOpacity.addPoint(label.index + LABEL_HALF_BAND, 0.0);
    }

    // Replace the opacity function on the volume property (live update)
    seg.volume.getProperty().setScalarOpacity(0, newOpacity);
    // Store for proper cleanup on next segmentation change
    seg.piecewiseFunc = newOpacity;

    state.renderWindow.render();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [segmentationOpacity, isReady]);

  // ------------------------------------------------------------------
  // 5. Handle container resize
  // ------------------------------------------------------------------
  useEffect(() => {
    const container = containerRef.current;
    const state = vtkRef.current;
    if (!container || !state || !isReady) return;

    const observer = new ResizeObserver(() => {
      const { width, height } = container.getBoundingClientRect();
      if (width > 0 && height > 0) {
        state.openglRW.setSize(width, height);
        state.renderWindow.render();
      }
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, [isReady]);

  // ------------------------------------------------------------------
  // 5. Preset button handler
  // ------------------------------------------------------------------
  const handlePresetChange = useCallback((name: PresetName) => {
    setActivePreset(name);
  }, []);

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------
  return (
    <div
      className="relative h-full w-full overflow-hidden bg-black"
      onContextMenu={(event) => event.preventDefault()}
    >
      {/* Orientation label */}
      <div className="pointer-events-none absolute left-2 top-2 z-10">
        <span className="rounded bg-black/60 px-2 py-0.5 text-xs font-medium uppercase text-white">
          3D Volume
        </span>
      </div>

      {/* Loading indicator */}
      {isLoading && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/50">
          <div className="flex flex-col items-center gap-2">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            <span className="text-xs text-muted-foreground">
              {mode === 'dicom' ? 'Loading DICOM volume...' : 'Loading volume...'}
            </span>
          </div>
        </div>
      )}

      {/* Error overlay */}
      {loadError && !isLoading && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/80">
          <div className="max-w-sm text-center">
            <p className="mb-1 text-sm font-medium text-red-400">
              Failed to load volume
            </p>
            <p className="text-xs text-white/50">{loadError}</p>
          </div>
        </div>
      )}

      {/* vtk.js container */}
      <div
        ref={containerRef}
        className="h-full w-full"
        style={{ opacity: isReady ? 1 : 0 }}
      />

      {/* ---- Scan-view overlays (scan plane + direction labels) ---- */}
      {scanView && isReady && syntheticDims && syntheticClipIndex !== undefined && syntheticClipIndex >= 0 && (() => {
        const axisIdx = syntheticScanAxis === 'x' ? 0 : syntheticScanAxis === 'y' ? 1 : 2;
        const totalDepth = syntheticDims[axisIdx];
        const scanFraction = totalDepth > 1
          ? (syntheticClipIndex + 0.5) / totalDepth
          : 0;
        const topPct = scanFraction * 100;
        const isHeadFirst = scanDirection === 'head_to_feet';
        const topLabel = isHeadFirst ? 'TOP / Head-Chest' : 'TOP / Abdomen-Pelvis';
        const bottomLabel = isHeadFirst ? 'BOTTOM / Abdomen-Pelvis' : 'BOTTOM / Head-Chest';
        return (
          <>
            {/* Direction labels */}
            <div className="pointer-events-none absolute inset-x-0 top-1 z-10 flex justify-center">
              <span className="rounded bg-black/60 px-2 py-0.5 text-[10px] font-medium text-cyan-300/80">
                ▲ {topLabel}
              </span>
            </div>
            <div className="pointer-events-none absolute inset-x-0 bottom-1 z-10 flex justify-center">
              <span className="rounded bg-black/60 px-2 py-0.5 text-[10px] font-medium text-cyan-300/80">
                ▼ {bottomLabel}
              </span>
            </div>

            {/* Scan plane — horizontal cyan line at current clip position */}
            <div
              className="pointer-events-none absolute inset-x-0 z-10"
              style={{ top: `${topPct}%` }}
            >
              {/* Glow line */}
              <div className="h-px w-full bg-cyan-400/60 shadow-[0_0_8px_2px_rgba(34,211,238,0.4)]" />
              {/* Scan arrow indicator on the right side */}
              <div className="absolute right-2 -translate-y-1/2">
                <span className="text-[9px] text-cyan-300/70 tabular-nums">
                  slice {syntheticClipIndex + 1}/{totalDepth}
                </span>
              </div>
            </div>
          </>
        );
      })()}

      {/* Controls overlay */}
      {showControls && isReady && (
        <div className="absolute bottom-2 right-2 z-10 flex flex-col gap-1.5 items-end">
          {/* Preset buttons */}
          <div className="flex gap-1">
            <PresetButton
              label="Bone"
              active={activePreset === 'ct-bone'}
              onClick={() => handlePresetChange('ct-bone')}
            />
            <PresetButton
              label="Soft"
              active={activePreset === 'ct-soft-tissue'}
              onClick={() => handlePresetChange('ct-soft-tissue')}
            />
            <PresetButton
              label="Lung"
              active={activePreset === 'ct-lung'}
              onClick={() => handlePresetChange('ct-lung')}
            />
            <PresetButton
              label="Angio"
              active={activePreset === 'ct-angio'}
              onClick={() => handlePresetChange('ct-angio')}
            />
          </div>

          {/* Segmentation opacity slider — only when overlay is active */}
          {segmentationMask && segmentationLabels && segmentationLabels.length > 0 && (
            <div className="flex flex-col gap-0.5 rounded bg-black/60 px-2 py-1.5 min-w-[220px]">
              <SegOpacitySlider
                value={segmentationOpacity}
                onChange={setSegmentationOpacity}
              />
            </div>
          )}

          {/* Clipping plane sliders */}
          <div className="flex flex-col gap-0.5 rounded bg-black/60 px-2 py-1.5 min-w-[220px]">
            <ClipSlider
              label="X"
              value={clip.x}
              onChange={(v) => setClip((prev) => ({ ...prev, x: v }))}
            />
            <ClipSlider
              label="Y"
              value={clip.y}
              onChange={(v) => setClip((prev) => ({ ...prev, y: v }))}
            />
            <ClipSlider
              label="Z"
              value={clip.z}
              onChange={(v) => setClip((prev) => ({ ...prev, z: v }))}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Apply a transfer-function preset to brand-new TF instances. */
function applyPreset(
  colorTF: any,
  opacityTF: any,
  preset: TransferFunctionPreset,
): void {
  for (const [x, r, g, b] of preset.colorPoints) {
    colorTF.addRGBPoint(x, r, g, b);
  }
  for (const [x, y] of preset.opacityPoints) {
    opacityTF.addPoint(x, y);
  }
}

/** Small slider for a single clipping axis. */
function ClipSlider({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="w-3 text-right text-[10px] font-medium text-white/50">
        {label}
      </span>
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="h-1 flex-1 cursor-pointer accent-primary"
      />
      <span className="w-8 text-right text-[10px] tabular-nums text-white/50">
        {Math.round(value * 100)}%
      </span>
    </div>
  );
}

/** Small internal button for the preset bar. */
function PresetButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded px-2 py-1 text-xs transition-colors ${
        active
          ? 'bg-primary text-primary-foreground'
          : 'bg-black/60 text-white/80 hover:bg-black/80'
      }`}
    >
      {label}
    </button>
  );
}

/** Slider for adjusting segmentation overlay opacity. */
function SegOpacitySlider({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="w-auto text-[10px] font-medium text-white/70 whitespace-nowrap">
        Opacity
      </span>
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="h-1 flex-1 cursor-pointer accent-primary"
      />
      <span className="w-8 text-right text-[10px] tabular-nums text-white/50">
        {Math.round(value * 100)}%
      </span>
    </div>
  );
}
