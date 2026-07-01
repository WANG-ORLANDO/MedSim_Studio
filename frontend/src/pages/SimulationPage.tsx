import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Button } from '@components/ui/button';
import { useSimulationStore } from '@store/useSimulationStore';
import { simulationService } from '@/services/simulationService';
import type { PhantomResponse, DicomLesionPreviewResponse } from '@/services/simulationService';
import { VolumeRenderer } from '@vtk/volumeRendering/VolumeRenderer';
import { dicomService } from '@/services/dicomService';
import type {
  CtParamsPreviewParams,
  CtParamsPreviewResponse,
  LesionConfig,
  LesionType,
  LesionShape,
} from '@/types/simulation';
import type { DicomStudy, DicomSeries } from '@/types/dicom';

// ---------------------------------------------------------------------------
// Window/Level presets (WL, WW in HU)
// ---------------------------------------------------------------------------

interface WindowPreset {
  label: string;
  windowLevel: number;
  windowWidth: number;
}

const WINDOW_PRESETS: WindowPreset[] = [
  { label: 'Soft', windowLevel: 40, windowWidth: 400 },
  { label: 'Lung', windowLevel: -600, windowWidth: 1500 },
  { label: 'Bone', windowLevel: 500, windowWidth: 2000 },
];

// ---------------------------------------------------------------------------
// Organ color map — RGB 0-255 for label overlay on axial slices
// ---------------------------------------------------------------------------

const ORGAN_COLORS: Record<number, [number, number, number]> = {
  1: [255, 255, 0],   // left_adrenal_gland — yellow
  2: [255, 255, 0],   // right_adrenal_gland — yellow
  3: [139, 69, 19],   // colon — brown
  4: [255, 192, 203], // duodenum — pink
  5: [173, 216, 230], // esophagus — light blue
  6: [0, 255, 0],     // gallbladder — green
  7: [255, 165, 0],   // left_kidney — orange
  8: [255, 165, 0],   // right_kidney — orange
  9: [139, 0, 0],     // liver — dark red
  10: [0, 200, 255],  // left_lung_lower_lobe — cyan
  11: [0, 200, 255],  // right_lung_lower_lobe — cyan
  12: [0, 200, 255],  // right_lung_middle_lobe — cyan
  13: [100, 200, 255],// left_lung_upper_lobe — light cyan
  14: [100, 200, 255],// right_lung_upper_lobe — light cyan
  15: [255, 255, 100],// pancreas — light yellow
  16: [255, 182, 193],// small_bowel — light pink
  17: [128, 0, 128],  // spleen — purple
  18: [210, 180, 140],// stomach — tan
  19: [0, 255, 255],  // trachea — cyan
  20: [0, 0, 255],    // urinary_bladder — blue
};

/** Semi-transparency alpha for organ label overlay (0-1) */
const LABEL_OVERLAY_ALPHA = 0.35;

// ---------------------------------------------------------------------------
// HU → grayscale helper
//    windowLevel = WL, windowWidth = WW
//    visible range: [WL - WW/2,  WL + WW/2]
// ---------------------------------------------------------------------------

function applyWindowLevel(
  hu: number,
  windowLevel: number,
  windowWidth: number,
): number {
  const low = windowLevel - windowWidth / 2;
  const high = windowLevel + windowWidth / 2;
  if (hu <= low) return 0;
  if (hu >= high) return 255;
  return ((hu - low) / windowWidth) * 255;
}

// ---------------------------------------------------------------------------
// Decode base64 → Float32Array
// ---------------------------------------------------------------------------

function decodeBase64ToFloat32(base64: string): Float32Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}

function renderVolumeSliceToCanvas({
  canvas,
  width,
  height,
  depth,
  volumeData,
  sliceIndex,
  windowLevel,
  windowWidth,
  labelData,
  showLabelOverlay = false,
  pickedPosition,
}: {
  canvas: HTMLCanvasElement;
  width: number;
  height: number;
  depth: number;
  volumeData: Float32Array;
  sliceIndex: number;
  windowLevel: number;
  windowWidth: number;
  labelData?: Uint8Array | null;
  showLabelOverlay?: boolean;
  pickedPosition?: { x: number; y: number; z: number } | null;
}): void {
  if (sliceIndex < 0 || sliceIndex >= depth) return;

  const sliceSize = width * height;
  const offset = sliceIndex * sliceSize;
  const slice = volumeData.subarray(offset, offset + sliceSize);
  const labelSlice =
    labelData && showLabelOverlay
      ? labelData.subarray(offset, offset + sliceSize)
      : null;

  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const imageData = ctx.createImageData(width, height);

  for (let i = 0; i < sliceSize; i++) {
    const gray = applyWindowLevel(slice[i], windowLevel, windowWidth);
    const pixelIdx = i * 4;

    if (labelSlice && labelSlice[i] > 0) {
      const organColor = ORGAN_COLORS[labelSlice[i]];
      if (organColor) {
        const alpha = LABEL_OVERLAY_ALPHA;
        imageData.data[pixelIdx] = gray * (1 - alpha) + organColor[0] * alpha;
        imageData.data[pixelIdx + 1] = gray * (1 - alpha) + organColor[1] * alpha;
        imageData.data[pixelIdx + 2] = gray * (1 - alpha) + organColor[2] * alpha;
        imageData.data[pixelIdx + 3] = 255;
      } else {
        imageData.data[pixelIdx] = gray;
        imageData.data[pixelIdx + 1] = gray;
        imageData.data[pixelIdx + 2] = gray;
        imageData.data[pixelIdx + 3] = 255;
      }
    } else {
      imageData.data[pixelIdx] = gray;
      imageData.data[pixelIdx + 1] = gray;
      imageData.data[pixelIdx + 2] = gray;
      imageData.data[pixelIdx + 3] = 255;
    }
  }

  ctx.putImageData(imageData, 0, 0);

  if (pickedPosition && pickedPosition.z === sliceIndex) {
    const cx = pickedPosition.x;
    const cy = pickedPosition.y;
    const size = 12;

    ctx.save();
    ctx.strokeStyle = '#ff3333';
    ctx.lineWidth = 2;
    ctx.shadowColor = 'rgba(0,0,0,0.8)';
    ctx.shadowBlur = 3;

    ctx.beginPath();
    ctx.moveTo(cx - size, cy);
    ctx.lineTo(cx + size, cy);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(cx, cy - size);
    ctx.lineTo(cx, cy + size);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(cx, cy, size + 4, 0, 2 * Math.PI);
    ctx.stroke();

    ctx.restore();
  }
}

// ---------------------------------------------------------------------------
// Default form values
// ---------------------------------------------------------------------------

const DEFAULT_FORM: {
  lesionType: LesionType;
  shape: LesionShape;
  huMean: number;
  huStd: number;
  diameter: number;
  marginSharpness: number;
  spiculationDegree: number;
  centerX: number;
  centerY: number;
  centerZ: number;
  normalizedCenterX: number;
  normalizedCenterY: number;
  normalizedCenterZ: number;
} = {
  lesionType: 'tumor',
  shape: 'spherical',
  huMean: 40,
  huStd: 20,
  diameter: 20,
  marginSharpness: 0.8,
  spiculationDegree: 0,
  centerX: 0,
  centerY: 0,
  centerZ: 0,
  normalizedCenterX: 0,
  normalizedCenterY: 0,
  normalizedCenterZ: 0,
};

const DEFAULT_CT_PARAMS: CtParamsPreviewParams = {
  sliceThicknessMm: 5.0,
  doseLevel: 'standard',
  mAs: 150,
  kVp: 120,
  pitch: 1.2,
  fovMm: 250,
  matrixSize: 256,
  kernel: 'bone',
  contrastPhase: 'venous',
};

const DEFAULT_MAS = 150;
const MIN_MAS = 30;
const MAX_MAS = 300;

const lesionTypeLabel: Record<LesionType, string> = {
  tumor: 'Tumor',
  nodule: 'Nodule',
  cyst: 'Cyst',
  calcification: 'Calcification',
  metastasis: 'Metastasis',
};

const shapeLabel: Record<LesionShape, string> = {
  spherical: 'Spherical',
  ellipsoidal: 'Ellipsoidal',
  irregular: 'Irregular',
  lobulated: 'Lobulated',
  spiculated: 'Spiculated',
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * SimulationPage
 *
 * Provides two modes:
 * 1. Lesion/Organ/Deformation simulation (existing) — configures and runs
 *    simulation jobs for lesion/organ synthesis and result export.
 * 2. CT Phantom (new) — generates a synthetic upper-body CT phantom from
 *    the backend, displays axial slices with window/level controls, and
 *    shows a 3D volume preview via vtk.js.
 */
export default function SimulationPage() {
  const { lesions, organs, addLesion, removeLesion, activeJobs, completedJobs, addJob } =
    useSimulationStore();

  // ---- Tab state ----
  const [activeTab, setActiveTab] = useState<
    'lesion' | 'organ' | 'deformation' | 'phantom'
  >('lesion');

  // ---- Source volume state ----
  const [sourceType, setSourceType] = useState<'synthetic' | 'dicom'>('synthetic');
  const [studies, setStudies] = useState<DicomStudy[]>([]);
  const [seriesList, setSeriesList] = useState<DicomSeries[]>([]);
  const [selectedStudyId, setSelectedStudyId] = useState<string | null>(null);
  const [selectedSeriesId, setSelectedSeriesId] = useState<string | null>(null);
  const [loadingStudies, setLoadingStudies] = useState(false);
  const [loadingSeries, setLoadingSeries] = useState(false);

  // ---- Form state ----
  const [form, setForm] = useState(DEFAULT_FORM);

  // ---- Preview state ----
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewResult, setPreviewResult] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  // ---- DICOM preview state ----
  const [dicomPreviewLoading, setDicomPreviewLoading] = useState(false);
  const [dicomPreviewImage, setDicomPreviewImage] = useState<string | null>(null);
  const [dicomPreviewStats, setDicomPreviewStats] = useState<string | null>(null);
  const [dicomPreviewError, setDicomPreviewError] = useState<string | null>(null);

  // ---- Job running state ----
  const [jobLoading, setJobLoading] = useState(false);
  const [jobError, setJobError] = useState<string | null>(null);

  // ---- Export state ----
  const [exportFormat, setExportFormat] = useState<'dicom' | 'nifti' | 'nrrd'>('dicom');
  const [exporting, setExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [exportError, setExportError] = useState<string | null>(null);

  const latestCompletedJob = completedJobs[completedJobs.length - 1];

  // ---- CT Phantom state ----
  const [phantom, setPhantom] = useState<PhantomResponse | null>(null);
  const [phantomLoading, setPhantomLoading] = useState(false);
  const [phantomError, setPhantomError] = useState<string | null>(null);
  const [phantomSource, setPhantomSource] = useState<'procedural' | 'atlas'>('procedural');
  const [phantomSize, setPhantomSize] = useState(192);
  const [loadedPhantomSize, setLoadedPhantomSize] = useState<number | null>(null);
  const [sliceIndex, setSliceIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(10); // slices per second
  const [activePreset, setActivePreset] = useState<WindowPreset>(WINDOW_PRESETS[0]);
  const [showLabelOverlay, setShowLabelOverlay] = useState(true);
  const [pickingMode, setPickingMode] = useState(false);
  const [pickedPosition, setPickedPosition] = useState<{ x: number; y: number; z: number } | null>(null);
  const [ctParams, setCtParams] = useState<CtParamsPreviewParams>(DEFAULT_CT_PARAMS);
  const [mAsInput, setMAsInput] = useState(String(DEFAULT_CT_PARAMS.mAs));
  const [ctParamsLoading, setCtParamsLoading] = useState(false);
  const [ctParamsError, setCtParamsError] = useState<string | null>(null);
  const [ctParamsResult, setCtParamsResult] = useState<CtParamsPreviewResponse | null>(null);
  const [ctParamsCopyState, setCtParamsCopyState] = useState<string | null>(null);
  const [ctParamsDownloadState, setCtParamsDownloadState] = useState<string | null>(null);
  const [standardizedCaseCopyState, setStandardizedCaseCopyState] = useState<string | null>(null);
  const [standardizedCaseDownloadState, setStandardizedCaseDownloadState] = useState<string | null>(null);

  // ---- vtk.js volume data (decoded once from phantom) ----
  const [vtkVolumeData, setVtkVolumeData] = useState<Float32Array | null>(null);
  const [vtkDims, setVtkDims] = useState<[number, number, number] | null>(null);
  const [vtkSpacing, setVtkSpacing] = useState<[number, number, number] | null>(null);

  // ---- Refs ----
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const originalCompareCanvasRef = useRef<HTMLCanvasElement>(null);
  const simulatedCompareCanvasRef = useRef<HTMLCanvasElement>(null);
  const cachedPhantomDataRef = useRef<Float32Array | null>(null);
  const cachedLabelDataRef = useRef<Uint8Array | null>(null);
  const playTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ---- Reset slice index to 0 when a new phantom is loaded ----
  useEffect(() => {
    if (phantom) {
      setSliceIndex(0);
      setPlaying(false);
    }
  }, [phantom]);

  useEffect(() => {
    if (!ctParamsCopyState) return;
    const timer = window.setTimeout(() => setCtParamsCopyState(null), 2000);
    return () => window.clearTimeout(timer);
  }, [ctParamsCopyState]);

  useEffect(() => {
    if (!ctParamsDownloadState) return;
    const timer = window.setTimeout(() => setCtParamsDownloadState(null), 2000);
    return () => window.clearTimeout(timer);
  }, [ctParamsDownloadState]);

  useEffect(() => {
    if (!standardizedCaseCopyState) return;
    const timer = window.setTimeout(() => setStandardizedCaseCopyState(null), 2000);
    return () => window.clearTimeout(timer);
  }, [standardizedCaseCopyState]);

  useEffect(() => {
    if (!standardizedCaseDownloadState) return;
    const timer = window.setTimeout(() => setStandardizedCaseDownloadState(null), 2000);
    return () => window.clearTimeout(timer);
  }, [standardizedCaseDownloadState]);

  // ---- Decode vtk.js data when phantom changes ----
  useEffect(() => {
    if (!phantom) {
      setVtkVolumeData(null);
      setVtkDims(null);
      setVtkSpacing(null);
      setLoadedPhantomSize(null);
      cachedPhantomDataRef.current = null;
      cachedLabelDataRef.current = null;
      return;
    }
    try {
      const data = decodeBase64ToFloat32(phantom.volumeBase64);
      // Cache the original (z,y,x)-ordered data for slice rendering
      cachedPhantomDataRef.current = data;

      // Decode label data if present
      if (phantom.labelBase64) {
        try {
          const binary = atob(phantom.labelBase64);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
          }
          cachedLabelDataRef.current = bytes;
        } catch {
          cachedLabelDataRef.current = null;
        }
      } else {
        cachedLabelDataRef.current = null;
      }

      const { width, height, depth, spacing } = phantom.metadata;
      // Volume stored as (z, y, x) in backend; vtk.js expects (x, y, z)
      // so we need to transpose. Build transposed volume: x fastest, then y, then z.
      const xDim = width;
      const yDim = height;
      const zDim = depth;
      const transposed = new Float32Array(xDim * yDim * zDim);
      for (let z = 0; z < zDim; z++) {
        for (let y = 0; y < yDim; y++) {
          for (let x = 0; x < xDim; x++) {
            const srcIdx = z * yDim * xDim + y * xDim + x;
            const dstIdx = x + y * xDim + z * xDim * yDim;
            transposed[dstIdx] = data[srcIdx];
          }
        }
      }
      setVtkVolumeData(transposed);
      setVtkDims([xDim, yDim, zDim]);
      setVtkSpacing([spacing[2], spacing[1], spacing[0]]); // (z,y,x)→(x,y,z)
    } catch (err: any) {
      console.error('[SimulationPage] Failed to decode phantom volume:', err);
      setVtkVolumeData(null);
      setVtkDims(null);
      setVtkSpacing(null);
      cachedPhantomDataRef.current = null;
      cachedLabelDataRef.current = null;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phantom?.volumeBase64, phantom?.labelBase64]);

  const simulatedVolumeData = useMemo(() => {
    if (!ctParamsResult?.simulatedVolumeBase64) return null;
    try {
      return decodeBase64ToFloat32(ctParamsResult.simulatedVolumeBase64);
    } catch (err) {
      console.error('[SimulationPage] Failed to decode simulated CT volume:', err);
      return null;
    }
  }, [ctParamsResult?.simulatedVolumeBase64]);

  // ---- Axial slice canvas rendering ----
  const renderSlice = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !phantom) return;

    const { width, height, depth } = phantom.metadata;
    const data = cachedPhantomDataRef.current;
    if (!data) return;
    const { windowLevel, windowWidth } = activePreset;
    renderVolumeSliceToCanvas({
      canvas,
      width,
      height,
      depth,
      volumeData: data,
      sliceIndex,
      windowLevel,
      windowWidth,
      labelData: cachedLabelDataRef.current,
      showLabelOverlay,
      pickedPosition,
    });
  }, [phantom, sliceIndex, activePreset, showLabelOverlay, pickedPosition]);

  // Re-render on slice/preset change
  useEffect(() => {
    renderSlice();
  }, [renderSlice]);

  useEffect(() => {
    const canvas = originalCompareCanvasRef.current;
    const data = cachedPhantomDataRef.current;
    if (!canvas || !phantom || !data) return;
    const { width, height, depth } = phantom.metadata;
    renderVolumeSliceToCanvas({
      canvas,
      width,
      height,
      depth,
      volumeData: data,
      sliceIndex,
      windowLevel: activePreset.windowLevel,
      windowWidth: activePreset.windowWidth,
    });
  }, [phantom, sliceIndex, activePreset]);

  useEffect(() => {
    const canvas = simulatedCompareCanvasRef.current;
    if (!canvas || !phantom || !simulatedVolumeData) return;
    const { width, height, depth } = phantom.metadata;
    renderVolumeSliceToCanvas({
      canvas,
      width,
      height,
      depth,
      volumeData: simulatedVolumeData,
      sliceIndex,
      windowLevel: activePreset.windowLevel,
      windowWidth: activePreset.windowWidth,
    });
  }, [phantom, simulatedVolumeData, sliceIndex, activePreset]);

  // ---- Playback effect ----
  useEffect(() => {
    if (!playing || !phantom) return;

    const intervalMs = 1000 / Math.max(playSpeed, 1);
    playTimerRef.current = setInterval(() => {
      setSliceIndex((prev) => {
        const maxIdx = phantom.metadata.depth - 1;
        if (prev >= maxIdx) {
          // Stop at end (loop disabled by default for MVP)
          setPlaying(false);
          return maxIdx;
        }
        return prev + 1;
      });
    }, intervalMs);

    return () => {
      if (playTimerRef.current) {
        clearInterval(playTimerRef.current);
        playTimerRef.current = null;
      }
    };
  }, [playing, playSpeed, phantom]);

  // ---- Load available DICOM studies on mount ----
  useEffect(() => {
    let mounted = true;
    const loadStudies = async () => {
      setLoadingStudies(true);
      try {
        const res = await dicomService.getStudies(1, 50);
        if (mounted) setStudies(res.items ?? []);
      } catch {
        // Silently ignore — studies fetch is non-critical
      } finally {
        if (mounted) setLoadingStudies(false);
      }
    };
    loadStudies();
    return () => { mounted = false; };
  }, []);

  // ---- Load series when a study is selected ----
  useEffect(() => {
    if (!selectedStudyId) {
      setSeriesList([]);
      setSelectedSeriesId(null);
      return;
    }
    let mounted = true;
    const loadSeries = async () => {
      setLoadingSeries(true);
      setSelectedSeriesId(null);
      try {
        const series = await dicomService.getSeries(selectedStudyId);
        // Filter to CT/MR imaging modalities only
        const IMAGE_MODALITIES = ['CT', 'MR', 'PT', 'NM', 'US', 'XA', 'CR'];
        if (mounted) setSeriesList(series.filter((s) => IMAGE_MODALITIES.includes(s.modality)));
      } catch {
        // Silently ignore
      } finally {
        if (mounted) setLoadingSeries(false);
      }
    };
    loadSeries();
    return () => { mounted = false; };
  }, [selectedStudyId]);

  // -----------------------------------------------------------------------
  // Handlers: phantom
  // -----------------------------------------------------------------------

  const handleGeneratePhantom = async () => {
    setPickedPosition(null);
    setPickingMode(false);
    setPhantomLoading(true);
    setPhantomError(null);
    setCtParamsError(null);
    setCtParamsResult(null);
    setCtParamsCopyState(null);
    setCtParamsDownloadState(null);
    setPhantom(null);
    setSliceIndex(0);
    setPlaying(false);
    try {
      const size = phantomSource === 'atlas' ? phantomSize : 128;
      const response = await simulationService.getPhantom(
        phantomSource,
        size,
        's0001',
        'head_to_feet',
      );
      setPhantom(response);
      setLoadedPhantomSize(size);
    } catch (err: any) {
      console.error('[SimulationPage] Phantom generation failed:', err);
      const detail =
        err?.response?.data?.detail || err.message || 'Failed to generate CT phantom';
      setPhantomError(detail);
    } finally {
      setPhantomLoading(false);
    }
  };

  const handlePlayPause = () => {
    if (!phantom) return;
    if (playing) {
      setPlaying(false);
    } else {
      // If at end, restart from top
      if (sliceIndex >= phantom.metadata.depth - 1) {
        setSliceIndex(0);
      }
      setPlaying(true);
    }
  };

  const handleSliceChange = (value: number) => {
    setSliceIndex(value);
  };

  const normalizeMAsInput = useCallback(() => {
    const trimmed = mAsInput.trim();
    const parsed = Number(trimmed);
    const normalized = !trimmed || Number.isNaN(parsed)
      ? DEFAULT_MAS
      : Math.max(MIN_MAS, Math.min(MAX_MAS, Math.round(parsed)));

    setCtParams((prev) => ({ ...prev, mAs: normalized }));
    setMAsInput(String(normalized));
    return normalized;
  }, [mAsInput]);

  const handleRunCtParamsSimulation = async () => {
    if (!phantom) {
      setCtParamsError('Please generate a CT phantom first.');
      return;
    }

    const currentSource = phantom.metadata.source ?? phantomSource;
    if (currentSource !== 'atlas' && currentSource !== 'procedural') {
      setCtParamsError(`Unsupported source for CT parameter simulation: ${String(currentSource)}`);
      return;
    }

    const normalizedMAs = normalizeMAsInput();
    const normalizedParams: CtParamsPreviewParams = {
      ...ctParams,
      mAs: normalizedMAs,
    };

    setCtParamsLoading(true);
    setCtParamsError(null);
    setCtParamsCopyState(null);
    setCtParamsDownloadState(null);
    setStandardizedCaseCopyState(null);
    setStandardizedCaseDownloadState(null);
    try {
      const response = await simulationService.runCtParamsPreview({
        source: currentSource,
        caseId: currentSource === 'atlas' ? (phantom.metadata.caseId || 's0001') : null,
        size: phantom.metadata.width,
        scanDirection: 'head_to_feet',
        params: normalizedParams,
      });
      setCtParamsResult(response);
    } catch (err: any) {
      console.error('[SimulationPage] CT parameter simulation failed:', err);
      setCtParamsError(err?.message || 'Failed to run CT parameter simulation.');
    } finally {
      setCtParamsLoading(false);
    }
  };

  const handleCopyCtParamsJson = async () => {
    if (!ctParamsResult) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(ctParamsResult.paramsJson, null, 2));
      setCtParamsCopyState('Copied');
    } catch {
      setCtParamsCopyState('Copy failed');
    }
  };

  const handleDownloadCtParamsJson = useCallback(() => {
    if (!ctParamsResult) return;
    try {
      const caseId = phantom?.metadata.caseId || 's0001';
      const filename = `ct_params_simulation_${caseId}_${formatTimestampForFilename(new Date())}.json`;
      const blob = new Blob([JSON.stringify(ctParamsResult.paramsJson, null, 2)], {
        type: 'application/json;charset=utf-8',
      });
      triggerDownload(blob, filename);
      setCtParamsDownloadState('Downloaded');
    } catch (err) {
      console.error('[SimulationPage] CT params JSON download failed:', err);
      setCtParamsDownloadState('Download failed');
    }
  }, [ctParamsResult, phantom?.metadata.caseId]);

  const handleCopyStandardizedCaseJson = async () => {
    if (!ctParamsResult?.standardizedCase) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(ctParamsResult.standardizedCase, null, 2));
      setStandardizedCaseCopyState('Copied');
    } catch {
      setStandardizedCaseCopyState('Copy failed');
    }
  };

  const handleDownloadStandardizedCaseJson = useCallback(() => {
    if (!ctParamsResult?.standardizedCase) return;
    try {
      const caseId = ctParamsResult.standardizedCase.caseId;
      const filename = `standardized_ct_case_${caseId}.json`;
      const blob = new Blob([JSON.stringify(ctParamsResult.standardizedCase, null, 2)], {
        type: 'application/json;charset=utf-8',
      });
      triggerDownload(blob, filename);
      setStandardizedCaseDownloadState('Downloaded');
    } catch (err) {
      console.error('[SimulationPage] Standardized case JSON download failed:', err);
      setStandardizedCaseDownloadState('Download failed');
    }
  }, [ctParamsResult]);

  // -----------------------------------------------------------------------
  // Handlers: lesion form
  // -----------------------------------------------------------------------

  /** Update a single form field */
  const updateForm = useCallback(
    <K extends keyof typeof form>(key: K, value: (typeof form)[K]) =>
      setForm((prev) => ({ ...prev, [key]: value })),
    [],
  );

  /** Handle click on CT phantom canvas to pick lesion position */
  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!pickingMode || !phantom) return;

      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;

      const x = Math.round((e.clientX - rect.left) * scaleX);
      const y = Math.round((e.clientY - rect.top) * scaleY);

      // Clamp to volume bounds
      const { width, height, depth } = phantom.metadata;
      const clampedX = Math.max(0, Math.min(x, width - 1));
      const clampedY = Math.max(0, Math.min(y, height - 1));
      const clampedZ = Math.max(0, Math.min(sliceIndex, depth - 1));

      // Store absolute phantom coords (for crosshair display)
      setPickedPosition({ x: clampedX, y: clampedY, z: clampedZ });

      // Store absolute phantom coords (for display in number inputs, works for same-volume synthetic)
      updateForm('centerX', clampedX);
      updateForm('centerY', clampedY);
      updateForm('centerZ', clampedZ);

      // Store normalized coords (0-1) for cross-volume scaling (e.g., phantom → DICOM)
      const normX = width > 0 ? clampedX / width : 0;
      const normY = height > 0 ? clampedY / height : 0;
      const normZ = depth > 0 ? clampedZ / depth : 0;
      updateForm('normalizedCenterX', normX);
      updateForm('normalizedCenterY', normY);
      updateForm('normalizedCenterZ', normZ);
      setPickingMode(false);
      setActiveTab('lesion');
    },
    [pickingMode, phantom, sliceIndex, updateForm],
  );

  /** Add a lesion from the current form values */
  const handleAddLesion = useCallback(() => {
    const radius = form.diameter / 2;

    // Compute center coords based on target volume
    let cx = form.centerX;
    let cy = form.centerY;
    let cz = form.centerZ;

    // When normalized coords are set (from phantom pick) and target is DICOM,
    // scale to DICOM volume dimensions.
    // Skip if user manually cleared center values (wanting auto-center: 0→backend).
    if (
      sourceType === 'dicom' &&
      form.normalizedCenterX > 0 &&
      selectedSeriesId &&
      form.centerX !== 0 && form.centerY !== 0 && form.centerZ !== 0
    ) {
      const series = seriesList.find((s) => s.id === selectedSeriesId);
      if (series) {
        cx = form.normalizedCenterX * series.columns;
        cy = form.normalizedCenterY * series.rows;
        cz = form.normalizedCenterZ * series.imageCount;
      }
    }

    const lesion: LesionConfig = {
      type: form.lesionType,
      shape: form.shape,
      center: [cx, cy, cz],
      radiusMm: [radius, radius, radius],
      huMean: form.huMean,
      huStd: form.huStd,
      marginSharpness: form.marginSharpness,
      calcificationFraction: 0,
      necrosisFraction: 0,
      spiculationDegree: form.spiculationDegree,
    };
    addLesion(lesion);
  }, [form, addLesion, sourceType, selectedSeriesId, seriesList]);

  /** Run simulation with all configured lesions */
  const handleRunSimulation = useCallback(async () => {
    if (lesions.length === 0 && organs.length === 0) {
      setJobError('Add at least one lesion or organ before running simulation');
      return;
    }

    if (sourceType === 'dicom' && !selectedSeriesId) {
      setJobError('Select a DICOM series to use as the base volume');
      return;
    }

    setJobLoading(true);
    setJobError(null);

    try {
      const jobConfig: Record<string, unknown> = {
        lesions,
        organs,
        outputFormat: 'dicom',
      };
      if (sourceType === 'dicom' && selectedStudyId && selectedSeriesId) {
        jobConfig.studyId = selectedStudyId;
        jobConfig.seriesId = selectedSeriesId;
      }
      const job = await simulationService.createJob(jobConfig);
      addJob(job);
    } catch (err: any) {
      setJobError(err?.message || 'Failed to create simulation job');
    } finally {
      setJobLoading(false);
    }
  }, [lesions, organs, addJob, sourceType, selectedStudyId, selectedSeriesId]);

  /** Preview current lesion configuration */
  const handlePreview = useCallback(async () => {
    const radius = form.diameter / 2;
    const lesion: LesionConfig = {
      type: form.lesionType,
      shape: form.shape,
      center: [form.centerX, form.centerY, form.centerZ],
      radiusMm: [radius, radius, radius],
      huMean: form.huMean,
      huStd: form.huStd,
      marginSharpness: form.marginSharpness,
      calcificationFraction: 0,
      necrosisFraction: 0,
      spiculationDegree: form.spiculationDegree,
    };

    setPreviewLoading(true);
    setPreviewResult(null);
    setPreviewError(null);

    try {
      const result = await simulationService.previewLesion(lesion);
      const pd = result.previewData;
      const meanHu = typeof pd.huMean === 'number' ? pd.huMean.toFixed(0) : '?';
      const volMm3 = typeof pd.volumeMm3 === 'number' ? pd.volumeMm3.toFixed(0) : '?';
      setPreviewResult(
        `HU range: [${result.huRange[0].toFixed(0)}, ${result.huRange[1].toFixed(0)}] · ` +
        `Mean: ${meanHu} · Volume: ${volMm3} mm³ · Voxels: ${result.voxelCount}`,
      );
    } catch (err: any) {
      setPreviewError(err?.message || 'Preview failed');
    } finally {
      setPreviewLoading(false);
    }
  }, [form]);

  /** Remove a lesion by index */
  const handleRemoveLesion = useCallback(
    (index: number) => removeLesion(index),
    [removeLesion],
  );

  /** Preview current lesion on the selected DICOM series */
  const handleDicomPreview = useCallback(async () => {
    if (!selectedSeriesId) {
      setDicomPreviewError('No DICOM series selected');
      return;
    }

    // Find the selected series to get DICOM volume dimensions
    const selectedSeries = seriesList.find((s) => s.id === selectedSeriesId);

    // Compute center: if normalized coords are available (from phantom pick),
    // scale them to the DICOM volume dimensions. Otherwise use centerX/Y/Z
    // as-is (manual entry, or 0 → auto-center in backend).
    let centerX = form.centerX;
    let centerY = form.centerY;
    let centerZ = form.centerZ;

    if (
      selectedSeries &&
      form.normalizedCenterX > 0 &&
      form.normalizedCenterY > 0 &&
      form.normalizedCenterZ > 0 &&
      form.centerX !== 0 && form.centerY !== 0 && form.centerZ !== 0
    ) {
      // Scale normalized phantom coords to DICOM volume space
      centerX = form.normalizedCenterX * selectedSeries.columns;
      centerY = form.normalizedCenterY * selectedSeries.rows;
      centerZ = form.normalizedCenterZ * selectedSeries.imageCount;
    }

    const radius = form.diameter / 2;
    const lesion: LesionConfig = {
      type: form.lesionType,
      shape: form.shape,
      center: [centerX, centerY, centerZ],
      radiusMm: [radius, radius, radius],
      huMean: form.huMean,
      huStd: form.huStd,
      marginSharpness: form.marginSharpness,
      calcificationFraction: 0,
      necrosisFraction: 0,
      spiculationDegree: form.spiculationDegree,
    };

    setDicomPreviewLoading(true);
    setDicomPreviewImage(null);
    setDicomPreviewStats(null);
    setDicomPreviewError(null);

    try {
      const res = await simulationService.previewLesionOnDicom(
        selectedSeriesId,
        lesion,
        40,
        400,
      );
      // Combine stats into a summary string
      const stats = `HU: ${res.huMean.toFixed(0)} ± ${res.huStd.toFixed(0)} · Range: [${res.huMin.toFixed(0)}, ${res.huMax.toFixed(0)}] · Volume: ${res.volumeMm3.toFixed(0)} mm³ · Slice ${res.sliceIndex + 1}/${res.totalSlices}`;
      setDicomPreviewImage(`data:image/png;base64,${res.imageBase64}`);
      setDicomPreviewStats(stats);
    } catch (err: any) {
      setDicomPreviewError(err?.message || 'DICOM preview failed');
    } finally {
      setDicomPreviewLoading(false);
    }
  }, [form, selectedSeriesId, seriesList]);

  /** Close the DICOM preview modal */
  const closeDicomPreview = useCallback(() => {
    setDicomPreviewImage(null);
    setDicomPreviewStats(null);
    setDicomPreviewError(null);
  }, []);

  // -----------------------------------------------------------------------
  // Handlers: export
  // -----------------------------------------------------------------------

  /** Export simulation results */
  const handleExport = useCallback(async () => {
    if (!latestCompletedJob) {
      setExportError('No completed simulation jobs available for export');
      return;
    }

    setExporting(true);
    setExportProgress(0);
    setExportError(null);

    try {
      const response = await simulationService.exportResults(
        latestCompletedJob.id,
        exportFormat,
        (progressEvent) => {
          if (progressEvent.total) {
            const percent = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total,
            );
            setExportProgress(percent);
          } else {
            setExportProgress(-1);
          }
        },
      );

      const fallbackNames: Record<string, string> = {
        dicom: `simulation_${latestCompletedJob.id}_dicom.zip`,
        nifti: `simulation_${latestCompletedJob.id}.nii.gz`,
        nrrd: `simulation_${latestCompletedJob.id}.nrrd`,
      };
      const filename = parseFilename(response.headers, fallbackNames[exportFormat]);
      triggerDownload(response.data, filename);
      setExportProgress(100);
    } catch (error: any) {
      const status = error.status;
      let message = 'Export failed';
      if (status === 409) message = 'Simulation job is not completed';
      else if (status === 404) message = 'Simulation job not found';
      else if (status === 400) message = error.message || 'Invalid export request';
      else if (status === 500) message = 'Server error during export';
      else if (error.message) message = error.message;
      setExportError(message);
    } finally {
      setExporting(false);
      setTimeout(() => setExportProgress(0), 1000);
    }
  }, [latestCompletedJob, exportFormat]);

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  const totalSlices = phantom ? phantom.metadata.depth : 0;
  const ctParamsWarnings = ctParamsResult?.metadata.warnings ?? ctParamsResult?.paramsJson.warnings ?? [];
  const huRangeBefore = ctParamsResult?.paramsJson.huRangeBefore;
  const huRangeAfter = ctParamsResult?.paramsJson.huRangeAfter ?? ctParamsResult?.metadata.huRange;
  const inputShape = ctParamsResult?.paramsJson.inputShape;
  const outputShape = ctParamsResult?.paramsJson.outputShape ?? ctParamsResult?.metadata.shape;
  const standardizedCase = ctParamsResult?.standardizedCase;

  return (
    <div className="flex h-full flex-col">
      {/* ---- Header ---- */}
      <div className="flex-shrink-0 border-b border-border px-6 py-4">
        <h1 className="text-2xl font-bold text-foreground">
          {activeTab === 'phantom'
            ? 'CT Simulation / Synthetic Upper-body CT'
            : 'Lesion Simulation'}
        </h1>
        <p className="text-sm text-muted-foreground">
          {activeTab === 'phantom'
            ? 'Synthetic CT phantom — geometric upper-body anatomy for demo & development'
            : 'Generate synthetic lesions and organs for AI training and validation'}
        </p>
      </div>

      {/* ---- Tab navigation ---- */}
      <div className="flex-shrink-0 flex gap-2 border-b border-border px-6">
        {(['lesion', 'organ', 'deformation', 'phantom'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`border-b-2 px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === tab
                ? 'border-primary text-primary'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            }`}
          >
            {tab === 'phantom'
              ? 'CT Phantom'
              : tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* ---- Tab Content ---- */}
      {activeTab !== 'phantom' ? (
        /* ================================================================ */
        /*  Lesion / Organ / Deformation                                     */
        /* ================================================================ */
        <div className="flex-1 overflow-y-auto p-6">
          {/* Source Volume Selection */}
          <div className="mb-4 rounded-lg border border-border bg-card p-4">
            <h3 className="mb-3 text-sm font-medium">Source Volume</h3>
            <div className="flex gap-6">
              <label className="flex cursor-pointer items-center gap-2 text-sm">
                <input
                  type="radio"
                  name="sourceType"
                  checked={sourceType === 'synthetic'}
                  onChange={() => setSourceType('synthetic')}
                  className="text-primary"
                />
                Synthetic (empty volume)
              </label>
              <label className="flex cursor-pointer items-center gap-2 text-sm">
                <input
                  type="radio"
                  name="sourceType"
                  checked={sourceType === 'dicom'}
                  onChange={() => setSourceType('dicom')}
                  className="text-primary"
                />
                Imported DICOM
              </label>
            </div>

            {sourceType === 'dicom' && (
              <div className="mt-3 grid gap-2 sm:grid-cols-2">
                {/* Study selector */}
                <div>
                  <label className="mb-1 block text-xs text-muted-foreground">Study</label>
                  <select
                    value={selectedStudyId ?? ''}
                    onChange={(e) => {
                      setSelectedStudyId(e.target.value || null);
                      setSelectedSeriesId(null);
                    }}
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                    disabled={loadingStudies}
                  >
                    <option value="">{loadingStudies ? 'Loading studies...' : '— Select a study —'}</option>
                    {studies.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.studyDescription || `Study ${s.studyDate}`} ({s.modalities?.join(', ') || '?'})
                      </option>
                    ))}
                  </select>
                </div>

                {/* Series selector */}
                <div>
                  <label className="mb-1 block text-xs text-muted-foreground">Series</label>
                  <select
                    value={selectedSeriesId ?? ''}
                    onChange={(e) => setSelectedSeriesId(e.target.value || null)}
                    className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                    disabled={!selectedStudyId || loadingSeries}
                  >
                    <option value="">
                      {!selectedStudyId
                        ? '— Select a study first —'
                        : loadingSeries
                          ? 'Loading series...'
                          : '— Select a series —'}
                    </option>
                    {seriesList.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.seriesDescription || `Series ${s.seriesNumber}`} ({s.modality}, {s.imageCount} slices)
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            )}
          </div>

          {/* Configuration panel */}
          <div className="mb-6 grid gap-6 lg:grid-cols-2">
            {/* Lesion parameters */}
            <div className="rounded-lg border border-border bg-card p-4">
              <h3 className="mb-4 text-sm font-medium">Lesion Parameters</h3>
              <div className="space-y-4">
                {/* Lesion Type */}
                <div>
                  <label className="text-xs text-muted-foreground">Lesion Type</label>
                  <select
                    value={form.lesionType}
                    onChange={(e) => updateForm('lesionType', e.target.value as LesionType)}
                    className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  >
                    {Object.entries(lesionTypeLabel).map(([value, label]) => (
                      <option key={value} value={value}>{label}</option>
                    ))}
                  </select>
                </div>

                {/* Shape */}
                <div>
                  <label className="text-xs text-muted-foreground">Shape</label>
                  <select
                    value={form.shape}
                    onChange={(e) => updateForm('shape', e.target.value as LesionShape)}
                    className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                  >
                    {Object.entries(shapeLabel).map(([value, label]) => (
                      <option key={value} value={value}>{label}</option>
                    ))}
                  </select>
                </div>

                {/* Mean HU */}
                <div>
                  <label className="text-xs text-muted-foreground">
                    Mean HU Value: <span className="text-primary font-mono">{form.huMean}</span>
                  </label>
                  <input
                    type="range"
                    min={-1000}
                    max={1000}
                    value={form.huMean}
                    onChange={(e) => updateForm('huMean', Number(e.target.value))}
                    className="mt-1 w-full"
                  />
                </div>

                {/* HU Std */}
                <div>
                  <label className="text-xs text-muted-foreground">
                    HU Heterogeneity (Std): <span className="text-primary font-mono">{form.huStd}</span>
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={200}
                    value={form.huStd}
                    onChange={(e) => updateForm('huStd', Number(e.target.value))}
                    className="mt-1 w-full"
                  />
                </div>

                {/* Diameter */}
                <div>
                  <label className="text-xs text-muted-foreground">
                    Diameter (mm): <span className="text-primary font-mono">{form.diameter}</span>
                  </label>
                  <input
                    type="range"
                    min={2}
                    max={100}
                    value={form.diameter}
                    onChange={(e) => updateForm('diameter', Number(e.target.value))}
                    className="mt-1 w-full"
                  />
                </div>

                {/* Margin Sharpness (for advanced shapes) */}
                <div>
                  <label className="text-xs text-muted-foreground">
                    Margin Sharpness: <span className="text-primary font-mono">{form.marginSharpness.toFixed(2)}</span>
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={Math.round(form.marginSharpness * 100)}
                    onChange={(e) => updateForm('marginSharpness', Number(e.target.value) / 100)}
                    className="mt-1 w-full"
                  />
                </div>

                {/* --- Position controls --- */}
                <div className="border-t border-border pt-3">
                  <h4 className="mb-2 text-xs font-medium text-muted-foreground">
                    Position (voxel coordinates)
                  </h4>
                  <div className="grid grid-cols-3 gap-2">
                    <div>
                      <label className="text-xs text-muted-foreground">X</label>
                      <input
                        type="number"
                        value={form.centerX}
                        onChange={(e) => updateForm('centerX', Number(e.target.value))}
                        className="mt-0.5 w-full rounded border border-border bg-background px-2 py-1 text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground">Y</label>
                      <input
                        type="number"
                        value={form.centerY}
                        onChange={(e) => updateForm('centerY', Number(e.target.value))}
                        className="mt-0.5 w-full rounded border border-border bg-background px-2 py-1 text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground">Z (Slice)</label>
                      <input
                        type="number"
                        value={form.centerZ}
                        onChange={(e) => updateForm('centerZ', Number(e.target.value))}
                        className="mt-0.5 w-full rounded border border-border bg-background px-2 py-1 text-sm"
                      />
                    </div>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setPickedPosition(null);
                      setPickingMode(true);
                      setActiveTab('phantom');
                    }}
                    className="mt-2 w-full"
                    disabled={!phantom}
                  >
                    Pick from Phantom
                  </Button>
                  {!phantom && (
                    <p className="mt-1 text-[10px] text-muted-foreground">
                      Generate a CT phantom first to use position picking
                    </p>
                  )}
                </div>

                {/* Add Lesion button */}
                <Button onClick={handleAddLesion} className="w-full">
                  Add Lesion
                </Button>
              </div>
            </div>

            {/* Lesion list */}
            <div className="rounded-lg border border-border bg-card p-4">
              <h3 className="mb-4 text-sm font-medium">Lesion List ({lesions.length})</h3>
              {lesions.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No lesions configured. Adjust parameters above and click <strong>Add Lesion</strong>.
                </p>
              ) : (
                <div className="space-y-2 max-h-[420px] overflow-y-auto">
                  {lesions.map((lesion, index) => (
                    <div
                      key={index}
                      className="group flex items-start justify-between rounded border border-border p-2 text-sm"
                    >
                      <div className="min-w-0 flex-1">
                        <div className="font-medium">
                          Lesion {index + 1}: {lesionTypeLabel[lesion.type]} ({shapeLabel[lesion.shape]})
                        </div>
                        <div className="mt-0.5 text-xs text-muted-foreground">
                          HU: {lesion.huMean} ± {lesion.huStd} · Ø {Math.round(lesion.radiusMm[0] * 2)} mm
                          {lesion.spiculationDegree > 0 && ` · Spic: ${lesion.spiculationDegree.toFixed(1)}`}
                        </div>
                      </div>
                      <button
                        onClick={() => handleRemoveLesion(index)}
                        className="ml-2 shrink-0 rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-destructive/10 hover:text-destructive group-hover:opacity-100"
                        title="Remove lesion"
                      >
                        <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Action buttons */}
          <div className="mt-auto flex flex-wrap items-center gap-2 border-t border-border pt-4">
            {/* Run Simulation */}
            <Button
              variant="default"
              onClick={handleRunSimulation}
              disabled={jobLoading || (lesions.length === 0 && organs.length === 0)}
            >
              {jobLoading ? 'Creating Job...' : 'Run Simulation'}
            </Button>

            {/* Preview */}
            <Button
              variant="outline"
              onClick={handlePreview}
              disabled={previewLoading}
            >
              {previewLoading ? 'Previewing...' : 'Preview'}
            </Button>

            {/* Preview on CT — only when DICOM source is selected */}
            {sourceType === 'dicom' && selectedSeriesId && (
              <Button
                variant="outline"
                onClick={handleDicomPreview}
                disabled={dicomPreviewLoading}
              >
                {dicomPreviewLoading ? 'Loading CT Preview...' : 'Preview on CT'}
              </Button>
            )}

            {/* Preview result */}
            {previewResult && (
              <span className="ml-2 text-xs text-green-600">{previewResult}</span>
            )}
            {previewError && (
              <span className="ml-2 text-xs text-destructive">{previewError}</span>
            )}
            {dicomPreviewError && (
              <span className="ml-2 text-xs text-destructive">{dicomPreviewError}</span>
            )}

            {/* Spacer */}
            <div className="flex-1" />

            {/* Export section */}
            <div className="flex items-center gap-2">
              <select
                value={exportFormat}
                onChange={(e) =>
                  setExportFormat(e.target.value as 'dicom' | 'nifti' | 'nrrd')
                }
                className="rounded-md border border-border bg-background px-3 py-2 text-sm"
                disabled={exporting}
              >
                <option value="dicom">DICOM</option>
                <option value="nifti">NIfTI</option>
                <option value="nrrd">NRRD</option>
              </select>

              <Button
                variant="ghost"
                onClick={handleExport}
                disabled={exporting || !latestCompletedJob}
              >
                {exporting
                  ? exportProgress === -1
                    ? 'Exporting...'
                    : exportProgress > 0 && exportProgress < 100
                      ? `Exporting ${exportProgress}%`
                      : 'Exporting...'
                  : 'Export Results'}
              </Button>
            </div>
          </div>

          {/* Export error */}
          {exportError && (
            <div className="mt-2 text-sm text-destructive">{exportError}</div>
          )}

          {/* Active / completed job summary */}
          {(activeJobs.length > 0 || completedJobs.length > 0) && (
            <div className="mt-4 border-t border-border pt-3 text-xs text-muted-foreground">
              {activeJobs.length > 0 && (
                <div>
                  Active jobs: {activeJobs.map((j) => `${j.id.slice(0, 8)} (${j.status})`).join(', ')}
                </div>
              )}
              {completedJobs.length > 0 && (
                <div>Completed jobs: {completedJobs.length}</div>
              )}
            </div>
          )}
        </div>
      ) : (
        /* ================================================================ */
        /*  CT Phantom View                                                  */
        /* ================================================================ */
        <div className="flex flex-1 flex-col overflow-hidden">
          {/* ---- Main area: Axial Viewer (left) + 3D Preview (right) ---- */}
          <div className="flex flex-1 overflow-hidden">
            {/* ---- Left: Axial Slice Viewer ---- */}
            <div className="flex w-1/2 flex-col border-r border-border bg-black">
              <div className="flex items-center justify-between border-b border-white/10 px-3 py-2">
                <div className="flex flex-col gap-0.5">
                  <span className="text-xs font-medium text-white/70">
                    Axial Slice Viewer
                  </span>
                  {phantom && (
                    <span className="text-[10px] text-cyan-400/70">
                      Scanning: Head/Chest → Abdomen/Pelvis
                    </span>
                  )}
                </div>
                {phantom && (
                  <span className="text-xs tabular-nums text-white/50">
                    Slice {sliceIndex + 1} / {totalSlices}
                    {pickedPosition && (
                      <span className="ml-2 text-red-400/80">
                        · Pick: ({pickedPosition.x}, {pickedPosition.y}, z={pickedPosition.z})
                      </span>
                    )}
                    {pickingMode && (
                      <span className="ml-2 text-amber-400 animate-pulse">
                        · Click canvas to set position
                      </span>
                    )}
                  </span>
                )}
              </div>
              <div className="flex-1 flex items-center justify-center p-2">
                {!phantom ? (
                  <div className="flex flex-col items-center gap-3 text-center">
                    <svg
                      className="h-12 w-12 text-white/15"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth={1}
                    >
                      <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
                      <path d="M8 21h8" />
                      <path d="M12 17v4" />
                    </svg>
                    <p className="text-sm text-white/40">
                      {phantomSource === 'atlas'
                        ? 'Click "Generate Atlas CT" to load the real CT phantom'
                        : 'Click "Generate Synthetic CT" to load the phantom'}
                    </p>
                    {phantomLoading && (
                      <div className="flex items-center gap-2 text-xs text-white/50">
                        <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                        Generating phantom...
                      </div>
                    )}
                    {phantomError && (
                      <p className="text-xs text-red-400">{phantomError}</p>
                    )}
                  </div>
                ) : (
                  <canvas
                    ref={canvasRef}
                    onClick={handleCanvasClick}
                    className={`max-h-full max-w-full border border-white/10 object-contain ${
                      pickingMode ? 'cursor-crosshair' : 'cursor-default'
                    }`}
                    style={{ imageRendering: 'pixelated' }}
                  />
                )}
              </div>
            </div>

            {/* ---- Right: 3D Volume Preview ---- */}
            <div className="flex w-1/2 flex-col bg-black">
              <div className="flex items-center justify-between border-b border-white/10 px-3 py-2">
                <div className="flex flex-col gap-0.5">
                  <span className="text-xs font-medium text-white/70">
                    3D Volume Preview
                  </span>
                  {phantom && (
                    <span className="text-[10px] text-cyan-400/70">
                      Progressive Scan · Head → Feet
                    </span>
                  )}
                </div>
              </div>
              <div className="flex-1">
                {vtkVolumeData && vtkDims ? (
                  <VolumeRenderer
                    mode="synthetic"
                    showControls
                    scanView
                    scanDirection="head_to_feet"
                    opacityPreset={
                      activePreset.label === 'Soft'
                        ? 'ct-soft-tissue'
                        : activePreset.label === 'Lung'
                          ? 'ct-lung'
                          : 'ct-bone'
                    }
                    syntheticData={vtkVolumeData}
                    syntheticDims={vtkDims}
                    syntheticSpacing={vtkSpacing ?? undefined}
                    syntheticClipIndex={sliceIndex}
                    syntheticScanAxis="z"
                  />
                ) : (
                  <div className="flex h-full items-center justify-center">
                    <p className="text-sm text-white/30">
                      Generate a phantom to see 3D preview
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="grid flex-shrink-0 grid-cols-1 gap-3 border-t border-border bg-zinc-950 px-4 py-3 lg:grid-cols-2">
            <div className="flex min-h-[280px] flex-col rounded border border-white/10 bg-black">
              <div className="flex items-center justify-between border-b border-white/10 px-3 py-2">
                <span className="text-xs font-medium text-white/70">Original CT Slice</span>
                {phantom && (
                  <span className="text-[10px] text-white/50">
                    Slice {sliceIndex + 1} / {totalSlices}
                  </span>
                )}
              </div>
              <div className="flex flex-1 items-center justify-center p-2">
                {phantom ? (
                  <canvas
                    ref={originalCompareCanvasRef}
                    className="max-h-full max-w-full border border-white/10 object-contain"
                    style={{ imageRendering: 'pixelated' }}
                  />
                ) : (
                  <p className="text-sm text-white/35">Generate a CT phantom to preview original slices.</p>
                )}
              </div>
            </div>

            <div className="flex min-h-[280px] flex-col rounded border border-white/10 bg-black">
              <div className="flex items-center justify-between border-b border-white/10 px-3 py-2">
                <span className="text-xs font-medium text-white/70">Simulated CT Slice</span>
                {ctParamsLoading && (
                  <span className="text-[10px] text-cyan-400/80">Running CT parameter simulation...</span>
                )}
              </div>
              <div className="flex flex-1 items-center justify-center p-2">
                {ctParamsResult && simulatedVolumeData ? (
                  <canvas
                    ref={simulatedCompareCanvasRef}
                    className="max-h-full max-w-full border border-white/10 object-contain"
                    style={{ imageRendering: 'pixelated' }}
                  />
                ) : (
                  <div className="px-6 text-center text-sm text-white/35">
                    {ctParamsError && !ctParamsResult
                      ? ctParamsError
                      : 'Run CT Parameter Simulation to preview simulated CT.'}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* ---- Bottom: Controls bar ---- */}
          <div className="flex-shrink-0 border-t border-border bg-card px-4 py-3">
            <div className="flex items-center gap-4 flex-wrap">
              {/* Source selector */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Source:</span>
                <select
                  value={phantomSource}
                  onChange={(e) => {
                    const src = e.target.value as 'procedural' | 'atlas';
                    setPhantomSource(src);
                    setPhantom(null);
                    setPhantomError(null);
                    setCtParamsResult(null);
                    setCtParamsError(null);
                    setCtParamsCopyState(null);
                    setCtParamsDownloadState(null);
                    // Reset size to sensible default for each mode
                    setPhantomSize(src === 'atlas' ? 192 : 128);
                  }}
                  disabled={phantomLoading}
                  className="rounded border border-border bg-background px-2 py-1 text-xs"
                >
                  <option value="atlas">Real CT Atlas</option>
                  <option value="procedural">Procedural Phantom</option>
                </select>
              </div>

              {/* Size selector */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Size:</span>
                <select
                  value={phantomSize}
                  onChange={(e) => {
                    setPhantomSize(Number(e.target.value));
                    setCtParamsResult(null);
                    setCtParamsError(null);
                    setCtParamsCopyState(null);
                    setCtParamsDownloadState(null);
                  }}
                  disabled={phantomLoading}
                  className="rounded border border-border bg-background px-2 py-1 text-xs"
                >
                  <option value={96}>96 Fast</option>
                  <option value={160}>160 Balanced</option>
                  <option value={192}>192 Detail</option>
                  <option value={256}>256 Max</option>
                </select>
                <span className="text-[10px] text-muted-foreground/60">
                  larger = clearer, slower
                </span>
              </div>

              {/* Generate button */}
              <Button
                variant="default"
                size="sm"
                onClick={handleGeneratePhantom}
                disabled={phantomLoading}
              >
                {phantomLoading
                  ? 'Generating...'
                  : phantomSource === 'atlas'
                    ? 'Generate Atlas CT'
                    : 'Generate Synthetic CT'}
              </Button>

              {/* Play / Pause */}
              <Button
                variant="outline"
                size="sm"
                onClick={handlePlayPause}
                disabled={!phantom}
              >
                {playing ? '⏸ Pause' : '▶ Play'}
              </Button>

              {/* Slice slider */}
              <div className="flex items-center gap-2 min-w-[200px]">
                <span className="text-xs text-muted-foreground w-16 tabular-nums">
                  {phantom
                    ? `${sliceIndex + 1} / ${totalSlices}`
                    : '— / —'}
                </span>
                <input
                  type="range"
                  min={0}
                  max={Math.max(totalSlices - 1, 0)}
                  step={1}
                  value={sliceIndex}
                  onChange={(e) => handleSliceChange(Number(e.target.value))}
                  disabled={!phantom}
                  className="h-1 flex-1 cursor-pointer accent-primary"
                />
              </div>

              {/* Speed control */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Speed:</span>
                <select
                  value={playSpeed}
                  onChange={(e) => setPlaySpeed(Number(e.target.value))}
                  disabled={!phantom}
                  className="rounded border border-border bg-background px-2 py-1 text-xs"
                >
                  <option value={5}>5 fps</option>
                  <option value={10}>10 fps</option>
                  <option value={20}>20 fps</option>
                  <option value={30}>30 fps</option>
                </select>
              </div>

              <div className="h-5 w-px bg-border" />

              {/* Window preset */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Window:</span>
                {WINDOW_PRESETS.map((preset) => (
                  <button
                    key={preset.label}
                    onClick={() => setActivePreset(preset)}
                    className={`rounded px-2 py-1 text-xs transition-colors ${
                      activePreset.label === preset.label
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted text-muted-foreground hover:bg-muted-foreground/20'
                    }`}
                    disabled={!phantom}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>

              {/* Preset info */}
              {phantom && (
                <span className="text-xs text-muted-foreground">
                  WL={activePreset.windowLevel} WW={activePreset.windowWidth}
                </span>
              )}

              {/* Pick Position toggle */}
              <div className="h-5 w-px bg-border" />
              <Button
                variant={pickingMode ? 'default' : 'outline'}
                size="sm"
                onClick={() => setPickingMode((prev) => !prev)}
                disabled={!phantom}
              >
                {pickingMode ? 'Cancel Pick' : 'Pick Position'}
              </Button>

              {/* Label overlay toggle (only when labels available) */}
              {phantom?.labelBase64 && (
                <>
                  <div className="h-5 w-px bg-border" />
                  <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={showLabelOverlay}
                      onChange={(e) => setShowLabelOverlay(e.target.checked)}
                      className="accent-primary"
                    />
                    Organ Labels
                  </label>
                </>
              )}

              {/* Source badge */}
              {phantom && phantom.metadata.source && (
                <>
                  <div className="h-5 w-px bg-border" />
                  <span className="rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
                    {phantom.metadata.source === 'atlas'
                      ? `atlas · ${phantom.metadata.caseId || '—'}`
                      : 'procedural'}
                  </span>
                </>
              )}

              {/* Scan direction indicator */}
              {phantom && phantom.metadata.scanDirection && (
                <>
                  <div className="h-5 w-px bg-border" />
                  <span className="flex items-center gap-1 text-xs text-muted-foreground">
                    <span className="text-[10px]">
                      {phantom.metadata.scanDirection === 'head_to_feet'
                        ? '🔽 Head → Feet'
                        : '🔼 Feet → Head'}
                    </span>
                    {phantom.metadata.flippedZ && (
                      <span
                        className="rounded bg-amber-500/20 px-1 text-[9px] text-amber-400"
                        title="Volume was flipped along z to match requested scan direction"
                      >
                        flipped
                      </span>
                    )}
                  </span>
                </>
              )}

              {/* Shape info */}
              {phantom && phantom.metadata.outputShape && (
                <>
                  <div className="h-5 w-px bg-border" />
                  <span className="text-[10px] text-muted-foreground/60">
                    shape{' '}
                    {phantom.metadata.outputShape.join('×')}
                  </span>
                </>
              )}
            </div>

            <div className="mt-3 grid gap-3 border-t border-border/60 pt-3 xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
              <div className="rounded border border-border bg-background/60 p-3">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-medium">CT Scan Params Simulation</h3>
                    <p className="text-xs text-muted-foreground">
                      CT parameter simulation supports Real CT Atlas and Procedural Phantom.
                    </p>
                  </div>
                  <Button
                    variant="default"
                    size="sm"
                    onClick={handleRunCtParamsSimulation}
                    disabled={ctParamsLoading}
                  >
                    {ctParamsLoading ? 'Running...' : 'Run CT Parameter Simulation'}
                  </Button>
                </div>

                <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                  <label className="flex flex-col gap-1 text-xs text-muted-foreground">
                    Slice Thickness (mm)
                    <select
                      value={ctParams.sliceThicknessMm}
                      onChange={(e) =>
                        setCtParams((prev) => ({
                          ...prev,
                          sliceThicknessMm: Number(e.target.value) as CtParamsPreviewParams['sliceThicknessMm'],
                        }))
                      }
                      className="rounded border border-border bg-background px-2 py-1.5 text-sm text-foreground"
                    >
                      {[0.625, 1.0, 2.5, 5.0, 10.0].map((value) => (
                        <option key={value} value={value}>
                          {value}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="flex flex-col gap-1 text-xs text-muted-foreground">
                    Dose Level
                    <select
                      value={ctParams.doseLevel}
                      onChange={(e) =>
                        setCtParams((prev) => ({
                          ...prev,
                          doseLevel: e.target.value as CtParamsPreviewParams['doseLevel'],
                        }))
                      }
                      className="rounded border border-border bg-background px-2 py-1.5 text-sm text-foreground"
                    >
                      {['low', 'standard', 'high'].map((value) => (
                        <option key={value} value={value}>
                          {value}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="flex flex-col gap-1 text-xs text-muted-foreground">
                    mAs
                    <input
                      type="number"
                      min={MIN_MAS}
                      max={MAX_MAS}
                      step={1}
                      value={mAsInput}
                      onChange={(e) => setMAsInput(e.target.value)}
                      onBlur={normalizeMAsInput}
                      className="rounded border border-border bg-background px-2 py-1.5 text-sm text-foreground"
                    />
                    <span className="text-[11px] text-muted-foreground/80">mAs range: 30-300</span>
                  </label>

                  <label className="flex flex-col gap-1 text-xs text-muted-foreground">
                    kVp
                    <select
                      value={ctParams.kVp}
                      onChange={(e) =>
                        setCtParams((prev) => ({
                          ...prev,
                          kVp: Number(e.target.value) as CtParamsPreviewParams['kVp'],
                        }))
                      }
                      className="rounded border border-border bg-background px-2 py-1.5 text-sm text-foreground"
                    >
                      {[80, 100, 120, 140].map((value) => (
                        <option key={value} value={value}>
                          {value}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="flex flex-col gap-1 text-xs text-muted-foreground">
                    Pitch
                    <select
                      value={ctParams.pitch}
                      onChange={(e) =>
                        setCtParams((prev) => ({
                          ...prev,
                          pitch: Number(e.target.value) as CtParamsPreviewParams['pitch'],
                        }))
                      }
                      className="rounded border border-border bg-background px-2 py-1.5 text-sm text-foreground"
                    >
                      {[0.5, 0.8, 1.0, 1.2, 1.5].map((value) => (
                        <option key={value} value={value}>
                          {value}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="flex flex-col gap-1 text-xs text-muted-foreground">
                    FOV (mm)
                    <select
                      value={ctParams.fovMm}
                      onChange={(e) =>
                        setCtParams((prev) => ({
                          ...prev,
                          fovMm: Number(e.target.value) as CtParamsPreviewParams['fovMm'],
                        }))
                      }
                      className="rounded border border-border bg-background px-2 py-1.5 text-sm text-foreground"
                    >
                      {[150, 250, 350, 500].map((value) => (
                        <option key={value} value={value}>
                          {value}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="flex flex-col gap-1 text-xs text-muted-foreground">
                    Matrix Size
                    <select
                      value={ctParams.matrixSize}
                      onChange={(e) =>
                        setCtParams((prev) => ({
                          ...prev,
                          matrixSize: Number(e.target.value) as CtParamsPreviewParams['matrixSize'],
                        }))
                      }
                      className="rounded border border-border bg-background px-2 py-1.5 text-sm text-foreground"
                    >
                      {[256, 512, 1024].map((value) => (
                        <option key={value} value={value}>
                          {value}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="flex flex-col gap-1 text-xs text-muted-foreground">
                    Kernel
                    <select
                      value={ctParams.kernel}
                      onChange={(e) =>
                        setCtParams((prev) => ({
                          ...prev,
                          kernel: e.target.value as CtParamsPreviewParams['kernel'],
                        }))
                      }
                      className="rounded border border-border bg-background px-2 py-1.5 text-sm text-foreground"
                    >
                      {['smooth', 'soft', 'standard', 'lung', 'bone', 'sharp'].map((value) => (
                        <option key={value} value={value}>
                          {value}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="flex flex-col gap-1 text-xs text-muted-foreground">
                    Contrast Phase
                    <select
                      value={ctParams.contrastPhase}
                      onChange={(e) =>
                        setCtParams((prev) => ({
                          ...prev,
                          contrastPhase: e.target.value as CtParamsPreviewParams['contrastPhase'],
                        }))
                      }
                      className="rounded border border-border bg-background px-2 py-1.5 text-sm text-foreground"
                    >
                      {['noncontrast', 'arterial', 'venous', 'delayed'].map((value) => (
                        <option key={value} value={value}>
                          {value}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>

                {ctParamsError && (
                  <p className="mt-3 text-xs text-red-500">{ctParamsError}</p>
                )}
              </div>

              <div className="rounded border border-border bg-background/60 p-3">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-medium">Simulation Result</h3>
                    <p className="text-xs text-muted-foreground">
                      Preview metadata and the exact parameter payload returned by the backend.
                    </p>
                  </div>
                  {ctParamsResult && (
                    <div className="flex items-center gap-2">
                      <Button variant="outline" size="sm" onClick={handleCopyCtParamsJson}>
                        {ctParamsCopyState || 'Copy Params JSON'}
                      </Button>
                      <Button variant="outline" size="sm" onClick={handleDownloadCtParamsJson}>
                        {ctParamsDownloadState || 'Download Params JSON'}
                      </Button>
                    </div>
                  )}
                </div>

                {ctParamsResult ? (
                  <div className="space-y-3 text-sm">
                    <div className="grid gap-2 sm:grid-cols-2">
                      <div className="rounded border border-border/70 bg-card px-3 py-2">
                        <div className="text-[11px] uppercase tracking-wide text-muted-foreground">HU Range</div>
                        <div className="mt-1 text-xs text-foreground">
                          Before: {huRangeBefore ? `${huRangeBefore[0].toFixed(1)} to ${huRangeBefore[1].toFixed(1)}` : '—'}
                        </div>
                        <div className="text-xs text-foreground">
                          After: {huRangeAfter ? `${huRangeAfter[0].toFixed(1)} to ${huRangeAfter[1].toFixed(1)}` : '—'}
                        </div>
                      </div>

                      <div className="rounded border border-border/70 bg-card px-3 py-2">
                        <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Shape</div>
                        <div className="mt-1 text-xs text-foreground">
                          Input: {inputShape ? inputShape.join(' × ') : '—'}
                        </div>
                        <div className="text-xs text-foreground">
                          Output: {outputShape ? outputShape.join(' × ') : '—'}
                        </div>
                      </div>
                    </div>

                    <div className="grid gap-2 sm:grid-cols-2">
                      <div className="rounded border border-border/70 bg-card px-3 py-2">
                        <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
                          Effective Slice Thickness
                        </div>
                        <div className="mt-1 text-xs text-foreground">
                          {ctParamsResult.metadata.effectiveSliceThicknessMm.toFixed(3)} mm
                        </div>
                      </div>

                      <div className="rounded border border-border/70 bg-card px-3 py-2">
                        <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Warnings</div>
                        <div className="mt-1 text-xs text-foreground">
                          {ctParamsWarnings.length > 0 ? ctParamsWarnings.join(' | ') : 'None'}
                        </div>
                      </div>
                    </div>

                    {standardizedCase && (
                      <div className="rounded border border-border/70 bg-card px-3 py-2">
                        <div className="mb-2 flex items-center justify-between gap-3">
                          <div>
                            <div className="text-xs font-medium text-foreground">
                              Standardized Output for Downstream Modules
                            </div>
                            <div className="text-[11px] text-muted-foreground">
                              Lightweight standardized case metadata for downstream artifact or lesion modules.
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <Button variant="outline" size="sm" onClick={handleCopyStandardizedCaseJson}>
                              {standardizedCaseCopyState || 'Copy Standardized Case JSON'}
                            </Button>
                            <Button variant="outline" size="sm" onClick={handleDownloadStandardizedCaseJson}>
                              {standardizedCaseDownloadState || 'Download Standardized Case JSON'}
                            </Button>
                          </div>
                        </div>

                        <div className="grid gap-2 text-xs text-foreground sm:grid-cols-2 xl:grid-cols-3">
                          <div><span className="text-muted-foreground">case_id:</span> {standardizedCase.caseId}</div>
                          <div><span className="text-muted-foreground">source:</span> {standardizedCase.source}</div>
                          <div><span className="text-muted-foreground">shape:</span> {standardizedCase.volume.shape.join(' x ')}</div>
                          <div><span className="text-muted-foreground">spacing:</span> {standardizedCase.volume.spacing.join(', ')}</div>
                          <div><span className="text-muted-foreground">hu_range:</span> {standardizedCase.volume.huRange[0].toFixed(1)} to {standardizedCase.volume.huRange[1].toFixed(1)}</div>
                          <div><span className="text-muted-foreground">slice_count:</span> {standardizedCase.volume.sliceCount}</div>
                          <div><span className="text-muted-foreground">dtype:</span> {standardizedCase.volume.dtype}</div>
                          <div><span className="text-muted-foreground">axis_order:</span> {standardizedCase.volume.axisOrder}</div>
                          <div><span className="text-muted-foreground">image_data_field:</span> {standardizedCase.volume.imageDataField}</div>
                        </div>
                      </div>
                    )}

                    <details className="rounded border border-border/70 bg-card px-3 py-2">
                      <summary className="cursor-pointer text-xs font-medium text-foreground">
                        Params JSON
                      </summary>
                      <pre className="mt-2 overflow-auto whitespace-pre-wrap break-all text-[11px] text-muted-foreground">
                        {JSON.stringify(ctParamsResult.paramsJson, null, 2)}
                      </pre>
                    </details>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    Run CT Parameter Simulation to populate preview metadata and params JSON.
                  </p>
                )}
              </div>
            </div>

            {/* ---- Label legend (compact, shown when labels available) ---- */}
            {phantom?.labelBase64 && showLabelOverlay && (
              <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 border-t border-border/50 pt-2">
                <span className="text-[10px] text-muted-foreground/60">
                  Labels:
                </span>
                {[
                  [9, 'Liver'],
                  [7, 'L Kidney'],
                  [8, 'R Kidney'],
                  [10, 'Lungs'],
                  [17, 'Spleen'],
                  [15, 'Pancreas'],
                  [19, 'Trachea'],
                  [20, 'Bladder'],
                ].map(([id, name]) => {
                  const color = ORGAN_COLORS[id as number];
                  return (
                    <span
                      key={id}
                      className="flex items-center gap-1 text-[10px] text-muted-foreground"
                    >
                      <span
                        className="inline-block h-2.5 w-2.5 rounded-sm border border-white/20"
                        style={{
                          backgroundColor: color
                            ? `rgb(${color[0]},${color[1]},${color[2]})`
                            : '#888',
                        }}
                      />
                      {name}
                    </span>
                  );
                })}
              </div>
            )}

            {/* ---- Organ slice ranges (atlas debug info) ---- */}
            {phantom?.metadata?.sliceLabelPresence && (
              <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-0.5 border-t border-border/30 pt-1.5">
                <span className="text-[10px] text-muted-foreground/50">
                  Organ z-ranges:
                </span>
                {(
                  [
                    ['lung', 'Lungs'],
                    ['liver', 'Liver'],
                    ['spleen', 'Spleen'],
                    ['kidney_left', 'L Kidney'],
                    ['kidney_right', 'R Kidney'],
                    ['bladder', 'Bladder'],
                  ] as [string, string][]
                ).map(([key, label]) => {
                  const range = phantom.metadata.sliceLabelPresence![key];
                  if (!range) return null;
                  const [zMin, zMax] = range;
                  const total = phantom.metadata.depth;
                  const pct0 = total > 0 ? Math.round((zMin / total) * 100) : 0;
                  const pct1 = total > 0 ? Math.round((zMax / total) * 100) : 0;
                  return (
                    <span
                      key={key}
                      className="text-[10px] text-muted-foreground/70"
                    >
                      {label}: {zMin + 1}–{zMax + 1}{' '}
                      <span className="text-muted-foreground/40">
                        ({pct0}–{pct1}%)
                      </span>
                    </span>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ---- DICOM Preview Modal ---- */}
      {dicomPreviewImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
          onClick={closeDicomPreview}
        >
          <div
            className="relative max-h-[90vh] max-w-[90vw] rounded-lg bg-card p-4 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={closeDicomPreview}
              className="absolute -right-3 -top-3 flex h-7 w-7 items-center justify-center rounded-full bg-destructive text-sm text-destructive-foreground hover:bg-destructive/90"
              title="Close"
            >
              ✕
            </button>
            <h3 className="mb-3 text-sm font-medium">CT Preview — Lesion on DICOM</h3>
            <p className="mb-2 text-xs text-muted-foreground">
              Left: Original · Right: With lesion (center slice, WW/WC: 400/40)
            </p>
            <img
              src={dicomPreviewImage}
              alt="DICOM lesion preview"
              className="max-h-[65vh] w-auto rounded border border-border"
            />
            {dicomPreviewStats && (
              <p className="mt-2 text-xs text-green-600">{dicomPreviewStats}</p>
            )}
          </div>
        </div>
      )}

      {/* ---- Error messages ---- */}
      {jobError && (
        <div className="mt-2 text-sm text-destructive">{jobError}</div>
      )}
      {exportError && (
        <div className="mt-2 text-sm text-destructive">{exportError}</div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/**
 * Parse filename from Content-Disposition header
 */
function parseFilename(headers: any, fallback: string): string {
  const disposition = headers['content-disposition'];
  if (disposition) {
    const match = disposition.match(/filename\*?=['"]?([^'";]+)['"]?/);
    if (match && match[1]) {
      return decodeURIComponent(match[1]);
    }
  }
  return fallback;
}

/**
 * Trigger browser download
 */
function triggerDownload(blob: Blob, filename: string) {
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}

function formatTimestampForFilename(date: Date): string {
  const pad = (value: number) => String(value).padStart(2, '0');
  return [
    date.getFullYear(),
    pad(date.getMonth() + 1),
    pad(date.getDate()),
  ].join('') + '_' + [
    pad(date.getHours()),
    pad(date.getMinutes()),
    pad(date.getSeconds()),
  ].join('');
}
