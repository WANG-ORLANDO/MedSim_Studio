/**
 * DICOM Volume Loader for vtk.js
 *
 * Fetches all slices of a DICOM series from the backend, parses them with
 * dicom-parser, extracts pixel data (applying RescaleSlope / RescaleIntercept
 * to convert to Hounsfield Units), and assembles a single Float32Array volume
 * suitable for consumption by vtkImageData.
 *
 * Currently only supports uncompressed transfer syntaxes:
 *   - 1.2.840.10008.1.2     (Implicit VR Little Endian)
 *   - 1.2.840.10008.1.2.1   (Explicit VR Little Endian)
 *   - 1.2.840.10008.1.2.2   (Explicit VR Big Endian)
 *
 * Compressed syntaxes (JPEG, JPEG-2000, RLE, etc.) return a clear error
 * rather than silently producing garbage.
 */

import * as dicomParser from 'dicom-parser';
import { dicomService } from '@services/index';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DicomVolumeData {
  /** HU-normalised scalar array (Float32) — shape [x, y, z] flat */
  scalarData: Float32Array;
  /** [columns, rows, sliceCount] */
  dimensions: [number, number, number];
  /** Voxel size in mm — [x, y, z] */
  spacing: [number, number, number];
  /** Volume origin in patient coordinates — [x, y, z] */
  origin: [number, number, number];
  /** [min, max] HU range observed in the data */
  scalarRange: [number, number];
}

/** Per-slice metadata extracted from a single DICOM file. */
interface DicomSlice {
  instanceNumber: number;
  pixelData: Int16Array | Uint16Array;
  rows: number;
  columns: number;
  slope: number;
  intercept: number;
  sliceLocation: number | null;
  imagePositionZ: number | null;
  pixelSpacing: [number, number] | null;
  sliceThickness: number | null;
  transferSyntaxUID: string;
}

// ---------------------------------------------------------------------------
// Supported transfer syntaxes (uncompressed only)
// ---------------------------------------------------------------------------

const UNCOMPRESSED_TS = new Set([
  '1.2.840.10008.1.2', // Implicit VR Little Endian
  '1.2.840.10008.1.2.1', // Explicit VR Little Endian
  '1.2.840.10008.1.2.2', // Explicit VR Big Endian
]);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Fetch a single DICOM file as an ArrayBuffer from the backend. */
async function fetchDicomArrayBuffer(instanceId: string): Promise<ArrayBuffer> {
  const url = `/api/v1/dicom/instances/${instanceId}/file`;
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(
      `Failed to fetch instance ${instanceId}: HTTP ${resp.status}`,
    );
  }
  return resp.arrayBuffer();
}

/**
 * Parse a single DICOM slice from raw bytes.
 * Only uncompressed transfer syntaxes are accepted.
 */
function parseDicomSlice(
  arrayBuffer: ArrayBuffer,
  instanceId: string,
): DicomSlice {
  const byteArray = new Uint8Array(arrayBuffer);
  const dataSet = dicomParser.parseDicom(byteArray);

  // ---- transfer syntax check ----
  const tsUid: string =
    dataSet.string('x00020010') ?? '1.2.840.10008.1.2';
  if (!UNCOMPRESSED_TS.has(tsUid)) {
    throw new Error(
      `Compressed DICOM transfer syntax ${tsUid} is not supported by ` +
        `vtk.js demo loader yet. Instance: ${instanceId}`,
    );
  }

  // ---- image dimensions ----
  const rows = dataSet.uint16('x00280010');
  const columns = dataSet.uint16('x00280011');
  if (!rows || !columns) {
    throw new Error(
      `Instance ${instanceId} is missing rows/columns (0028,0010/0011).`,
    );
  }

  // ---- rescale ----
  const slope = dataSet.floatString('x00281053') ?? 1.0;
  const intercept = dataSet.floatString('x00281052') ?? 0.0;

  // ---- pixel representation ----
  const pixelRepresentation = dataSet.uint16('x00280103') ?? 0; // 0=unsigned, 1=signed
  const bitsAllocated = dataSet.uint16('x00280100') ?? 16;
  const bitsStored = dataSet.uint16('x00280101') ?? bitsAllocated;

  // ---- pixel data ----
  const pixelElement = dataSet.elements['x7fe00010'];
  if (!pixelElement || pixelElement.length === 0) {
    throw new Error(`Instance ${instanceId} has no pixel data.`);
  }

  // Determine signedness and construct the correct typed view.
  let rawPixels: Int16Array | Uint16Array;
  // Most CT uses 16-bit; we take a simple path for other bit-depths.
  if (bitsAllocated === 16) {
    if (pixelRepresentation === 1) {
      rawPixels = new Int16Array(
        dataSet.byteArray.buffer,
        pixelElement.dataOffset,
        pixelElement.length / 2,
      );
    } else {
      rawPixels = new Uint16Array(
        dataSet.byteArray.buffer,
        pixelElement.dataOffset,
        pixelElement.length / 2,
      );
    }
  } else if (bitsAllocated === 8) {
    // 8-bit is always unsigned
    const u8 = new Uint8Array(
      dataSet.byteArray.buffer,
      pixelElement.dataOffset,
      pixelElement.length,
    );
    rawPixels = new Uint16Array(u8); // upcast for consistency
  } else {
    throw new Error(
      `Instance ${instanceId} bitsAllocated=${bitsAllocated} is not supported ` +
        `(only 8 or 16).`,
    );
  }

  // ---- position / spacing ----
  const sliceLocation = dataSet.floatString('x00201041') ?? null;
  const imagePositionPatient = dataSet.string('x00200032'); // multi-valued string
  let imagePositionZ: number | null = null;
  if (imagePositionPatient) {
    const parts = imagePositionPatient.split('\\').map(Number);
    imagePositionZ = parts[2] ?? null;
  }

  const pixelSpacingStr = dataSet.string('x00280030');
  let pixelSpacing: [number, number] | null = null;
  if (pixelSpacingStr) {
    const parts = pixelSpacingStr.split('\\').map(Number);
    if (parts.length >= 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
      pixelSpacing = [parts[0], parts[1]];
    }
  }

  const sliceThickness = dataSet.floatString('x00180050') ?? null;
  const instanceNumber = dataSet.intString('x00200013') ?? 0;

  return {
    instanceNumber,
    pixelData: rawPixels,
    rows,
    columns,
    slope,
    intercept,
    sliceLocation,
    imagePositionZ,
    pixelSpacing,
    sliceThickness,
    transferSyntaxUID: tsUid,
  };
}

/** Sort slices by Z position, falling back to instance number. */
function sortSlices(slices: DicomSlice[]): DicomSlice[] {
  // Prefer imagePositionZ for ordering — it is the most reliable spatial key.
  const hasZ = slices.every((s) => s.imagePositionZ !== null);
  if (hasZ) {
    return [...slices].sort(
      (a, b) => (a.imagePositionZ as number) - (b.imagePositionZ as number),
    );
  }

  // Fallback to sliceLocation
  const hasLoc = slices.every((s) => s.sliceLocation !== null);
  if (hasLoc) {
    return [...slices].sort(
      (a, b) => (a.sliceLocation as number) - (b.sliceLocation as number),
    );
  }

  // Final fallback: instance number
  return [...slices].sort((a, b) => a.instanceNumber - b.instanceNumber);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Load a full DICOM series as a vtk.js-compatible volume.
 *
 * @param seriesId  Backend series UUID
 * @param concurrency  Max concurrent DICOM file fetches (default 8)
 * @returns  Assembled volume data ready for vtkImageData
 */
export async function loadDicomVolume(
  seriesId: string,
  concurrency = 8,
): Promise<DicomVolumeData> {
  // 1. Fetch instance list
  const instances = await dicomService.getInstances(seriesId);
  if (!instances || instances.length === 0) {
    throw new Error(`No instances found for series ${seriesId}`);
  }

  // 2. Fetch & parse all slices with concurrency limit
  const slices: DicomSlice[] = [];
  const errors: string[] = [];

  // Manual concurrency-limited parallel fetch
  const queue = [...instances];
  async function worker() {
    while (queue.length > 0) {
      const inst = queue.shift()!;
      try {
        const buf = await fetchDicomArrayBuffer(inst.id);
        const slice = parseDicomSlice(buf, inst.id);
        slices.push(slice);
      } catch (err: any) {
        errors.push(err.message ?? String(err));
      }
    }
  }

  const workers = Array.from(
    { length: Math.min(concurrency, queue.length) },
    () => worker(),
  );
  await Promise.allSettled(workers);

  // If no slices succeeded, fail
  if (slices.length === 0) {
    throw new Error(
      `Failed to load any slices for series ${seriesId}. ` +
        (errors.length > 0 ? `First error: ${errors[0]}` : ''),
    );
  }

  // Log errors for slices that failed individually
  if (errors.length > 0) {
    console.warn(
      `[dicomVolumeLoader] ${errors.length} slice(s) failed:`,
      errors.slice(0, 5),
    );
  }

  // 3. Sort slices spatially
  const sorted = sortSlices(slices);

  // 4. Validate slice consistency
  const { rows, columns } = sorted[0];
  for (const s of sorted) {
    if (s.rows !== rows || s.columns !== columns) {
      throw new Error(
        `Slice dimensions mismatch: expected ${rows}x${columns}, ` +
          `got ${s.rows}x${s.columns} (instance ${s.instanceNumber})`,
      );
    }
  }

  const sliceCount = sorted.length;

  // 5. Compute spacing
  const first = sorted[0];

  // X-Y spacing
  let spacingX = 1.0;
  let spacingY = 1.0;
  if (first.pixelSpacing) {
    spacingX = first.pixelSpacing[0];
    spacingY = first.pixelSpacing[1];
  }

  // Z spacing: prefer adjacent imagePositionZ difference
  let spacingZ = first.sliceThickness ?? 1.0;
  if (sorted.length >= 2) {
    const z0 = sorted[0].imagePositionZ;
    const z1 = sorted[1].imagePositionZ;
    if (z0 !== null && z1 !== null) {
      const dz = Math.abs(z1 - z0);
      if (dz > 0) spacingZ = dz;
    } else {
      const loc0 = sorted[0].sliceLocation;
      const loc1 = sorted[1].sliceLocation;
      if (loc0 !== null && loc1 !== null) {
        const dz = Math.abs(loc1 - loc0);
        if (dz > 0) spacingZ = dz;
      }
    }
  }

  // 6. Compute origin (first slice's imagePositionPatient, or zero)
  const originX = 0.0;
  const originY = 0.0;
  let originZ = 0.0;
  if (first.imagePositionZ !== null) {
    // We only stored Z — fetch full IPP from a re-parse isn't worth it.
    // For basic 3D rendering the origin is primarily cosmetic;
    // anatomically-aware applications should use the full ImagePositionPatient.
    originZ = first.imagePositionZ;
  } else if (first.sliceLocation !== null) {
    originZ = first.sliceLocation;
  }

  // 7. Assemble HU-normalised Float32 volume
  const voxelCount = columns * rows * sliceCount;
  const scalarData = new Float32Array(voxelCount);
  let dataMin = Infinity;
  let dataMax = -Infinity;

  for (let z = 0; z < sliceCount; z++) {
    const slice = sorted[z];
    const raw = slice.pixelData;
    const slope = slice.slope;
    const intercept = slice.intercept;

    const sliceOffset = z * rows * columns;
    const nPixels = rows * columns;

    for (let i = 0; i < nPixels; i++) {
      const hu = raw[i] * slope + intercept;
      scalarData[sliceOffset + i] = hu;
      if (hu < dataMin) dataMin = hu;
      if (hu > dataMax) dataMax = hu;
    }
  }

  const scalarRange: [number, number] = [
    isFinite(dataMin) ? dataMin : -1024,
    isFinite(dataMax) ? dataMax : 3071,
  ];

  return {
    scalarData,
    dimensions: [columns, rows, sliceCount],
    spacing: [spacingX, spacingY, spacingZ],
    origin: [originX, originY, originZ],
    scalarRange,
  };
}
