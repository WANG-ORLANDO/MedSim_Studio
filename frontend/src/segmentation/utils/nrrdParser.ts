/**
 * NRRD File Parser (Browser-compatible)
 *
 * Parses NRRD (Nearly Raw Raster Data) format files into label arrays
 * for use with Cornerstone3D segmentation volumes.
 *
 * Supported encodings:
 *   - raw (uncompressed binary)
 *   - gzip / gz (gzip-compressed binary, via browser-native DecompressionStream)
 *
 * This parser is designed for SimpleITK-generated NRRD files containing
 * integer label maps (int8, int16, int32, uint8, uint16, uint32).
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface NrrdHeader {
  /** Magic number: NRRD0001-NRRD0005 */
  magic: string;
  /** Data type string from NRRD header */
  type: string;
  /** Number of dimensions */
  dimension: number;
  /** Size in each dimension (x, y, z) */
  sizes: number[];
  /** Voxel spacing in each dimension (x, y, z) */
  spacings: number[];
  /** Origin in world coordinates (x, y, z) */
  origin: number[];
  /** Encoding: raw, gzip, or ascii */
  encoding: string;
  /** Endianness: little or big */
  endian: string;
}

export interface NrrdVolume {
  header: NrrdHeader;
  /** Raw pixel data as the appropriate typed array */
  data: Int32Array;
}

// ---------------------------------------------------------------------------
// Byte order helpers
// ---------------------------------------------------------------------------

function swapEndian32(buffer: ArrayBuffer): ArrayBuffer {
  const view = new DataView(buffer);
  const out = new DataView(new ArrayBuffer(buffer.byteLength));
  for (let i = 0; i < buffer.byteLength; i += 4) {
    out.setUint32(i, view.getUint32(i, true), false); // false = big-endian read
  }
  return out.buffer;
}

// ---------------------------------------------------------------------------
// Main parser
// ---------------------------------------------------------------------------

/**
 * Parse an NRRD file from an ArrayBuffer.
 *
 * @param buffer - Raw file contents as ArrayBuffer
 * @returns Parsed NrrdVolume with Int32Array pixel data
 * @throws Error if the file format is unsupported or parsing fails
 */
export async function parseNrrd(buffer: ArrayBuffer): Promise<NrrdVolume> {
  const header = parseHeader(buffer);
  const pixelOffset = getHeaderByteLength(buffer);
  let rawData: ArrayBuffer;

  if (header.encoding === 'raw') {
    // Raw binary — data starts right after the header
    rawData = buffer.slice(pixelOffset);
  } else if (header.encoding === 'gzip' || header.encoding === 'gz') {
    // Gzip-compressed — decompress using browser-native API
    rawData = await decompressGzip(buffer.slice(pixelOffset));
  } else if (header.encoding === 'ascii' || header.encoding === 'txt') {
    rawData = parseAsciiData(buffer, pixelOffset, header);
  } else {
    throw new Error(`Unsupported NRRD encoding: ${header.encoding}`);
  }

  // Determine the expected byte length
  const typeInfo = getTypeInfo(header.type);
  const numPixels = header.sizes.reduce((a, b) => a * b, 1);
  const expectedBytes = numPixels * typeInfo.bytes;

  if (rawData.byteLength < expectedBytes) {
    throw new Error(
      `NRRD data truncated: expected ${expectedBytes} bytes, got ${rawData.byteLength}`
    );
  }

  // Convert to Int32Array, handling endianness and type conversion
  const data = convertToInt32(rawData, header.type, header.endian, numPixels);

  return { header, data };
}

// ---------------------------------------------------------------------------
// Header parser
// ---------------------------------------------------------------------------

function parseHeader(buffer: ArrayBuffer): NrrdHeader {
  const text = new TextDecoder('utf-8').decode(buffer);
  const lines = text.split(/\r?\n/);

  const header: Record<string, string> = {};

  for (const line of lines) {
    if (line.startsWith('#')) continue;
    if (line.trim() === '') break; // empty line = end of header

    const colonIdx = line.indexOf(':');
    if (colonIdx === -1) continue;

    const key = line.slice(0, colonIdx).trim();
    const value = line.slice(colonIdx + 1).trim();
    header[key] = value;
  }

  if (!header['NRRD']) {
    // The magic "NRRD000X" line has no key, just the value
    // It may appear as "NRRD: NRRD0005" or just "NRRD0005"
    const firstLine = lines[0]?.trim() ?? '';
    if (firstLine.startsWith('NRRD')) {
      header['NRRD'] = firstLine;
    } else {
      throw new Error('Not a valid NRRD file: missing magic number');
    }
  }

  const sizes = (header['sizes'] ?? '')
    .split(/\s+/)
    .filter(Boolean)
    .map(Number);

  const spacings = (header['spacings'] ?? '')
    .split(/\s+/)
    .filter(Boolean)
    .map(Number);

  // Parse origin from "space origin: (x,y,z)" format
  const originStr = header['space origin'] ?? '(0,0,0)';
  const origin = originStr
    .replace(/[()]/g, '')
    .split(',')
    .map((s) => parseFloat(s.trim()));

  return {
    magic: header['NRRD'] ?? '',
    type: header['type'] ?? 'int16',
    dimension: parseInt(header['dimension'] ?? '3', 10),
    sizes,
    spacings: spacings.length > 0 ? spacings : [1, 1, 1],
    origin: origin.length === 3 ? origin : [0, 0, 0],
    encoding: header['encoding'] ?? 'raw',
    endian: header['endian'] ?? 'little',
  };
}

/**
 * Calculate the byte offset where pixel data starts.
 * Scans for the first empty line after the header.
 */
function getHeaderByteLength(buffer: ArrayBuffer): number {
  const bytes = new Uint8Array(buffer);
  // Look for \n\n or \r\n\r\n
  for (let i = 0; i < bytes.length - 3; i++) {
    if (bytes[i] === 0x0a && bytes[i + 1] === 0x0a) {
      // \n\n
      return i + 2;
    }
    if (
      bytes[i] === 0x0d &&
      bytes[i + 1] === 0x0a &&
      bytes[i + 2] === 0x0d &&
      bytes[i + 3] === 0x0a
    ) {
      // \r\n\r\n
      return i + 4;
    }
  }
  // Fallback: search for \n\n in text
  const text = new TextDecoder('utf-8').decode(buffer);
  const idx = text.indexOf('\n\n');
  if (idx !== -1) {
    return new TextEncoder().encode(text.slice(0, idx + 2)).length;
  }
  throw new Error('NRRD: cannot find end of header (no empty line)');
}

// ---------------------------------------------------------------------------
// Gzip decompression
// ---------------------------------------------------------------------------

async function decompressGzip(compressed: ArrayBuffer): Promise<ArrayBuffer> {
  // Use the browser-native CompressionStream API when available
  if (typeof DecompressionStream !== 'undefined') {
    const ds = new DecompressionStream('gzip');
    const writer = ds.writable.getWriter();
    writer.write(new Uint8Array(compressed));
    writer.close();
    const reader = ds.readable.getReader();
    const chunks: Uint8Array[] = [];
    let totalLength = 0;
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        totalLength += value.length;
      }
    }
    const result = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      result.set(chunk, offset);
      offset += chunk.length;
    }
    return result.buffer;
  }

  throw new Error(
    'Gzip decompression not available. ' +
    'This browser does not support DecompressionStream. ' +
    'Please use a modern browser (Chrome 80+, Firefox 113+, Safari 16.4+).'
  );
}

// ---------------------------------------------------------------------------
// ASCII data parsing
// ---------------------------------------------------------------------------

function parseAsciiData(
  buffer: ArrayBuffer,
  offset: number,
  header: NrrdHeader,
): ArrayBuffer {
  const text = new TextDecoder('utf-8').decode(buffer.slice(offset));
  const values = text.trim().split(/\s+/).map(Number);
  const numPixels = header.sizes.reduce((a, b) => a * b, 1);

  if (values.length < numPixels) {
    throw new Error(`ASCII data truncated: expected ${numPixels}, got ${values.length}`);
  }

  const arr = new Int32Array(numPixels);
  for (let i = 0; i < numPixels; i++) {
    arr[i] = Math.round(values[i]);
  }
  return arr.buffer;
}

// ---------------------------------------------------------------------------
// Type conversion
// ---------------------------------------------------------------------------

interface TypeInfo {
  bytes: number;
  signed: boolean;
}

function getTypeInfo(type: string): TypeInfo {
  const map: Record<string, TypeInfo> = {
    'signed char': { bytes: 1, signed: true },
    'int8': { bytes: 1, signed: true },
    'int8_t': { bytes: 1, signed: true },
    'unsigned char': { bytes: 1, signed: false },
    'uint8': { bytes: 1, signed: false },
    'uint8_t': { bytes: 1, signed: false },
    'short': { bytes: 2, signed: true },
    'short int': { bytes: 2, signed: true },
    'signed short': { bytes: 2, signed: true },
    'signed short int': { bytes: 2, signed: true },
    'int16': { bytes: 2, signed: true },
    'int16_t': { bytes: 2, signed: true },
    'unsigned short': { bytes: 2, signed: false },
    'unsigned short int': { bytes: 2, signed: false },
    'uint16': { bytes: 2, signed: false },
    'uint16_t': { bytes: 2, signed: false },
    'int': { bytes: 4, signed: true },
    'signed int': { bytes: 4, signed: true },
    'int32': { bytes: 4, signed: true },
    'int32_t': { bytes: 4, signed: true },
    'unsigned int': { bytes: 4, signed: false },
    'unsigned': { bytes: 4, signed: false },
    'uint32': { bytes: 4, signed: false },
    'uint32_t': { bytes: 4, signed: false },
    'float': { bytes: 4, signed: true },
    'double': { bytes: 8, signed: true },
  };

  return map[type.toLowerCase()] ?? { bytes: 2, signed: true };
}

function convertToInt32(
  rawData: ArrayBuffer,
  type: string,
  endian: string,
  numPixels: number,
): Int32Array {
  const typeInfo = getTypeInfo(type);
  const littleEndian = endian !== 'big';

  // If the data is already int32 (most common for label maps), just wrap it
  if (
    type.toLowerCase() === 'int32' ||
    type.toLowerCase() === 'signed int' ||
    type.toLowerCase() === 'int'
  ) {
    // Check endianness
    if (littleEndian) {
      // Check if we need to swap
      return new Int32Array(rawData.slice(0, numPixels * 4));
    } else {
      const swapped = swapEndian32(rawData.slice(0, numPixels * 4));
      return new Int32Array(swapped);
    }
  }

  // For other types, convert element by element
  const result = new Int32Array(numPixels);
  const view = new DataView(rawData);

  for (let i = 0; i < numPixels; i++) {
    const offset = i * typeInfo.bytes;
    if (offset + typeInfo.bytes > rawData.byteLength) break;

    let value: number;
    switch (typeInfo.bytes) {
      case 1:
        value = typeInfo.signed
          ? view.getInt8(offset)
          : view.getUint8(offset);
        break;
      case 2:
        value = typeInfo.signed
          ? view.getInt16(offset, littleEndian)
          : view.getUint16(offset, littleEndian);
        break;
      case 4:
        if (type.toLowerCase() === 'float') {
          value = Math.round(view.getFloat32(offset, littleEndian));
        } else {
          value = typeInfo.signed
            ? view.getInt32(offset, littleEndian)
            : view.getUint32(offset, littleEndian);
        }
        break;
      case 8:
        value = Math.round(view.getFloat64(offset, littleEndian));
        break;
      default:
        value = 0;
    }
    result[i] = value;
  }

  return result;
}
