/**
 * Hounsfield Unit (HU) Utilities
 *
 * Standard CT tissue density values and conversion utilities.
 * HU scale: Air = -1000, Water = 0, Bone = +1000+
 */

/** Standard CT tissue types and their HU ranges */
export const TISSUE_HU_RANGES = {
  air: { min: -1000, max: -800, mean: -1000 },
  lung: { min: -800, max: -200, mean: -500 },
  fat: { min: -150, max: -50, mean: -100 },
  water: { min: -10, max: 10, mean: 0 },
  csf: { min: 5, max: 15, mean: 10 },
  softTissue: { min: 20, max: 80, mean: 40 },
  muscle: { min: 30, max: 60, mean: 45 },
  liver: { min: 50, max: 70, mean: 60 },
  kidney: { min: 20, max: 50, mean: 35 },
  blood: { min: 40, max: 60, mean: 50 },
  bone: { min: 300, max: 1500, mean: 700 },
  contrast: { min: 100, max: 400, mean: 200 },
  metal: { min: 1500, max: 10000, mean: 3000 },
} as const;

/** Lesion HU characteristics by type */
export const LESION_HU_RANGES = {
  tumor: { mean: 40, std: 20 },
  nodule: { mean: -100, std: 50 },
  cyst: { mean: 10, std: 5 },
  calcification: { mean: 300, std: 100 },
  metastasis: { mean: 50, std: 25 },
} as const;

/**
 * Convert pixel value to HU using DICOM Rescale Slope/Intercept
 */
export function pixelToHU(
  pixelValue: number,
  rescaleSlope: number = 1,
  rescaleIntercept: number = -1024,
): number {
  return pixelValue * rescaleSlope + rescaleIntercept;
}

/**
 * Convert HU value back to pixel value
 */
export function huToPixel(
  hu: number,
  rescaleSlope: number = 1,
  rescaleIntercept: number = -1024,
): number {
  return (hu - rescaleIntercept) / rescaleSlope;
}

/**
 * Map HU value to grayscale (0-255) with window/level
 */
export function huToGrayscale(
  hu: number,
  windowCenter: number = 40,
  windowWidth: number = 400,
): number {
  const halfWidth = windowWidth / 2;
  const lower = windowCenter - halfWidth;
  const upper = windowCenter + halfWidth;

  if (hu <= lower) return 0;
  if (hu >= upper) return 255;

  return Math.round(((hu - lower) / windowWidth) * 255);
}
