/**
 * DICOM Type Definitions
 *
 * Core types for DICOM medical imaging data structures used
 * across the viewer, simulation, and segmentation modules.
 */

/** DICOM study metadata */
export interface DicomStudy {
  id: string;
  patientId: string;
  patientName: string;
  patientBirthDate: string;
  patientSex: string;
  studyInstanceUid: string;
  studyDate: string;
  studyTime: string;
  studyDescription: string;
  accessionNumber: string;
  referringPhysician: string;
  modalities: string[];
  seriesCount: number;
  instanceCount: number;
  createdAt: string;
  updatedAt: string;
}

/** DICOM series within a study */
export interface DicomSeries {
  id: string;
  studyId: string;
  seriesInstanceUid: string;
  seriesNumber: number;
  seriesDescription: string;
  modality: string;
  manufacturer: string;
  bodyPartExamined: string;
  laterality: string;
  protocolName: string;
  imageCount: number;
  seriesDate: string;
  rows: number;
  columns: number;
  sliceThickness: number;
  pixelSpacing: [number, number];
  windowCenter: number;
  windowWidth: number;
}

/** DICOM instance (single slice) within a series */
export interface DicomInstance {
  id: string;
  seriesId: string;
  sopInstanceUid: string;
  instanceNumber: number | null;
  sliceLocation: number | null;
  rows: number | null;
  columns: number | null;
}

/** Viewport layout configuration */
export interface ViewportLayout {
  id: string;
  type: 'axial' | 'sagittal' | 'coronal' | '3d' | 'mpr';
  rows: number;
  columns: number;
}

/** MPR (Multi-Planar Reconstruction) view state */
export interface MPRViewState {
  axial: ViewportState;
  sagittal: ViewportState;
  coronal: ViewportState;
  crosshairPosition: [number, number, number];
  linked: boolean;
}

/** Individual viewport state */
export interface ViewportState {
  windowCenter: number;
  windowWidth: number;
  rotation: number;
  zoom: number;
  pan: [number, number];
  sliceIndex: number;
}

/** Window/Level preset */
export interface WindowLevelPreset {
  name: string;
  windowCenter: number;
  windowWidth: number;
  description: string;
}

/** Common CT window/level presets */
export const WINDOW_LEVEL_PRESETS: WindowLevelPreset[] = [
  { name: 'Lung', windowCenter: -600, windowWidth: 1500, description: 'Lung parenchyma' },
  { name: 'Mediastinum', windowCenter: 40, windowWidth: 400, description: 'Soft tissues' },
  { name: 'Bone', windowCenter: 300, windowWidth: 1500, description: 'Bone structures' },
  { name: 'Brain', windowCenter: 40, windowWidth: 80, description: 'Brain tissue' },
  { name: 'Abdomen', windowCenter: 50, windowWidth: 350, description: 'Abdominal organs' },
  { name: 'Liver', windowCenter: 60, windowWidth: 150, description: 'Liver parenchyma' },
];
