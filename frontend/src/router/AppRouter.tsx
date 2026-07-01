import { lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';

// Lazy-loaded pages for code splitting
const ViewerPage = lazy(() => import('@pages/ViewerPage'));
const StudiesPage = lazy(() => import('@pages/StudiesPage'));
const SimulationPage = lazy(() => import('@pages/SimulationPage'));
const SegmentationPage = lazy(() => import('@pages/SegmentationPage'));
const VtkDemoPage = lazy(() => import('@pages/VtkDemoPage'));
const ArtifactPage = lazy(() => import('@pages/ArtifactPage'));
const ClassifierPage = lazy(() => import('@pages/ClassifierPage'));

/**
 * Application Router
 *
 * Defines all application routes for the CT Simulator platform.
 * Uses lazy loading for each major module to keep initial bundle size small.
 *
 * Routes:
 * - /viewer/:studyId — Main DICOM viewer with MPR
 * - /studies — Study management and DICOM upload
 * - /simulation — Lesion and organ simulation tools
 * - /segmentation — AI-powered segmentation
 * - / — Redirects to studies as landing page
 */
export function AppRouter() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/studies" replace />} />
      <Route path="/studies" element={<StudiesPage />} />
      <Route path="/viewer/:studyId" element={<ViewerPage />} />
      <Route path="/simulation" element={<SimulationPage />} />
      <Route path="/segmentation" element={<SegmentationPage />} />
      <Route path="/vtk-demo" element={<VtkDemoPage />} />
      <Route path="/artifact" element={<ArtifactPage />} />
      <Route path="/classifier" element={<ClassifierPage />} />
    </Routes>
  );
}
