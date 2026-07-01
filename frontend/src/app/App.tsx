import { Suspense } from 'react';
import { AppRouter } from '../router/AppRouter';
import { Layout } from '../components/layout/Layout';

/**
 * App Component
 *
 * Root application component for the CT Simulator.
 * Provides the layout shell and wraps routes with Suspense
 * for code-splitting support across all major modules:
 * - DICOM Viewer (Cornerstone3D)
 * - Volume Rendering (vtk.js)
 * - Lesion Simulation
 * - AI Segmentation
 */
function App() {
  return (
    <Suspense
      fallback={
        <div className="flex h-screen items-center justify-center bg-background">
          <div className="flex flex-col items-center gap-4">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            <p className="text-sm text-muted-foreground">Loading CT Simulator...</p>
          </div>
        </div>
      }
    >
      <Layout>
        <AppRouter />
      </Layout>
    </Suspense>
  );
}

export default App;
