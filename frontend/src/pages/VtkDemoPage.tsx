import { useState } from 'react';
import { VolumeRenderer } from '@vtk';

/**
 * VtkDemoPage — development sandbox for vtk.js volume rendering.
 *
 * Supports two modes:
 *   - Synthetic: 64³ CT phantom (Phase 1)
 *   - DICOM:     real DICOM series loaded from the backend (Phase 2)
 *
 * Route: /vtk-demo
 */
export default function VtkDemoPage() {
  const [mode, setMode] = useState<'synthetic' | 'dicom'>('synthetic');
  const [seriesIdInput, setSeriesIdInput] = useState('');
  const [activeSeriesId, setActiveSeriesId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleLoadDicom = () => {
    const trimmed = seriesIdInput.trim();
    if (!trimmed) {
      setError('Please enter a series UUID.');
      return;
    }
    setError(null);
    setMode('dicom');
    setActiveSeriesId(trimmed);
  };

  const handleBackToSynthetic = () => {
    setError(null);
    setMode('synthetic');
    setActiveSeriesId(null);
  };

  return (
    <div className="flex min-h-screen flex-col bg-black text-white">
      {/* ---- header bar ---- */}
      <div className="flex flex-wrap items-center gap-3 border-b border-white/10 bg-black/80 px-4 py-2">
        <span className="text-sm font-medium text-white/80">
          {mode === 'synthetic'
            ? 'vtk.js Demo — Synthetic CT Phantom'
            : 'vtk.js Demo — DICOM Volume'}
        </span>

        <span className="rounded bg-yellow-500/20 px-2 py-0.5 text-[10px] font-medium uppercase text-yellow-400">
          Dev
        </span>

        {/* spacer */}
        <div className="flex-1" />

        {/* mode controls */}
        <div className="flex items-center gap-2">
          {mode === 'synthetic' ? (
            <>
              <input
                type="text"
                value={seriesIdInput}
                onChange={(e) => {
                  setSeriesIdInput(e.target.value);
                  setError(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleLoadDicom();
                }}
                placeholder="Paste series UUID..."
                className="rounded border border-white/20 bg-white/10 px-2 py-1 text-xs text-white placeholder:text-white/40 w-64"
              />
              <button
                type="button"
                onClick={handleLoadDicom}
                className="rounded bg-primary px-3 py-1 text-xs font-medium text-primary-foreground hover:bg-primary/80"
              >
                Load DICOM
              </button>
            </>
          ) : (
            <button
              type="button"
              onClick={handleBackToSynthetic}
              className="rounded bg-white/10 px-3 py-1 text-xs font-medium text-white/80 hover:bg-white/20"
            >
              ← Back to Synthetic
            </button>
          )}
        </div>
      </div>

      {/* ---- error banner ---- */}
      {error && (
        <div className="border-b border-red-500/30 bg-red-500/10 px-4 py-1.5 text-xs text-red-400">
          {error}
        </div>
      )}

      {/* ---- volume viewport ---- */}
      <div className="relative flex-1 overflow-hidden bg-black">
        {mode === 'dicom' && activeSeriesId ? (
          <VolumeRenderer
            mode="dicom"
            seriesId={activeSeriesId}
            showControls
            opacityPreset="ct-soft-tissue"
          />
        ) : (
          <VolumeRenderer showControls opacityPreset="ct-soft-tissue" />
        )}
      </div>
    </div>
  );
}
