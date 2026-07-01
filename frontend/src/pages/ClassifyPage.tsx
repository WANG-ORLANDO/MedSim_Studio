import { useState, useEffect, useCallback } from 'react';
import { Button } from '@components/ui/button';
import { artifactService, type ClassifyResponse, type SeriesInfo, type SliceClassifyResult } from '@/services/artifactService';

const CLASS_COLORS: Record<string, string> = {
  clean: '#22c55e',
  metal: '#ef4444',
  motion: '#f59e0b',
  noise: '#a855f7',
  ring: '#3b82f6',
  streak: '#ec4899',
  beam_hardening: '#14b8a6',
  mixed: '#f97316',
};

const CLASS_LABELS: Record<string, string> = {
  clean: 'Clean',
  metal: 'Metal',
  motion: 'Motion',
  noise: 'Noise',
  ring: 'Ring',
  streak: 'Streak',
  beam_hardening: 'Beam Hardening',
  mixed: 'Mixed',
};

export default function ClassifyPage() {
  const [source, setSource] = useState<'phantom' | 'dicom'>('dicom');
  const [seriesList, setSeriesList] = useState<SeriesInfo[]>([]);
  const [selectedSeries, setSelectedSeries] = useState<string>('');
  const [result, setResult] = useState<ClassifyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    artifactService.getSeries().then(setSeriesList).catch(() => {});
  }, []);

  const handleClassify = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await artifactService.classify({
        source,
        seriesId: source === 'dicom' ? selectedSeries : undefined,
      });
      setResult(res);
    } catch (e: any) {
      setError(e.message || 'Classification failed');
    } finally {
      setLoading(false);
    }
  }, [source, selectedSeries]);

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-gray-100 flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-3 flex items-center gap-4 shrink-0">
        <div className="h-8 w-8 rounded-lg bg-purple-500/20 flex items-center justify-center">
          <svg className="h-5 w-5 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
          </svg>
        </div>
        <h1 className="text-lg font-semibold">Artifact Classifier</h1>
        <span className="text-xs text-gray-500">AI-powered CT artifact detection & classification</span>

        <div className="ml-auto flex items-center gap-2 bg-gray-800 rounded-lg p-1">
          <button
            onClick={() => { setSource('dicom'); setResult(null); }}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${source === 'dicom' ? 'bg-purple-500/20 text-purple-300' : 'text-gray-400 hover:text-gray-200'}`}
          >
            DICOM
          </button>
          <button
            onClick={() => { setSource('phantom'); setResult(null); }}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${source === 'phantom' ? 'bg-purple-500/20 text-purple-300' : 'text-gray-400 hover:text-gray-200'}`}
          >
            Phantom
          </button>
        </div>
      </header>

      <div className="flex flex-1 min-h-0">
        {/* Left Panel — Controls */}
        <div className="w-72 border-r border-gray-800 p-4 overflow-y-auto shrink-0">
          <h2 className="text-sm font-semibold mb-3">Classification Settings</h2>

          {source === 'dicom' && (
            <div className="mb-4 p-3 bg-gray-800/50 rounded-lg border border-gray-700">
              <label className="text-xs text-gray-400 mb-1 block">CT Series</label>
              {seriesList.length === 0 ? (
                <p className="text-[11px] text-gray-500">No CT series found. Upload DICOM first.</p>
              ) : (
                <select
                  value={selectedSeries}
                  onChange={(e) => { setSelectedSeries(e.target.value); setResult(null); }}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-purple-500"
                >
                  <option value="">-- Select Series --</option>
                  {seriesList.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.description || `Series ${s.id}`} ({s.imageCount} slices)
                    </option>
                  ))}
                </select>
              )}
            </div>
          )}

          <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700 mb-4">
            <p className="text-[11px] text-gray-400 mb-2">Model: EfficientNet-B3</p>
            <p className="text-[11px] text-gray-400 mb-2">Classes: 8 types</p>
            <p className="text-[11px] text-gray-500">Analyzes slices across the volume to detect and classify artifacts.</p>
          </div>

          <Button onClick={handleClassify} disabled={loading || (source === 'dicom' && !selectedSeries)}
            className="w-full bg-purple-600 hover:bg-purple-500 text-white text-sm">
            {loading ? 'Classifying...' : 'Classify Artifact'}
          </Button>
          {error && <div className="mt-2 p-2 rounded bg-red-500/10 border border-red-500/20 text-red-400 text-xs">{error}</div>}
        </div>

        {/* Right Panel — Results */}
        <div className="flex-1 p-6 overflow-y-auto">
          {result ? (
            <div className="max-w-4xl mx-auto space-y-6">
              {/* Overall Result */}
              <div className="p-4 bg-gray-800/50 rounded-xl border border-gray-700">
                <div className="flex items-center gap-3 mb-4">
                  <span className="text-sm text-gray-400">Overall Dominant Artifact:</span>
                  <span
                    className="px-3 py-1 rounded-full text-sm font-semibold"
                    style={{
                      backgroundColor: `${CLASS_COLORS[result.dominantArtifact]}20`,
                      color: CLASS_COLORS[result.dominantArtifact],
                      border: `1px solid ${CLASS_COLORS[result.dominantArtifact]}40`,
                    }}
                  >
                    {CLASS_LABELS[result.dominantArtifact] || result.dominantArtifact}
                  </span>
                  <span className="text-xs text-gray-500 ml-auto">{result.sliceCount} slices analyzed</span>
                </div>

                {/* Overall Score Bars */}
                <div className="grid grid-cols-2 gap-x-6 gap-y-2">
                  {Object.entries(result.overallScores)
                    .sort(([, a], [, b]) => b - a)
                    .map(([name, score]) => (
                      <div key={name} className="flex items-center gap-2">
                        <span className="text-[11px] text-gray-400 w-24 shrink-0">{CLASS_LABELS[name] || name}</span>
                        <div className="flex-1 h-3 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                              width: `${Math.round(score * 100)}%`,
                              backgroundColor: CLASS_COLORS[name] || '#888',
                            }}
                          />
                        </div>
                        <span className="text-[10px] text-gray-300 w-10 text-right font-mono">{(score * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                </div>
              </div>

              {/* Per-Slice Results */}
              <div className="p-4 bg-gray-800/50 rounded-xl border border-gray-700">
                <h3 className="text-sm font-semibold mb-3">Per-Slice Results</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-gray-500 border-b border-gray-700">
                        <th className="text-left py-2 px-2">Slice</th>
                        <th className="text-left py-2 px-2">Dominant</th>
                        {Object.keys(CLASS_LABELS).map((name) => (
                          <th key={name} className="text-right py-2 px-1" style={{ color: CLASS_COLORS[name] }}>
                            {CLASS_LABELS[name].slice(0, 4)}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.perSliceScores.map((slice: SliceClassifyResult) => (
                        <tr key={slice.sliceIndex} className="border-b border-gray-800 hover:bg-gray-800/30">
                          <td className="py-1.5 px-2 font-mono text-gray-300">{slice.sliceIndex}</td>
                          <td className="py-1.5 px-2">
                            <span
                              className="px-1.5 py-0.5 rounded text-[10px] font-medium"
                              style={{
                                backgroundColor: `${CLASS_COLORS[slice.dominant]}20`,
                                color: CLASS_COLORS[slice.dominant],
                              }}
                            >
                              {CLASS_LABELS[slice.dominant] || slice.dominant}
                            </span>
                          </td>
                          {Object.keys(CLASS_LABELS).map((name) => (
                            <td key={name} className="py-1.5 px-1 text-right font-mono text-gray-400">
                              {(slice.scores[name] * 100).toFixed(0)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center h-full">
              <div className="text-center">
                <div className="h-20 w-20 mx-auto mb-3 rounded-2xl bg-gray-800/50 border border-gray-700 flex items-center justify-center">
                  <svg className="h-8 w-8 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                  </svg>
                </div>
                <p className="text-sm text-gray-500">Select a CT series and click Classify</p>
                {source === 'phantom' && <p className="text-xs text-gray-600 mt-1">Phantom mode uses a synthetic volume</p>}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
