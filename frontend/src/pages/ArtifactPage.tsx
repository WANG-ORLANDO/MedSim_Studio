import { useState, useRef, useEffect, useCallback } from 'react';
import { Button } from '@components/ui/button';
import { artifactService, type ArtifactGenerateResponse, type SeriesInfo } from '@/services/artifactService';

// ---------------------------------------------------------------------------
// Artifact type configs
// ---------------------------------------------------------------------------

interface ParamField {
  key: string;
  label: string;
  type: 'range' | 'select' | 'multiselect';
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  defaultValue: unknown;
}

interface ArtifactConfig {
  label: string;
  description: string;
  defaultParams: Record<string, unknown>;
  paramFields: ParamField[];
}

const ARTIFACT_CONFIGS: Record<string, ArtifactConfig> = {
  metal: {
    label: 'Metal Artifact',
    description: '高密度金属引起的射束硬化+条纹伪影',
    defaultParams: { metal_type: 'titanium', center: [0.5, 0.5, 0.5], radius_mm: [5, 5, 5], streak_intensity: 0.7, beam_hardening_strength: 0.5 },
    paramFields: [
      { key: 'metal_type', label: 'Metal Type', type: 'select', options: ['titanium', 'stainless_steel', 'dental_amalgam', 'gold', 'copper'], defaultValue: 'titanium' },
      { key: 'streak_intensity', label: 'Streak Intensity', type: 'range', min: 0, max: 1, step: 0.1, defaultValue: 0.7 },
      { key: 'beam_hardening_strength', label: 'Beam Hardening', type: 'range', min: 0, max: 1, step: 0.1, defaultValue: 0.5 },
    ],
  },
  noise: {
    label: 'Quantum Noise',
    description: 'Poisson 光子计数模型量子噪声',
    defaultParams: { mAs: 50, reference_mAs: 150 },
    paramFields: [
      { key: 'mAs', label: 'mAs (dose)', type: 'range', min: 10, max: 300, step: 10, defaultValue: 50 },
    ],
  },
  motion: {
    label: 'Motion Artifact',
    description: '呼吸/心跳/随机运动引起的伪影',
    defaultParams: { motion_type: 'respiratory', amplitude_mm: 10, blur_sigma: 1.5 },
    paramFields: [
      { key: 'motion_type', label: 'Motion Type', type: 'select', options: ['respiratory', 'cardiac', 'random'], defaultValue: 'respiratory' },
      { key: 'amplitude_mm', label: 'Amplitude (mm)', type: 'range', min: 0, max: 30, step: 1, defaultValue: 10 },
      { key: 'blur_sigma', label: 'Blur Sigma', type: 'range', min: 0, max: 5, step: 0.5, defaultValue: 1.5 },
    ],
  },
  ring: {
    label: 'Ring Artifact',
    description: '探测器通道增益不一致引起的同心环',
    defaultParams: { num_rings: 5, intensity: 80 },
    paramFields: [
      { key: 'num_rings', label: 'Number of Rings', type: 'range', min: 1, max: 15, step: 1, defaultValue: 5 },
      { key: 'intensity', label: 'Intensity', type: 'range', min: 10, max: 300, step: 10, defaultValue: 80 },
    ],
  },
  streak: {
    label: 'Streak Artifact',
    description: '穿过高密度区域的直线暗带',
    defaultParams: { num_streaks: 5, intensity: 80 },
    paramFields: [
      { key: 'num_streaks', label: 'Number of Streaks', type: 'range', min: 1, max: 15, step: 1, defaultValue: 5 },
      { key: 'intensity', label: 'Intensity', type: 'range', min: 10, max: 300, step: 10, defaultValue: 80 },
    ],
  },
  beam_hardening: {
    label: 'Beam Hardening',
    description: '多色 X 射线束硬化引起的杯状/暗带效应',
    defaultParams: { cupping_strength: 0.5, dark_band_strength: 0.4 },
    paramFields: [
      { key: 'cupping_strength', label: 'Cupping Strength', type: 'range', min: 0, max: 1, step: 0.1, defaultValue: 0.5 },
      { key: 'dark_band_strength', label: 'Dark Band Strength', type: 'range', min: 0, max: 1, step: 0.1, defaultValue: 0.4 },
    ],
  },
  composite: {
    label: 'Composite Artifact',
    description: '同时施加多种伪影的组合生成器',
    defaultParams: { artifacts: [{ type: 'metal', params: {} }, { type: 'noise', params: {} }] },
    paramFields: [
      { key: 'artifacts', label: 'Sub-Artifacts', type: 'multiselect', options: ['metal', 'noise', 'motion', 'ring', 'streak', 'beam_hardening'], defaultValue: ['metal', 'noise'] },
    ],
  },
};

const ARTIFACT_TYPES = Object.keys(ARTIFACT_CONFIGS);

// ---------------------------------------------------------------------------
// Canvas renderer
// ---------------------------------------------------------------------------

function renderSlice(canvas: HTMLCanvasElement, data: number[][], wl: number, ww: number) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const h = data.length;
  const w = data[0].length;
  canvas.width = w;
  canvas.height = h;
  const imageData = ctx.createImageData(w, h);
  const low = wl - ww / 2;
  const high = wl + ww / 2;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const hu = data[y][x];
      let gray: number;
      if (hu <= low) gray = 0;
      else if (hu >= high) gray = 255;
      else gray = ((hu - low) / ww) * 255;
      const idx = (y * w + x) * 4;
      imageData.data[idx] = gray;
      imageData.data[idx + 1] = gray;
      imageData.data[idx + 2] = gray;
      imageData.data[idx + 3] = 255;
    }
  }
  ctx_putImageData(ctx!, imageData, 0, 0);
}

function ctx_putImageData(ctx: CanvasRenderingContext2D, data: ImageData, x: number, y: number) {
  ctx.putImageData(data, x, y);
}

// ---------------------------------------------------------------------------
// Main Page
// ---------------------------------------------------------------------------

export default function ArtifactPage() {
  const [selectedType, setSelectedType] = useState('metal');
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [source, setSource] = useState<'phantom' | 'dicom'>('phantom');
  const [seriesList, setSeriesList] = useState<SeriesInfo[]>([]);
  const [selectedSeries, setSelectedSeries] = useState<string>('');
  const [sliceIndex, setSliceIndex] = useState(0);
  const [result, setResult] = useState<ArtifactGenerateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [windowLevel, setWindowLevel] = useState(40);
  const [windowWidth, setWindowWidth] = useState(800);
  const [viewMode, setViewMode] = useState<'artifact' | 'original' | 'mask' | 'compare'>('compare');
  const canvasOriginalRef = useRef<HTMLCanvasElement>(null);
  const canvasArtifactRef = useRef<HTMLCanvasElement>(null);

  const config = ARTIFACT_CONFIGS[selectedType];

  // Load CT series on mount
  useEffect(() => {
    artifactService.getSeries().then(setSeriesList).catch(() => {});
  }, []);

  // Reset params when type changes
  useEffect(() => {
    const defaults: Record<string, unknown> = {};
    config.paramFields.forEach((f) => { defaults[f.key] = f.defaultValue; });
    setParams(defaults);
    setResult(null);
    setError(null);
  }, [selectedType]);

  // Render canvases
  useEffect(() => {
    if (!result) return;
    const wl = viewMode === 'mask' ? 0.5 : windowLevel;
    const ww = viewMode === 'mask' ? 1 : windowWidth;

    if ((viewMode === 'original' || viewMode === 'compare') && canvasOriginalRef.current) {
      renderSlice(canvasOriginalRef.current, result.originalSlice, wl, ww);
    }
    if ((viewMode === 'artifact' || viewMode === 'compare' || viewMode === 'mask') && canvasArtifactRef.current) {
      const data = viewMode === 'mask' ? result.maskSlice : result.artifactSlice;
      renderSlice(canvasArtifactRef.current, data, wl, ww);
    }
  }, [result, windowLevel, windowWidth, viewMode]);

  const handleGenerate = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const apiParams = { ...params };
      if (selectedType === 'composite' && Array.isArray(apiParams.artifacts)) {
        apiParams.artifacts = (apiParams.artifacts as string[]).map((t) => ({ type: t, params: {} }));
      }
      const res = await artifactService.generate({
        artifactType: selectedType,
        params: apiParams,
        source,
        seriesId: source === 'dicom' ? selectedSeries : undefined,
        sliceIndex,
      });
      setResult(res);
      setSliceIndex(Math.min(sliceIndex, res.shape[0] - 1));
    } catch (e: any) {
      setError(e.message || 'Generation failed');
    } finally {
      setLoading(false);
    }
  }, [selectedType, params, source, selectedSeries, sliceIndex]);

  const updateParam = (key: string, value: unknown) => {
    setParams((prev) => ({ ...prev, [key]: value }));
  };

  const selectedSeriesInfo = seriesList.find(s => s.id === selectedSeries);

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-gray-100 flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-3 flex items-center gap-4 shrink-0">
        <div className="h-8 w-8 rounded-lg bg-cyan-500/20 flex items-center justify-center">
          <svg className="h-5 w-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
        </div>
        <h1 className="text-lg font-semibold">CT Artifact Lab</h1>
        <span className="text-xs text-gray-500">Interactive artifact generation & visualization</span>

        {/* Source Toggle */}
        <div className="ml-auto flex items-center gap-2 bg-gray-800 rounded-lg p-1">
          <button
            onClick={() => setSource('phantom')}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${source === 'phantom' ? 'bg-cyan-500/20 text-cyan-300' : 'text-gray-400 hover:text-gray-200'}`}
          >
            Phantom
          </button>
          <button
            onClick={() => setSource('dicom')}
            className={`px-3 py-1 text-xs rounded-md transition-colors ${source === 'dicom' ? 'bg-cyan-500/20 text-cyan-300' : 'text-gray-400 hover:text-gray-200'}`}
          >
            DICOM
          </button>
        </div>
      </header>

      <div className="flex flex-1 min-h-0">
        {/* Left Sidebar — Artifact Types */}
        <aside className="w-48 border-r border-gray-800 p-3 flex flex-col gap-1 shrink-0 overflow-y-auto">
          <p className="text-[10px] uppercase tracking-wider text-gray-500 mb-2 px-2">Artifacts</p>
          {Object.keys(ARTIFACT_CONFIGS).map((type) => {
            const cfg = ARTIFACT_CONFIGS[type];
            const active = type === selectedType;
            return (
              <button
                key={type}
                onClick={() => setSelectedType(type)}
                className={`text-left px-3 py-2 rounded-lg text-sm transition-all ${active ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30' : 'text-gray-400 hover:bg-gray-800/50 hover:text-gray-200 border border-transparent'}`}
              >
                {cfg.label}
              </button>
            );
          })}
        </aside>

        {/* Controls Panel */}
        <div className="w-72 border-r border-gray-800 p-4 overflow-y-auto shrink-0">
          <h2 className="text-sm font-semibold mb-1">{config.label}</h2>
          <p className="text-[11px] text-gray-500 mb-4">{config.description}</p>

          {/* DICOM Series Selector */}
          {source === 'dicom' && (
            <div className="mb-4 p-3 bg-gray-800/50 rounded-lg border border-gray-700">
              <label className="text-xs text-gray-400 mb-1 block">CT Series</label>
              {seriesList.length === 0 ? (
                <p className="text-[11px] text-gray-500">No CT series found. Upload DICOM first.</p>
              ) : (
                <select
                  value={selectedSeries}
                  onChange={(e) => { setSelectedSeries(e.target.value); setResult(null); }}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-cyan-500"
                >
                  <option value="">-- Select Series --</option>
                  {seriesList.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.description || `Series ${s.id}`} ({s.imageCount} slices, {s.columns}×{s.rows})
                    </option>
                  ))}
                </select>
              )}
              {selectedSeriesInfo && (
                <div className="mt-2 text-[10px] text-gray-500 space-y-0.5">
                  <div>Slices: {selectedSeriesInfo.imageCount} | Size: {selectedSeriesInfo.columns}×{selectedSeriesInfo.rows}</div>
                </div>
              )}
            </div>
          )}

          {/* Artifact Params */}
          <div className="space-y-3">
            {config.paramFields.map((field) => (
              <div key={field.key}>
                <label className="text-[11px] text-gray-400 mb-1 block">{field.label}</label>
                {field.type === 'range' && (
                  <div className="flex items-center gap-2">
                    <input
                      type="range" min={field.min} max={field.max} step={field.step}
                      value={(params[field.key] as number) ?? field.defaultValue}
                      onChange={(e) => updateParam(field.key, parseFloat(e.target.value))}
                      className="flex-1 h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer accent-cyan-500"
                    />
                    <span className="text-[11px] text-gray-300 w-10 text-right font-mono">{String(params[field.key] ?? field.defaultValue)}</span>
                  </div>
                )}
                {field.type === 'select' && (
                  <select
                    value={String(params[field.key] ?? field.defaultValue)}
                    onChange={(e) => updateParam(field.key, e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-cyan-500"
                  >
                    {field.options?.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
                  </select>
                )}
                {field.type === 'multiselect' && (
                  <div className="flex flex-wrap gap-1.5">
                    {field.options?.map((opt) => {
                      const currentVal = (params[field.key] ?? field.defaultValue) as string[];
                      const selected = currentVal.includes(opt);
                      return (
                        <button
                          key={opt}
                          type="button"
                          onClick={() => {
                            const current = (params[field.key] ?? field.defaultValue) as string[];
                            const next = selected ? current.filter((v) => v !== opt) : [...current, opt];
                            updateParam(field.key, next);
                          }}
                          className={`px-2 py-1 text-[11px] rounded-md border transition-colors ${selected ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30' : 'bg-gray-800 text-gray-400 border-gray-700 hover:text-gray-200'}`}
                        >
                          {opt}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Slice Selector */}
          {result && (
            <div className="mt-4 pt-3 border-t border-gray-800">
              <label className="text-[11px] text-gray-400 mb-1 block">Slice: {sliceIndex} / {result.shape[0] - 1}</label>
              <input
                type="range" min={0} max={result.shape[0] - 1} step={1}
                value={sliceIndex}
                onChange={(e) => setSliceIndex(parseInt(e.target.value))}
                className="w-full h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer accent-cyan-500"
              />
            </div>
          )}

          {/* Window/Level */}
          <div className="mt-4 pt-3 border-t border-gray-800">
            <p className="text-[10px] uppercase tracking-wider text-gray-500 mb-2">Window / Level</p>
            <div className="space-y-2">
              <div>
                <label className="text-[11px] text-gray-400 block">WL: {windowLevel} HU</label>
                <input type="range" min={-1000} max={3000} step={10} value={windowLevel}
                  onChange={(e) => setWindowLevel(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer accent-cyan-500" />
              </div>
              <div>
                <label className="text-[11px] text-gray-400 block">WW: {windowWidth} HU</label>
                <input type="range" min={100} max={4000} step={50} value={windowWidth}
                  onChange={(e) => setWindowWidth(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer accent-cyan-500" />
              </div>
            </div>
            <div className="flex gap-1.5 mt-2">
              {[{ l: 'Soft', wl: 40, ww: 400 }, { l: 'Lung', wl: -600, ww: 1500 }, { l: 'Bone', wl: 500, ww: 2000 }].map((p) => (
                <button key={p.l} onClick={() => { setWindowLevel(p.wl); setWindowWidth(p.ww); }}
                  className="px-2 py-0.5 text-[10px] rounded bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200 transition-colors">
                  {p.l}
                </button>
              ))}
            </div>
          </div>

          {/* Generate */}
          <Button onClick={handleGenerate} disabled={loading || (source === 'dicom' && !selectedSeries)}
            className="w-full mt-4 bg-cyan-600 hover:bg-cyan-500 text-white text-sm">
            {loading ? 'Generating...' : 'Generate Artifact'}
          </Button>
          {error && <div className="mt-2 p-2 rounded bg-red-500/10 border border-red-500/20 text-red-400 text-xs">{error}</div>}
        </div>

        {/* Visualization Panel */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* View Mode Tabs */}
          <div className="flex items-center gap-1 px-4 py-2 border-b border-gray-800 shrink-0">
            {(['compare', 'original', 'artifact', 'mask'] as const).map((mode) => (
              <button key={mode} onClick={() => setViewMode(mode)}
                className={`px-3 py-1 text-xs rounded-md transition-colors capitalize ${viewMode === mode ? 'bg-cyan-500/20 text-cyan-300' : 'text-gray-400 hover:text-gray-200'}`}>
                {mode}
              </button>
            ))}
            {result && (
              <span className="ml-auto text-[10px] text-gray-500 font-mono">
                {result.shape.join('×')} | spacing {result.spacing.map(s => s.toFixed(1)).join(',')}mm | {result.source}
              </span>
            )}
          </div>

          {/* Canvas Area */}
          <div className="flex-1 flex items-center justify-center p-6 gap-6 overflow-auto">
            {result ? (
              <>
                {(viewMode === 'compare' || viewMode === 'original') && (
                  <div className="flex flex-col items-center gap-2">
                    <span className="text-[11px] text-gray-400">Original</span>
                    <canvas ref={canvasOriginalRef} className="border border-gray-700 rounded-lg bg-black" style={{ width: 360, height: 360, imageRendering: 'pixelated' }} />
                  </div>
                )}
                {(viewMode === 'compare' || viewMode === 'artifact' || viewMode === 'mask') && (
                  <div className="flex flex-col items-center gap-2">
                    <span className="text-[11px] text-gray-400">{viewMode === 'mask' ? 'Artifact Mask' : 'With Artifact'}</span>
                    <canvas ref={canvasArtifactRef} className="border border-gray-700 rounded-lg bg-black" style={{ width: 360, height: 360, imageRendering: 'pixelated' }} />
                  </div>
                )}
              </>
            ) : (
              <div className="text-center">
                <div className="h-20 w-20 mx-auto mb-3 rounded-2xl bg-gray-800/50 border border-gray-700 flex items-center justify-center">
                  <svg className="h-8 w-8 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                </div>
                <p className="text-sm text-gray-500">Select artifact type and click Generate</p>
                {source === 'dicom' && <p className="text-xs text-gray-600 mt-1">Choose a CT series from the left panel</p>}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
