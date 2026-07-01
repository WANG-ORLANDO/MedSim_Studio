import { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from '@components/ui/button';
import {
  artifactService,
  type TrainStatusResponse, type TrainEpochResult, type TrainHistoryResponse,
  type ClassifyResponse, type SeriesInfo, type SliceClassifyResult,
} from '@/services/artifactService';
// Re-exported from types for convenience

const CLASS_COLORS: Record<string, string> = {
  clean: '#22c55e', metal: '#ef4444', motion: '#f59e0b', noise: '#a855f7',
  ring: '#3b82f6', streak: '#ec4899', beam_hardening: '#14b8a6', mixed: '#f97316',
};
const CLASS_LABELS: Record<string, string> = {
  clean: 'Clean', metal: 'Metal', motion: 'Motion', noise: 'Noise',
  ring: 'Ring', streak: 'Streak', beam_hardening: 'Beam Hardening', mixed: 'Mixed',
};

type Tab = 'train' | 'classify';

export default function ClassifierPage() {
  const [tab, setTab] = useState<Tab>('classify');
  return (
    <div className="min-h-screen bg-[#0a0a0f] text-gray-100 flex flex-col">
      <header className="border-b border-gray-800 px-6 py-3 flex items-center gap-4 shrink-0">
        <div className="h-8 w-8 rounded-lg bg-purple-500/20 flex items-center justify-center">
          <svg className="h-5 w-5 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2" />
            <rect x="9" y="3" width="6" height="4" rx="1" />
            <path d="M9 14l2 2 4-4" />
          </svg>
        </div>
        <h1 className="text-lg font-semibold">Artifact Classifier</h1>
        <span className="text-xs text-gray-500">Train & classify CT artifacts</span>

        {/* Tab Switcher */}
        <div className="ml-auto flex items-center gap-1 bg-gray-800 rounded-lg p-1">
          {([['classify', 'Classify'], ['train', 'Train']] as const).map(([key, label]) => (
            <button key={key} onClick={() => setTab(key)}
              className={`px-4 py-1 text-xs rounded-md transition-colors ${tab === key ? 'bg-purple-500/20 text-purple-300' : 'text-gray-400 hover:text-gray-200'}`}>
              {label}
            </button>
          ))}
        </div>
      </header>

      <div className="flex-1 min-h-0 overflow-y-auto">
        {tab === 'classify' ? <ClassifyTab /> : <TrainTab />}
      </div>
    </div>
  );
}

// ============================================================
// Classify Tab
// ============================================================

function ClassifyTab() {
  const [source, setSource] = useState<'phantom' | 'dicom'>('dicom');
  const [seriesList, setSeriesList] = useState<SeriesInfo[]>([]);
  const [selectedSeries, setSelectedSeries] = useState<string>('');
  const [result, setResult] = useState<ClassifyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { artifactService.getSeries().then(setSeriesList).catch(() => {}); }, []);

  const handleClassify = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const res = await artifactService.classify({ source, seriesId: source === 'dicom' ? selectedSeries : undefined });
      setResult(res);
    } catch (e: any) { setError(e.message || 'Classification failed'); }
    finally { setLoading(false); }
  }, [source, selectedSeries]);

  return (
    <div className="flex h-full">
      {/* Left Controls */}
      <div className="w-64 border-r border-gray-800 p-4 shrink-0">
        <div className="flex items-center gap-2 bg-gray-800 rounded-lg p-1 mb-4">
          <button onClick={() => { setSource('dicom'); setResult(null); }}
            className={`flex-1 px-2 py-1 text-xs rounded-md transition-colors ${source === 'dicom' ? 'bg-purple-500/20 text-purple-300' : 'text-gray-400 hover:text-gray-200'}`}>DICOM</button>
          <button onClick={() => { setSource('phantom'); setResult(null); }}
            className={`flex-1 px-2 py-1 text-xs rounded-md transition-colors ${source === 'phantom' ? 'bg-purple-500/20 text-purple-300' : 'text-gray-400 hover:text-gray-200'}`}>Phantom</button>
        </div>

        {source === 'dicom' && (
          <div className="mb-4">
            <label className="text-xs text-gray-400 mb-1 block">CT Series</label>
            {seriesList.length === 0 ? (
              <p className="text-[11px] text-gray-500">No CT series found.</p>
            ) : (
              <select value={selectedSeries} onChange={e => { setSelectedSeries(e.target.value); setResult(null); }}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-purple-500">
                <option value="">-- Select Series --</option>
                {seriesList.map(s => <option key={s.id} value={s.id}>{s.description || `Series ${s.id}`} ({s.imageCount} slices)</option>)}
              </select>
            )}
          </div>
        )}

        <p className="text-[11px] text-gray-500 mb-4">Analyzes slices to detect & classify artifacts using EfficientNet-B3.</p>

        <Button onClick={handleClassify} disabled={loading || (source === 'dicom' && !selectedSeries)}
          className="w-full bg-purple-600 hover:bg-purple-500 text-white text-sm">
          {loading ? 'Classifying...' : 'Classify'}
        </Button>
        {error && <div className="mt-2 p-2 rounded bg-red-500/10 border border-red-500/20 text-red-400 text-xs">{error}</div>}
      </div>

      {/* Results */}
      <div className="flex-1 p-6 overflow-y-auto">
        {result ? (
          <div className="max-w-4xl space-y-4">
            <div className="p-4 bg-gray-800/50 rounded-xl border border-gray-700">
              <div className="flex items-center gap-3 mb-3">
                <span className="text-sm text-gray-400">Dominant:</span>
                <span className="px-3 py-1 rounded-full text-sm font-semibold" style={{ backgroundColor: `${CLASS_COLORS[result.dominantArtifact]}20`, color: CLASS_COLORS[result.dominantArtifact], border: `1px solid ${CLASS_COLORS[result.dominantArtifact]}40` }}>
                  {CLASS_LABELS[result.dominantArtifact] || result.dominantArtifact}
                </span>
                <span className="text-xs text-gray-500 ml-auto">{result.sliceCount} slices analyzed</span>
              </div>
              <div className="grid grid-cols-2 gap-x-6 gap-y-1.5">
                {Object.entries(result.overallScores).sort(([, a], [, b]) => b - a).map(([name, score]) => (
                  <div key={name} className="flex items-center gap-2">
                    <span className="text-[11px] text-gray-400 w-24 shrink-0">{CLASS_LABELS[name] || name}</span>
                    <div className="flex-1 h-2.5 bg-gray-700 rounded-full overflow-hidden">
                      <div className="h-full rounded-full transition-all duration-500" style={{ width: `${Math.round(score * 100)}%`, backgroundColor: CLASS_COLORS[name] || '#888' }} />
                    </div>
                    <span className="text-[10px] text-gray-300 w-10 text-right font-mono">{(score * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="p-4 bg-gray-800/50 rounded-xl border border-gray-700">
              <h3 className="text-sm font-semibold mb-2">Per-Slice Results</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-gray-500 border-b border-gray-700">
                      <th className="text-left py-2 px-2">Slice</th>
                      <th className="text-left py-2 px-2">Dominant</th>
                      {Object.keys(CLASS_LABELS).map(n => <th key={n} className="text-right py-2 px-1" style={{ color: CLASS_COLORS[n] }}>{CLASS_LABELS[n].slice(0, 4)}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {result.perSliceScores.map((s: SliceClassifyResult) => (
                      <tr key={s.sliceIndex} className="border-b border-gray-800 hover:bg-gray-800/30">
                        <td className="py-1.5 px-2 font-mono text-gray-300">{s.sliceIndex}</td>
                        <td className="py-1.5 px-2">
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-medium" style={{ backgroundColor: `${CLASS_COLORS[s.dominant]}20`, color: CLASS_COLORS[s.dominant] }}>
                            {CLASS_LABELS[s.dominant] || s.dominant}
                          </span>
                        </td>
                        {Object.keys(CLASS_LABELS).map(n => <td key={n} className="py-1.5 px-1 text-right font-mono text-gray-400">{(s.scores[n] * 100).toFixed(0)}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="h-16 w-16 mx-auto mb-3 rounded-2xl bg-gray-800/50 border border-gray-700 flex items-center justify-center">
                <svg className="h-7 w-7 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                </svg>
              </div>
              <p className="text-sm text-gray-500">Select a CT series and click Classify</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================
// Train Tab
// ============================================================

function TrainTab() {
  const [config, setConfig] = useState({ epochs: 10, batchSize: 32, learningRate: 0.0001, numVolumes: 20, outputDir: '/app/models/artifact_classifier' });
  const [status, setStatus] = useState<TrainStatusResponse | null>(null);
  const [history, setHistory] = useState<TrainHistoryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const pollStatus = useCallback(async () => {
    try {
      const s = await artifactService.getTrainStatus();
      setStatus(s);
      if (s.status === 'training' || s.status === 'starting') { if (s.startTime) setStartTime(s.startTime); }
      if (s.status === 'completed' || s.status === 'failed') { setLoading(false); const h = await artifactService.getTrainHistory(); setHistory(h); }
    } catch { /* ignore poll errors */ }
  }, []);

  useEffect(() => {
    if (status?.status === 'training' && startTime) {
      const tick = () => setElapsed(Date.now() / 1000 - startTime);
      tick(); const id = setInterval(tick, 1000); return () => clearInterval(id);
    }
  }, [status?.status, startTime]);

  useEffect(() => {
    if (status?.status === 'training' || status?.status === 'starting') {
      intervalRef.current = setInterval(pollStatus, 3000);
      return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
    }
  }, [status?.status, pollStatus]);

  useEffect(() => { artifactService.getTrainHistory().then(setHistory).catch(() => {}); pollStatus(); }, [pollStatus]);

  // Draw chart
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const epochs = status?.epochHistory?.length ? status.epochHistory : history?.epochs ?? [];
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = canvas.offsetWidth; canvas.height = 240;
    const W = canvas.width, H = canvas.height;
    const pad = { top: 25, right: 55, bottom: 25, left: 45 };
    const pW = W - pad.left - pad.right, pH = H - pad.top - pad.bottom;

    ctx.fillStyle = '#1f2937'; ctx.fillRect(0, 0, W, H);
    if (epochs.length === 0) { ctx.fillStyle = '#6b7280'; ctx.font = '13px sans-serif'; ctx.textAlign = 'center'; ctx.fillText('No training data yet', W / 2, H / 2); return; }

    const allLoss = epochs.flatMap(e => [e.trainLoss, e.valLoss]);
    const allAcc = epochs.flatMap(e => [e.trainAcc, e.valAcc]);
    const maxL = Math.max(...allLoss, 0.01) * 1.1, maxA = Math.min(Math.max(...allAcc, 0.5), 1.05), minA = Math.min(...allAcc, 0);
    const n = epochs.length;

    ctx.strokeStyle = '#374151'; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) { const y = pad.top + (pH / 4) * i; ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke(); }

    ctx.fillStyle = '#9ca3af'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
    const step = Math.max(1, Math.floor(n / 6));
    for (let i = 0; i < n; i += step) { ctx.fillText(String(epochs[i].epoch), pad.left + (i / (n - 1 || 1)) * pW, H - 6); }

    const c = ctx!;
    function line(data: number[], color: string, maxV: number, minV = 0) {
      c.strokeStyle = color; c.lineWidth = 2; c.beginPath();
      data.forEach((v, i) => { const x = pad.left + (i / (n - 1 || 1)) * pW; const y = pad.top + pH - ((v - minV) / (maxV - minV || 1)) * pH; i === 0 ? c.moveTo(x, y) : c.lineTo(x, y); });
      c.stroke();
    }
    line(epochs.map(e => e.trainLoss), '#60a5fa', maxL); line(epochs.map(e => e.valLoss), '#f87171', maxL);
    line(epochs.map(e => e.trainAcc), '#34d399', maxA, minA); line(epochs.map(e => e.valAcc), '#fbbf24', maxA, minA);

    c.fillStyle = '#60a5fa'; c.textAlign = 'right';
    for (let i = 0; i <= 4; i++) { c.fillText((maxL / 4 * (4 - i)).toFixed(3), pad.left - 4, pad.top + (pH / 4) * i + 3); }
    c.fillStyle = '#34d399'; c.textAlign = 'left';
    for (let i = 0; i <= 4; i++) { c.fillText((minA + (maxA - minA) / 4 * (4 - i)).toFixed(2), W - pad.right + 4, pad.top + (pH / 4) * i + 3); }

    c.font = '10px sans-serif';
    [{ l: 'Train Loss', c: '#60a5fa' }, { l: 'Val Loss', c: '#f87171' }, { l: 'Train Acc', c: '#34d399' }, { l: 'Val Acc', c: '#fbbf24' }].forEach((item, i) => {
      const x = pad.left + 8 + i * 90;
      c.fillStyle = item.c; c.fillRect(x, 6, 10, 8);
      c.fillStyle = '#d1d5db'; c.textAlign = 'left'; c.fillText(item.l, x + 14, 14);
    });
  }, [status?.epochHistory, history]);

  const handleStart = useCallback(async () => {
    setLoading(true); setError(null); setStartTime(Date.now() / 1000);
    try { await artifactService.startTraining(config); await pollStatus(); }
    catch (e: any) { setError(e.message || 'Failed to start training'); setLoading(false); }
  }, [config, pollStatus]);

  const isRunning = status?.status === 'training' || status?.status === 'starting';
  const progress = status && status.totalEpochs > 0 ? (status.currentEpoch / status.totalEpochs) * 100 : 0;

  return (
    <div className="flex h-full">
      {/* Left Config */}
      <div className="w-64 border-r border-gray-800 p-4 shrink-0 overflow-y-auto">
        <h2 className="text-sm font-semibold mb-3">Training Config</h2>
        <div className="space-y-3">
          <div>
            <label className="text-[11px] text-gray-400 mb-1 block">Epochs</label>
            <div className="flex items-center gap-2">
              <input type="range" min={1} max={200} value={config.epochs} onChange={e => setConfig(p => ({ ...p, epochs: +e.target.value }))} className="flex-1 h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer accent-green-500" />
              <span className="text-[11px] text-gray-300 w-8 text-right font-mono">{config.epochs}</span>
            </div>
          </div>
          <div>
            <label className="text-[11px] text-gray-400 mb-1 block">Batch Size</label>
            <div className="flex items-center gap-2">
              <input type="range" min={4} max={128} step={4} value={config.batchSize} onChange={e => setConfig(p => ({ ...p, batchSize: +e.target.value }))} className="flex-1 h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer accent-green-500" />
              <span className="text-[11px] text-gray-300 w-8 text-right font-mono">{config.batchSize}</span>
            </div>
          </div>
          <div>
            <label className="text-[11px] text-gray-400 mb-1 block">Learning Rate</label>
            <select value={config.learningRate} onChange={e => setConfig(p => ({ ...p, learningRate: +e.target.value }))}
              className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-green-500">
              {[0.001, 0.0005, 0.0001, 0.00005, 0.00001].map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
          <div>
            <label className="text-[11px] text-gray-400 mb-1 block">Volumes / Class</label>
            <div className="flex items-center gap-2">
              <input type="range" min={5} max={200} step={5} value={config.numVolumes} onChange={e => setConfig(p => ({ ...p, numVolumes: +e.target.value }))} className="flex-1 h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer accent-green-500" />
              <span className="text-[11px] text-gray-300 w-8 text-right font-mono">{config.numVolumes}</span>
            </div>
          </div>
        </div>

        <p className="text-[11px] text-gray-500 mt-3 mb-3">CPU-only on macOS. 10 epochs ~1-5 min.</p>

        <Button onClick={handleStart} disabled={loading || isRunning}
          className="w-full bg-green-600 hover:bg-green-500 text-white text-sm">
          {isRunning ? 'Training...' : loading ? 'Starting...' : 'Start Training'}
        </Button>
        {error && <div className="mt-2 p-2 rounded bg-red-500/10 border border-red-500/20 text-red-400 text-xs">{error}</div>}
      </div>

      {/* Right Results */}
      <div className="flex-1 p-5 overflow-y-auto">
        {(isRunning || status?.status === 'completed') && (
          <div className="mb-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-gray-400">
                {status?.status === 'completed' ? 'Training Complete' : `Epoch ${status?.currentEpoch || 0} / ${status?.totalEpochs || config.epochs}`}
              </span>
              <span className="text-xs text-gray-500 font-mono">{isRunning && elapsed > 0 ? `${Math.floor(elapsed / 60)}m ${Math.floor(elapsed % 60)}s` : ''}</span>
            </div>
            <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
              <div className="h-full bg-green-500 rounded-full transition-all duration-500" style={{ width: `${progress}%` }} />
            </div>
          </div>
        )}

        {status && (isRunning || status.status === 'completed') && (
          <div className="grid grid-cols-4 gap-2 mb-3">
            {[
              { l: 'Train Loss', v: status.trainLoss?.toFixed(4) || '-', c: 'text-blue-400' },
              { l: 'Val Loss', v: status.valLoss?.toFixed(4) || '-', c: 'text-red-400' },
              { l: 'Train Acc', v: `${(status.trainAcc * 100).toFixed(1)}%`, c: 'text-emerald-400' },
              { l: 'Val Acc', v: `${(status.valAcc * 100).toFixed(1)}%`, c: 'text-yellow-400' },
            ].map(m => (
              <div key={m.l} className="p-2 bg-gray-800/50 rounded-lg border border-gray-700">
                <p className="text-[9px] text-gray-500 uppercase">{m.l}</p>
                <p className={`text-sm font-mono font-semibold ${m.c}`}>{m.v}</p>
              </div>
            ))}
          </div>
        )}

        {status?.status === 'completed' && <div className="mb-3 p-2 bg-green-500/10 border border-green-500/20 rounded-lg text-sm text-green-400">Done! Best val loss: {status.bestValLoss?.toFixed(4)}</div>}
        {status?.status === 'failed' && <div className="mb-3 p-2 bg-red-500/10 border border-red-500/20 rounded-lg text-sm text-red-400">Failed: {status.error}</div>}

        <div className="p-3 bg-gray-800/50 rounded-xl border border-gray-700 mb-3">
          <h3 className="text-xs font-semibold mb-2">Training Curves</h3>
          <canvas ref={canvasRef} className="w-full rounded-lg" style={{ height: 240 }} />
        </div>

        {(status?.epochHistory?.length ?? 0) > 0 && (
          <div className="p-3 bg-gray-800/50 rounded-xl border border-gray-700">
            <h3 className="text-xs font-semibold mb-2">Epoch Details</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-[11px]">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-700">
                    <th className="text-left py-1.5 px-2">Epoch</th>
                    <th className="text-right py-1.5 px-2">Train Loss</th>
                    <th className="text-right py-1.5 px-2">Val Loss</th>
                    <th className="text-right py-1.5 px-2">Train Acc</th>
                    <th className="text-right py-1.5 px-2">Val Acc</th>
                    <th className="text-right py-1.5 px-2">Train F1</th>
                    <th className="text-right py-1.5 px-2">Val F1</th>
                  </tr>
                </thead>
                <tbody>
                  {(status?.epochHistory ?? []).map((e: TrainEpochResult) => (
                    <tr key={e.epoch} className="border-b border-gray-800 hover:bg-gray-800/30">
                      <td className="py-1 px-2 font-mono text-gray-300">{e.epoch}</td>
                      <td className="py-1 px-2 text-right font-mono text-blue-400">{e.trainLoss.toFixed(4)}</td>
                      <td className="py-1 px-2 text-right font-mono text-red-400">{e.valLoss.toFixed(4)}</td>
                      <td className="py-1 px-2 text-right font-mono text-emerald-400">{(e.trainAcc * 100).toFixed(1)}%</td>
                      <td className="py-1 px-2 text-right font-mono text-yellow-400">{(e.valAcc * 100).toFixed(1)}%</td>
                      <td className="py-1 px-2 text-right font-mono text-gray-400">{(e.trainF1 * 100).toFixed(1)}%</td>
                      <td className="py-1 px-2 text-right font-mono text-gray-400">{(e.valF1 * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {!isRunning && (!status?.epochHistory || status.epochHistory.length === 0) && status?.status !== 'completed' && (
          <div className="flex items-center justify-center h-48">
            <div className="text-center">
              <div className="h-14 w-14 mx-auto mb-2 rounded-2xl bg-gray-800/50 border border-gray-700 flex items-center justify-center">
                <svg className="h-6 w-6 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
              </div>
              <p className="text-xs text-gray-500">Configure and click Start Training</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
