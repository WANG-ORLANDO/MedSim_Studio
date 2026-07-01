import { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from '@components/ui/button';
import { artifactService, type TrainStatusResponse, type TrainEpochResult, type TrainHistoryResponse } from '@/services/artifactService';

export default function TrainPage() {
  const [config, setConfig] = useState({
    epochs: 10,
    batchSize: 32,
    learningRate: 0.0001,
    numVolumes: 20,
    outputDir: '/app/models/artifact_classifier',
  });
  const [status, setStatus] = useState<TrainStatusResponse | null>(null);
  const [history, setHistory] = useState<TrainHistoryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Poll training status
  const pollStatus = useCallback(async () => {
    try {
      const s = await artifactService.getTrainStatus();
      setStatus(s);
      if (s.status === 'training' || s.status === 'starting') {
        if (s.startTime) setStartTime(s.startTime);
      }
      if (s.status === 'completed' || s.status === 'failed') {
        setLoading(false);
        // Reload history
        const h = await artifactService.getTrainHistory();
        setHistory(h);
      }
    } catch {
      // ignore
    }
  }, []);

  // Elapsed time ticker
  useEffect(() => {
    if (status?.status === 'training' && startTime) {
      const tick = () => setElapsed(Date.now() / 1000 - startTime);
      tick();
      const id = setInterval(tick, 1000);
      return () => clearInterval(id);
    }
  }, [status?.status, startTime]);

  // Auto-poll during training
  useEffect(() => {
    if (status?.status === 'training' || status?.status === 'starting') {
      intervalRef.current = setInterval(pollStatus, 3000);
      return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
    }
  }, [status?.status, pollStatus]);

  // Load history on mount
  useEffect(() => {
    artifactService.getTrainHistory().then(setHistory).catch(() => {});
    pollStatus();
  }, [pollStatus]);

  // Draw training curves
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const epochs = status?.epochHistory?.length ? status.epochHistory : history?.epochs ?? [];
    if (epochs.length === 0) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        canvas.width = canvas.offsetWidth;
        canvas.height = 280;
        ctx.fillStyle = '#1f2937';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#6b7280';
        ctx.font = '13px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No training data yet', canvas.width / 2, 140);
      }
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = canvas.offsetWidth;
    canvas.height = 280;
    const W = canvas.width;
    const H = canvas.height;
    const pad = { top: 30, right: 60, bottom: 30, left: 50 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    ctx.fillStyle = '#1f2937';
    ctx.fillRect(0, 0, W, H);

    // Collect all values
    const allLoss = epochs.flatMap(e => [e.trainLoss, e.valLoss]);
    const allAcc = epochs.flatMap(e => [e.trainAcc, e.valAcc]);
    const maxLoss = Math.max(...allLoss, 0.01) * 1.1;
    const maxAcc = Math.min(Math.max(...allAcc, 0.5), 1.0) * 1.05;
    const minAcc = Math.min(...allAcc, 0);
    const n = epochs.length;

    // Grid lines
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH / 4) * i;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    }

    // X axis labels
    ctx.fillStyle = '#9ca3af';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    const step = Math.max(1, Math.floor(n / 8));
    for (let i = 0; i < n; i += step) {
      const x = pad.left + (i / (n - 1 || 1)) * plotW;
      ctx.fillText(String(epochs[i].epoch), x, H - 8);
    }

    // Draw curves
    function drawLine(data: number[], color: string, maxVal: number, minVal = 0) {
      const c = ctx!;
      c.strokeStyle = color;
      c.lineWidth = 2;
      c.beginPath();
      data.forEach((v, i) => {
        const x = pad.left + (i / (n - 1 || 1)) * plotW;
        const y = pad.top + plotH - ((v - minVal) / (maxVal - minVal || 1)) * plotH;
        i === 0 ? c.moveTo(x, y) : c.lineTo(x, y);
      });
      c.stroke();
    }

    // Loss (left axis)
    drawLine(epochs.map(e => e.trainLoss), '#60a5fa', maxLoss);
    drawLine(epochs.map(e => e.valLoss), '#f87171', maxLoss);

    // Acc (right axis)
    drawLine(epochs.map(e => e.trainAcc), '#34d399', maxAcc, minAcc);
    drawLine(epochs.map(e => e.valAcc), '#fbbf24', maxAcc, minAcc);

    // Y axis labels (loss)
    ctx!.fillStyle = '#60a5fa';
    ctx!.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const val = (maxLoss / 4) * (4 - i);
      const y = pad.top + (plotH / 4) * i;
      ctx!.fillText(val.toFixed(3), pad.left - 5, y + 3);
    }

    // Y axis labels (acc) — right side
    ctx!.fillStyle = '#34d399';
    ctx!.textAlign = 'left';
    for (let i = 0; i <= 4; i++) {
      const val = minAcc + ((maxAcc - minAcc) / 4) * (4 - i);
      const y = pad.top + (plotH / 4) * i;
      ctx!.fillText(val.toFixed(2), W - pad.right + 5, y + 3);
    }

    // Legend
    ctx!.font = '11px sans-serif';
    const legends = [
      { label: 'Train Loss', color: '#60a5fa' },
      { label: 'Val Loss', color: '#f87171' },
      { label: 'Train Acc', color: '#34d399' },
      { label: 'Val Acc', color: '#fbbf24' },
    ];
    let lx = pad.left + 10;
    legends.forEach(l => {
      ctx!.fillStyle = l.color;
      ctx!.fillRect(lx, 8, 12, 10);
      ctx!.fillStyle = '#d1d5db';
      ctx!.textAlign = 'left';
      ctx!.fillText(l.label, lx + 16, 17);
      lx += ctx!.measureText(l.label).width + 30;
    });
  }, [status?.epochHistory, history]);

  const handleStart = useCallback(async () => {
    setLoading(true);
    setError(null);
    setStartTime(Date.now() / 1000);
    try {
      await artifactService.startTraining(config);
      await pollStatus();
    } catch (e: any) {
      setError(e.message || 'Failed to start training');
      setLoading(false);
    }
  }, [config, pollStatus]);

  const isRunning = status?.status === 'training' || status?.status === 'starting';
  const progress = status && status.totalEpochs > 0 ? (status.currentEpoch / status.totalEpochs) * 100 : 0;

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-gray-100 flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-3 flex items-center gap-4 shrink-0">
        <div className="h-8 w-8 rounded-lg bg-green-500/20 flex items-center justify-center">
          <svg className="h-5 w-5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        <h1 className="text-lg font-semibold">Model Training</h1>
        <span className="text-xs text-gray-500">Train EfficientNet-B3 artifact classifier</span>
      </header>

      <div className="flex flex-1 min-h-0">
        {/* Left Panel — Config */}
        <div className="w-72 border-r border-gray-800 p-4 overflow-y-auto shrink-0">
          <h2 className="text-sm font-semibold mb-3">Training Config</h2>

          <div className="space-y-3">
            {/* Epochs */}
            <div>
              <label className="text-[11px] text-gray-400 mb-1 block">Epochs</label>
              <div className="flex items-center gap-2">
                <input type="range" min={1} max={200} step={1} value={config.epochs}
                  onChange={(e) => setConfig(p => ({ ...p, epochs: parseInt(e.target.value) }))}
                  className="flex-1 h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer accent-green-500" />
                <span className="text-[11px] text-gray-300 w-10 text-right font-mono">{config.epochs}</span>
              </div>
            </div>

            {/* Batch Size */}
            <div>
              <label className="text-[11px] text-gray-400 mb-1 block">Batch Size</label>
              <div className="flex items-center gap-2">
                <input type="range" min={4} max={128} step={4} value={config.batchSize}
                  onChange={(e) => setConfig(p => ({ ...p, batchSize: parseInt(e.target.value) }))}
                  className="flex-1 h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer accent-green-500" />
                <span className="text-[11px] text-gray-300 w-10 text-right font-mono">{config.batchSize}</span>
              </div>
            </div>

            {/* Learning Rate */}
            <div>
              <label className="text-[11px] text-gray-400 mb-1 block">Learning Rate</label>
              <select value={config.learningRate}
                onChange={(e) => setConfig(p => ({ ...p, learningRate: parseFloat(e.target.value) }))}
                className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1.5 text-xs text-gray-200 focus:outline-none focus:border-green-500">
                <option value={0.001}>1e-3</option>
                <option value={0.0005}>5e-4</option>
                <option value={0.0001}>1e-4</option>
                <option value={0.00005}>5e-5</option>
                <option value={0.00001}>1e-5</option>
              </select>
            </div>

            {/* Volumes per class */}
            <div>
              <label className="text-[11px] text-gray-400 mb-1 block">Volumes / Class</label>
              <div className="flex items-center gap-2">
                <input type="range" min={5} max={200} step={5} value={config.numVolumes}
                  onChange={(e) => setConfig(p => ({ ...p, numVolumes: parseInt(e.target.value) }))}
                  className="flex-1 h-1.5 bg-gray-700 rounded-full appearance-none cursor-pointer accent-green-500" />
                <span className="text-[11px] text-gray-300 w-10 text-right font-mono">{config.numVolumes}</span>
              </div>
            </div>

            {/* Output Dir */}
            <div>
              <label className="text-[11px] text-gray-400 mb-1 block">Output Directory</label>
              <input value={config.outputDir}
                onChange={(e) => setConfig(p => ({ ...p, outputDir: e.target.value }))}
                className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1.5 text-xs text-gray-200 font-mono focus:outline-none focus:border-green-500" />
            </div>
          </div>

          {/* Model Info */}
          <div className="mt-4 p-3 bg-gray-800/50 rounded-lg border border-gray-700">
            <p className="text-[11px] text-gray-400 mb-1">Architecture: EfficientNet-B3</p>
            <p className="text-[11px] text-gray-400 mb-1">Classes: 8 artifact types</p>
            <p className="text-[11px] text-gray-500">Training uses synthetic data from artifact generators. CPU-only on macOS.</p>
          </div>

          <Button onClick={handleStart} disabled={loading || isRunning}
            className="w-full mt-4 bg-green-600 hover:bg-green-500 text-white text-sm">
            {isRunning ? 'Training...' : loading ? 'Starting...' : 'Start Training'}
          </Button>
          {error && <div className="mt-2 p-2 rounded bg-red-500/10 border border-red-500/20 text-red-400 text-xs">{error}</div>}
        </div>

        {/* Right Panel — Results */}
        <div className="flex-1 p-6 overflow-y-auto">
          {/* Progress bar */}
          {(isRunning || status?.status === 'completed') && (
            <div className="mb-4">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-gray-400">
                  {status?.status === 'completed' ? 'Training Complete' : `Epoch ${status?.currentEpoch || 0} / ${status?.totalEpochs || config.epochs}`}
                </span>
                <span className="text-xs text-gray-500 font-mono">
                  {isRunning && elapsed > 0 ? `${Math.floor(elapsed / 60)}m ${Math.floor(elapsed % 60)}s` : ''}
                </span>
              </div>
              <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-green-500 rounded-full transition-all duration-500" style={{ width: `${progress}%` }} />
              </div>
            </div>
          )}

          {/* Live Metrics */}
          {status && (isRunning || status.status === 'completed') && (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
              {[
                { label: 'Train Loss', value: status.trainLoss?.toFixed(4) || '-', color: 'text-blue-400' },
                { label: 'Val Loss', value: status.valLoss?.toFixed(4) || '-', color: 'text-red-400' },
                { label: 'Train Acc', value: `${(status.trainAcc * 100).toFixed(1)}%` || '-', color: 'text-emerald-400' },
                { label: 'Val Acc', value: `${(status.valAcc * 100).toFixed(1)}%` || '-', color: 'text-yellow-400' },
              ].map(m => (
                <div key={m.label} className="p-3 bg-gray-800/50 rounded-lg border border-gray-700">
                  <p className="text-[10px] text-gray-500 uppercase tracking-wider">{m.label}</p>
                  <p className={`text-lg font-mono font-semibold ${m.color}`}>{m.value}</p>
                </div>
              ))}
            </div>
          )}

          {status?.status === 'completed' && (
            <div className="mb-4 p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
              <p className="text-sm text-green-400">
                Training completed! Best val loss: {status.bestValLoss?.toFixed(4)}
              </p>
            </div>
          )}

          {status?.status === 'failed' && (
            <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
              <p className="text-sm text-red-400">Training failed: {status.error}</p>
            </div>
          )}

          {/* Chart */}
          <div className="p-4 bg-gray-800/50 rounded-xl border border-gray-700">
            <h3 className="text-sm font-semibold mb-3">Training Curves</h3>
            <canvas ref={canvasRef} className="w-full rounded-lg" style={{ height: 280 }} />
          </div>

          {/* Epoch Table */}
          {(status?.epochHistory?.length ?? 0) > 0 && (
            <div className="mt-4 p-4 bg-gray-800/50 rounded-xl border border-gray-700">
              <h3 className="text-sm font-semibold mb-3">Epoch Details</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-gray-500 border-b border-gray-700">
                      <th className="text-left py-2 px-2">Epoch</th>
                      <th className="text-right py-2 px-2">Train Loss</th>
                      <th className="text-right py-2 px-2">Val Loss</th>
                      <th className="text-right py-2 px-2">Train Acc</th>
                      <th className="text-right py-2 px-2">Val Acc</th>
                      <th className="text-right py-2 px-2">Train F1</th>
                      <th className="text-right py-2 px-2">Val F1</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(status?.epochHistory ?? []).map((e: TrainEpochResult) => (
                      <tr key={e.epoch} className="border-b border-gray-800 hover:bg-gray-800/30">
                        <td className="py-1.5 px-2 font-mono text-gray-300">{e.epoch}</td>
                        <td className="py-1.5 px-2 text-right font-mono text-blue-400">{e.trainLoss.toFixed(4)}</td>
                        <td className="py-1.5 px-2 text-right font-mono text-red-400">{e.valLoss.toFixed(4)}</td>
                        <td className="py-1.5 px-2 text-right font-mono text-emerald-400">{(e.trainAcc * 100).toFixed(1)}%</td>
                        <td className="py-1.5 px-2 text-right font-mono text-yellow-400">{(e.valAcc * 100).toFixed(1)}%</td>
                        <td className="py-1.5 px-2 text-right font-mono text-gray-400">{(e.trainF1 * 100).toFixed(1)}%</td>
                        <td className="py-1.5 px-2 text-right font-mono text-gray-400">{(e.valF1 * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Empty State */}
          {!isRunning && (!status?.epochHistory || status.epochHistory.length === 0) && status?.status !== 'completed' && (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <div className="h-16 w-16 mx-auto mb-3 rounded-2xl bg-gray-800/50 border border-gray-700 flex items-center justify-center">
                  <svg className="h-7 w-7 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <p className="text-sm text-gray-500">Configure and click Start Training</p>
                <p className="text-xs text-gray-600 mt-1">Training runs on CPU (~1-5 min for 10 epochs)</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
