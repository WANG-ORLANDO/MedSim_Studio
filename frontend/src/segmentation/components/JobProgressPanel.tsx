import type { SegmentationJob } from '@/types/segmentation';

interface JobProgressPanelProps {
  job: SegmentationJob;
  onCancel?: (jobId: string) => void;
}

/**
 * JobProgressPanel
 *
 * Displays the current status and progress of a segmentation job.
 * Shows an animated progress bar during execution and a status
 * badge with appropriate coloring when finished.
 */
export function JobProgressPanel({ job, onCancel }: JobProgressPanelProps) {
  const { id, status, progress, errorMessage } = job;

  const statusColor = {
    pending: 'bg-yellow-500/20 text-yellow-600',
    running: 'bg-blue-500/20 text-blue-600',
    completed: 'bg-green-500/20 text-green-600',
    failed: 'bg-red-500/20 text-red-600',
  }[status];

  const statusLabel = {
    pending: 'Pending',
    running: 'Running',
    completed: 'Completed',
    failed: 'Failed',
  }[status];

  const isActive = status === 'pending' || status === 'running';

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`rounded px-2 py-0.5 text-xs font-medium ${statusColor}`}>
            {statusLabel}
          </span>
          <span className="font-mono text-xs text-muted-foreground">
            {id.substring(0, 8)}...
          </span>
        </div>

        {isActive && onCancel && (
          <button
            type="button"
            onClick={() => onCancel(id)}
            className="rounded px-2 py-1 text-xs text-red-500 hover:bg-red-500/10"
          >
            Cancel
          </button>
        )}
      </div>

      {/* Progress bar */}
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            status === 'failed'
              ? 'bg-red-500'
              : status === 'completed'
                ? 'bg-green-500'
                : 'bg-blue-500'
          }`}
          style={{ width: `${Math.min(progress, 100)}%` }}
        />
      </div>

      <div className="mt-1 flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          {isActive ? 'Processing...' : status === 'completed' ? 'Complete' : 'Failed'}
        </span>
        <span className="text-xs font-medium tabular-nums text-muted-foreground">
          {Math.round(progress)}%
        </span>
      </div>

      {status === 'failed' && errorMessage && (
        <div className="mt-2 rounded bg-red-500/10 px-2 py-1.5">
          <p className="text-xs text-red-600">{errorMessage}</p>
        </div>
      )}
    </div>
  );
}
