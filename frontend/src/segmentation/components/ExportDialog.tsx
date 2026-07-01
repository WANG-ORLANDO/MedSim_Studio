import { useState } from 'react';
import type { ExportFormat } from '@/types/segmentation';
import { segmentationService } from '@/services/segmentationService';

interface ExportDialogProps {
  jobId: string;
  onClose: () => void;
}

const FORMAT_OPTIONS: { value: ExportFormat; label: string; description: string }[] = [
  { value: 'nrrd', label: 'NRRD', description: 'SimpleITK NRRD format (default)' },
  { value: 'nifti', label: 'NIfTI', description: 'NIfTI .nii.gz format' },
  { value: 'dicom_seg', label: 'DICOM SEG', description: 'DICOM Segmentation Object' },
];

/**
 * ExportDialog
 *
 * Modal dialog for exporting segmentation masks in various formats.
 * Displays format options and triggers download on selection.
 */
export function ExportDialog({ jobId, onClose }: ExportDialogProps) {
  const [selectedFormat, setSelectedFormat] = useState<ExportFormat>('nrrd');
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleExport = async () => {
    setExporting(true);
    setError(null);

    try {
      const response = await segmentationService.exportMask(jobId, selectedFormat);

      // Extract filename from Content-Disposition header
      const disposition = response.headers?.['content-disposition'] ?? '';
      const match = disposition.match(/filename="?(.+?)"?$/);
      const filename = match?.[1] ?? `segmentation_${jobId}.${selectedFormat}`;

      // Trigger browser download
      const url = URL.createObjectURL(response.data as Blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      onClose();
    } catch (err: any) {
      setError(err?.message ?? 'Export failed');
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-sm rounded-lg border border-border bg-card p-6 shadow-lg">
        <h3 className="mb-4 text-base font-semibold">Export Segmentation</h3>

        <div className="mb-4 space-y-2">
          {FORMAT_OPTIONS.map((opt) => (
            <label
              key={opt.value}
              className={`flex cursor-pointer items-start gap-3 rounded border p-3 text-sm transition-colors ${
                selectedFormat === opt.value
                  ? 'border-primary bg-primary/5'
                  : 'border-border hover:bg-accent'
              }`}
            >
              <input
                type="radio"
                name="format"
                value={opt.value}
                checked={selectedFormat === opt.value}
                onChange={() => setSelectedFormat(opt.value)}
                className="mt-0.5 h-3.5 w-3.5 accent-primary"
              />
              <div>
                <div className="font-medium text-foreground">{opt.label}</div>
                <div className="text-xs text-muted-foreground">{opt.description}</div>
              </div>
            </label>
          ))}
        </div>

        {error && (
          <div className="mb-3 rounded bg-red-500/10 px-3 py-2 text-xs text-red-600">
            {error}
          </div>
        )}

        <div className="flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={exporting}
            className="rounded px-3 py-1.5 text-sm text-muted-foreground hover:bg-accent disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleExport}
            disabled={exporting}
            className="rounded bg-primary px-3 py-1.5 text-sm text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            {exporting ? 'Exporting...' : 'Export'}
          </button>
        </div>
      </div>
    </div>
  );
}
