/**
 * Segmentation Module
 *
 * AI-powered organ and lesion segmentation interface.
 * Integrates with MONAI backend for:
 * - Automatic organ segmentation
 * - Lesion detection and segmentation
 * - Interactive segmentation refinement (brush tool)
 * - Label map editing and export
 */

// Re-export segmentation page components for easy imports
export { LabelToggleList } from '@segmentation/components/LabelToggleList';
export { JobProgressPanel } from '@segmentation/components/JobProgressPanel';
export { ExportDialog } from '@segmentation/components/ExportDialog';
export { BrushToolToggle } from '@segmentation/components/BrushToolToggle';
// Segmentation hooks
export { useSegmentationOverlay } from '@segmentation/hooks/useSegmentationOverlay';
export { useSegmentation } from '@segmentation/hooks/useSegmentation';
