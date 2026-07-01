interface BrushToolToggleProps {
  active: boolean;
  onToggle: () => void;
  disabled?: boolean;
}

/**
 * BrushToolToggle
 *
 * Toggle button to enable/disable brush editing mode for
 * interactive segmentation refinement on viewport clicks.
 */
export function BrushToolToggle({
  active,
  onToggle,
  disabled = false,
}: BrushToolToggleProps) {
  return (
    <button
      type="button"
      onClick={onToggle}
      disabled={disabled}
      className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
        active
          ? 'bg-primary text-primary-foreground shadow-sm'
          : 'border border-border bg-transparent text-muted-foreground hover:bg-accent'
      } disabled:opacity-50 disabled:cursor-not-allowed`}
    >
      <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
        <path d="M12 2L2 7l10 5 10-5-10-5z" />
        <path d="M2 17l10 5 10-5" />
        <path d="M2 12l10 5 10-5" />
      </svg>
      {active ? 'Brush: ON' : 'Brush: OFF'}
    </button>
  );
}
