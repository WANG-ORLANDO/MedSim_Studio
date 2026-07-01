/**
 * Toolbar Component
 *
 * Medical imaging tool selection bar for Cornerstone3D tools.
 * Provides toggleable tools for:
 * - Window/Level adjustment
 * - Pan
 * - Zoom
 * - Length measurement
 * - ROI annotation (rectangle, ellipse)
 * - Crosshair (MPR linking)
 */

export type ToolType = 'windowLevel' | 'pan' | 'zoom' | 'length' | 'rectangleRoi' | 'ellipticalRoi' | 'crosshair';

interface ToolbarProps {
  activeTool: ToolType;
  onToolChange: (tool: ToolType) => void;
}

const tools: { id: ToolType; label: string; shortcut: string }[] = [
  { id: 'windowLevel', label: 'W/L', shortcut: 'W' },
  { id: 'pan', label: 'Pan', shortcut: 'P' },
  { id: 'zoom', label: 'Zoom', shortcut: 'Z' },
  { id: 'length', label: 'Meas', shortcut: 'M' },
  { id: 'rectangleRoi', label: 'Rect', shortcut: 'R' },
  { id: 'ellipticalRoi', label: 'Ellipse', shortcut: 'E' },
  { id: 'crosshair', label: 'Cross', shortcut: 'C' },
];

export function Toolbar({ activeTool, onToolChange }: ToolbarProps) {
  return (
    <div className="flex items-center gap-1 rounded-lg bg-card border border-border p-1">
      {tools.map((tool) => (
        <button
          key={tool.id}
          onClick={() => onToolChange(tool.id)}
          title={`${tool.label} (${tool.shortcut})`}
          className={`rounded px-2 py-1 text-xs font-medium transition-colors ${
            activeTool === tool.id
              ? 'bg-primary text-primary-foreground'
              : 'text-muted-foreground hover:text-foreground'
          }`}
        >
          {tool.label}
        </button>
      ))}
    </div>
  );
}
