import { useState, useMemo } from 'react';
import type { SegmentationLabel } from '@/types/segmentation';

interface LabelToggleListProps {
  labels: SegmentationLabel[];
  enabledLabels: Set<number>;
  onToggle: (labelIndex: number) => void;
}

interface LabelGroupItem {
  catKey: string;
  catLabel: string;
  items: SegmentationLabel[];
}

interface GroupedResult {
  kind: 'categorized';
  groups: LabelGroupItem[];
}

interface FlatResult {
  kind: 'flat';
  items: SegmentationLabel[];
}

type LabelListResult = GroupedResult | FlatResult;

/**
 * LabelToggleList
 *
 * Displays segmentation labels with color swatches and toggle checkboxes.
 * Supports search/filter, category grouping (for TotalSegmentator's 117 labels),
 * and batch toggle (show all / hide all).
 */
export function LabelToggleList({
  labels,
  enabledLabels,
  onToggle,
}: LabelToggleListProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [collapsedCategories, setCollapsedCategories] = useState<Set<string>>(new Set());

  // Detect if labels have category info (TotalSegmentator mode)
  const hasCategories = useMemo(
    () => labels.some((l) => l.category && l.category_label),
    [labels],
  );

  // Group labels by category (if available) or flat list
  const grouped: LabelListResult = useMemo(() => {
    if (!hasCategories) {
      return { kind: 'flat' as const, items: labels.filter((l) => l.index > 0) };
    }

    const groups: LabelGroupItem[] = [];
    const catMap = new Map<string, LabelGroupItem>();

    for (const label of labels) {
      if (label.index === 0) continue;
      const cat = label.category || 'other';
      let group = catMap.get(cat);
      if (!group) {
        group = { catKey: cat, catLabel: label.category_label || cat, items: [] };
        catMap.set(cat, group);
        groups.push(group);
      }
      group.items.push(label);
    }

    return { kind: 'categorized' as const, groups };
  }, [labels, hasCategories]);

  // Filter by search query
  const filtered: LabelListResult = useMemo(() => {
    if (!searchQuery.trim()) return grouped;

    const q = searchQuery.toLowerCase();

    if (grouped.kind === 'categorized') {
      const filteredGroups = grouped.groups
        .map((g) => ({
          ...g,
          items: g.items.filter(
            (l) =>
              l.name.toLowerCase().includes(q) ||
              `#${l.index}`.includes(q) ||
              (l.category_label || '').toLowerCase().includes(q),
          ),
        }))
        .filter((g) => g.items.length > 0);

      return { kind: 'categorized', groups: filteredGroups };
    }

    // flat
    return {
      kind: 'flat',
      items: grouped.items.filter(
        (l) => l.name.toLowerCase().includes(q) || `#${l.index}`.includes(q),
      ),
    };
  }, [searchQuery, grouped]);

  // ---- Stats ----
  const totalCount = labels.filter((l) => l.index > 0).length;
  const enabledCount = enabledLabels.size;

  // ---- Batch handlers ----
  const handleShowAll = () => {
    for (const label of labels) {
      if (label.index > 0 && !enabledLabels.has(label.index)) {
        onToggle(label.index);
      }
    }
  };

  const handleHideAll = () => {
    for (const label of labels) {
      if (label.index > 0 && enabledLabels.has(label.index)) {
        onToggle(label.index);
      }
    }
  };

  // ---- Render helpers ----
  const renderLabelRow = (label: SegmentationLabel) => {
    const [r, g, b] = label.color;
    const hexColor = `rgb(${r}, ${g}, ${b})`;
    const isChecked = enabledLabels.has(label.index);

    return (
      <label
        key={label.index}
        className="flex cursor-pointer items-center gap-2 rounded px-2 py-0.5 text-[11px] hover:bg-accent"
      >
        <input
          type="checkbox"
          checked={isChecked}
          onChange={() => onToggle(label.index)}
          className="h-3 w-3 accent-primary shrink-0"
        />
        <span
          className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm border border-border"
          style={{ backgroundColor: hexColor }}
        />
        <span className="text-foreground truncate">{label.name}</span>
        <span className="ml-auto text-muted-foreground shrink-0">#{label.index}</span>
      </label>
    );
  };

  const toggleCategory = (catKey: string) => {
    setCollapsedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(catKey)) next.delete(catKey);
      else next.add(catKey);
      return next;
    });
  };

  // ---- Empty state ----
  if (labels.length === 0) {
    return (
      <div className="text-xs text-muted-foreground p-2">
        No segmentation labels available.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1">
      {/* Search + batch controls */}
      <div className="flex items-center gap-1 px-1">
        <input
          type="text"
          placeholder="Search labels..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="flex-1 rounded border border-border bg-background px-1.5 py-1 text-[11px] focus:outline-none focus:ring-1 focus:ring-primary"
        />
        <button
          onClick={handleShowAll}
          title="Show all"
          className="rounded px-1.5 py-1 text-[10px] text-muted-foreground hover:bg-accent hover:text-foreground"
        >
          All
        </button>
        <button
          onClick={handleHideAll}
          title="Hide all"
          className="rounded px-1.5 py-1 text-[10px] text-muted-foreground hover:bg-accent hover:text-foreground"
        >
          None
        </button>
      </div>

      {/* Counter */}
      <div className="px-1 text-[10px] text-muted-foreground">
        {enabledCount} / {totalCount} visible
      </div>

      {/* Label list */}
      <div className="max-h-[320px] overflow-y-auto space-y-0.5 px-0.5">
        {filtered.kind === 'categorized' ? (
          // ---- Categorized view (TotalSegmentator) ----
          filtered.groups.map(({ catKey, catLabel, items }) => {
            const isCollapsed = collapsedCategories.has(catKey);
            const groupEnabled = items.filter((l) =>
              enabledLabels.has(l.index),
            ).length;

            return (
              <div key={catKey} className="border-b border-border pb-1 last:border-0">
                {/* Category header */}
                <button
                  onClick={() => toggleCategory(catKey)}
                  className="flex w-full items-center gap-1 px-1 py-0.5 text-[11px] font-medium text-muted-foreground hover:text-foreground"
                >
                  <span className="text-[10px]">{isCollapsed ? '▶' : '▼'}</span>
                  <span className="truncate">{catLabel}</span>
                  <span className="ml-auto text-[10px]">
                    {groupEnabled}/{items.length}
                  </span>
                </button>

                {/* Category items */}
                {!isCollapsed && (
                  <div className="space-y-0 pl-3">
                    {items.map(renderLabelRow)}
                  </div>
                )}
              </div>
            );
          })
        ) : (
          // ---- Flat view (MONAI models) ----
          filtered.items.map(renderLabelRow)
        )}
      </div>
    </div>
  );
}
