/**
 * Cornerstone3D Tool Groups
 *
 * Phase 5 of the Cornerstone3D rendering pipeline.
 * Manages ToolGroup creation and mouse/touch binding configuration.
 *
 * ToolGroups in Cornerstone3D:
 * - A ToolGroup defines which tools are available and their active bindings
 * - Multiple viewports can share the same ToolGroup
 * - Each tool has configurable mouse button bindings
 * - Active tool can be switched dynamically
 *
 * Mouse binding convention:
 * - Left button (Primary)   → WindowLevel (default active tool)
 * - Middle button (Middle)  → Pan
 * - Right button (Secondary) → Zoom
 * - Scroll wheel             → StackScroll (slice navigation)
 */

import {
  ToolGroupManager,
  WindowLevelTool,
  PanTool,
  ZoomTool,
  StackScrollTool,
  StackScrollMouseWheelTool,
  LengthTool,
  RectangleROITool,
  EllipticalROITool,
  ProbeTool,
  SegmentationDisplayTool,
  Enums as ToolEnums,
} from '@cornerstonejs/tools';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Default tool group identifier */
const DEFAULT_TOOL_GROUP_ID = 'ct-tool-group';

// ---------------------------------------------------------------------------
// Mouse button constants
// ---------------------------------------------------------------------------
const { MouseBindings } = ToolEnums;

// ---------------------------------------------------------------------------
// ToolGroup creation
// ---------------------------------------------------------------------------

/**
 * Create or retrieve the default CT tool group.
 *
 * This function sets up:
 * - WindowLevel tool on left mouse button (Primary) — default active
 * - Pan tool on middle mouse button (Middle)
 * - Zoom tool on right mouse button (Secondary)
 * - StackScroll tool on scroll wheel
 *
 * @param toolGroupId - Optional custom tool group ID
 * @returns The IToolGroup instance
 */
export function createToolGroup(
  toolGroupId: string = DEFAULT_TOOL_GROUP_ID
) {
  // Return existing group if already created
  const existing = ToolGroupManager.getToolGroup(toolGroupId);
  if (existing) {
    return existing;
  }

  // Create a new tool group
  const toolGroup = ToolGroupManager.createToolGroup(toolGroupId);
  if (!toolGroup) {
    throw new Error(`Failed to create tool group: ${toolGroupId}`);
  }

  // ------------------------------------------------------------------
  // Add tools with their mouse button bindings
  // ------------------------------------------------------------------

  // WindowLevel — left mouse button (default active tool)
  toolGroup.addTool(WindowLevelTool.toolName, {
    mouseButtonMask: MouseBindings.Primary,
  });

  // Pan — middle mouse button (numeric 4 = Middle button mask)
  toolGroup.addTool(PanTool.toolName, {
    mouseButtonMask: 4,
  });

  // Zoom — right mouse button
  toolGroup.addTool(ZoomTool.toolName, {
    mouseButtonMask: MouseBindings.Secondary,
  });

  // StackScroll — drag-based slice navigation (not wheel)
  toolGroup.addTool(StackScrollTool.toolName);

  // StackScrollMouseWheel — wheel-based slice scrolling for VolumeViewports
  toolGroup.addTool(StackScrollMouseWheelTool.toolName);

  // ------------------------------------------------------------------
  // Measurement tools (passive — placed on click, no drag binding)
  // ------------------------------------------------------------------
  toolGroup.addTool(LengthTool.toolName);
  toolGroup.addTool(RectangleROITool.toolName);
  toolGroup.addTool(EllipticalROITool.toolName);
  toolGroup.addTool(ProbeTool.toolName);

  // ------------------------------------------------------------------
  // Segmentation display tool (passive — renders overlays, no interaction)
  // ------------------------------------------------------------------
  toolGroup.addTool(SegmentationDisplayTool.toolName, {
    configuration: {
      // Re-render segmentation when the camera changes
      suppressToolConfig: false,
    },
  });

  // ------------------------------------------------------------------
  // Activate tools — each on its own mouse button
  // Multiple tools CAN be active simultaneously as long as they use
  // different mouse buttons or event sources (e.g., scroll wheel).
  // ------------------------------------------------------------------

  // WindowLevel — left mouse button (Primary)
  toolGroup.setToolActive(WindowLevelTool.toolName, {
    bindings: [{ mouseButton: MouseBindings.Primary }],
  });

  // Pan — middle mouse button (bitmask 4)
  toolGroup.setToolActive(PanTool.toolName, {
    bindings: [{ mouseButton: 4 }],
  });

  // Zoom — right mouse button (Secondary)
  toolGroup.setToolActive(ZoomTool.toolName, {
    bindings: [{ mouseButton: MouseBindings.Secondary }],
  });

  // StackScroll — drag-based slice navigation
  toolGroup.setToolActive(StackScrollTool.toolName);

  // StackScrollMouseWheel — wheel-based scrolling for VolumeViewports
  toolGroup.setToolActive(StackScrollMouseWheelTool.toolName);

  console.info(`[ToolGroup] Created: ${toolGroupId}`);
  return toolGroup;
}

// ---------------------------------------------------------------------------
// Viewport binding
// ---------------------------------------------------------------------------

/**
 * Add a viewport to the default tool group.
 *
 * Call this after enabling a viewport to make tools work on it.
 *
 * @param toolGroupId - The tool group identifier
 * @param viewportId - The viewport to add
 */
export function addViewportToToolGroup(
  toolGroupId: string = DEFAULT_TOOL_GROUP_ID,
  viewportId: string
): void {
  const toolGroup = ToolGroupManager.getToolGroup(toolGroupId);
  if (!toolGroup) {
    console.warn(
      `[ToolGroup] Cannot add viewport — group "${toolGroupId}" not found. ` +
        'Call createToolGroup() first.'
    );
    return;
  }

  // Only add viewport if not already in the group (avoids duplicate entries
  // when the cleanup skipped disableElement and the effect re-runs).
  const existingViewports = toolGroup.getViewportIds();
  if (!existingViewports?.includes(viewportId)) {
    toolGroup.addViewport(viewportId, 'ctViewerEngine');
    console.info(`[ToolGroup] Viewport "${viewportId}" added to group "${toolGroupId}"`);
  }

  // Re-apply tool modes so they take effect on the new viewport.
  // Each tool is active on its own mouse button — they coexist safely
  // because they don't conflict (different buttons / event sources).
  toolGroup.setToolActive(WindowLevelTool.toolName, {
    bindings: [{ mouseButton: MouseBindings.Primary }],
  });
  toolGroup.setToolActive(PanTool.toolName, {
    bindings: [{ mouseButton: 4 }],
  });
  toolGroup.setToolActive(ZoomTool.toolName, {
    bindings: [{ mouseButton: MouseBindings.Secondary }],
  });
  // StackScroll — drag-based slice navigation
  toolGroup.setToolActive(StackScrollTool.toolName);
  // StackScrollMouseWheel — wheel-based scrolling for VolumeViewports
  toolGroup.setToolActive(StackScrollMouseWheelTool.toolName);
}

// ---------------------------------------------------------------------------
// Tool activation
// ---------------------------------------------------------------------------

/**
 * Activate a tool in the tool group.
 *
 * Switches the active tool so the corresponding mouse binding takes effect.
 * Only one tool per mouse button can be active at a time.
 *
 * Supported tools: 'WindowLevel', 'Pan', 'Zoom', 'StackScroll'
 *
 * @param toolName - The tool to activate
 * @param toolGroupId - Optional custom tool group ID
 */
export function setActiveTool(
  toolName: string,
  toolGroupId: string = DEFAULT_TOOL_GROUP_ID
): void {
  const toolGroup = ToolGroupManager.getToolGroup(toolGroupId);
  if (!toolGroup) {
    console.warn(`[ToolGroup] Cannot set active tool — group "${toolGroupId}" not found`);
    return;
  }

  // First, set all tools passive
  toolGroup.setToolPassive(WindowLevelTool.toolName);
  toolGroup.setToolPassive(PanTool.toolName);
  toolGroup.setToolPassive(ZoomTool.toolName);
  toolGroup.setToolPassive(StackScrollTool.toolName);
  toolGroup.setToolPassive(StackScrollMouseWheelTool.toolName);
  toolGroup.setToolPassive(LengthTool.toolName);
  toolGroup.setToolPassive(RectangleROITool.toolName);
  toolGroup.setToolPassive(EllipticalROITool.toolName);
  toolGroup.setToolPassive(ProbeTool.toolName);
  toolGroup.setToolPassive(SegmentationDisplayTool.toolName);

  // Then activate the requested tool
  toolGroup.setToolActive(toolName as any, {
    bindings: [
      {
        mouseButton: MouseBindings.Primary,
      },
    ],
  });

  console.info(`[ToolGroup] Active tool: ${toolName}`);
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

/**
 * Destroy a tool group and release its resources.
 *
 * @param toolGroupId - The tool group to destroy
 */
export function destroyToolGroup(
  toolGroupId: string = DEFAULT_TOOL_GROUP_ID
): void {
  ToolGroupManager.destroyToolGroup(toolGroupId);
  console.info(`[ToolGroup] Destroyed: ${toolGroupId}`);
}
