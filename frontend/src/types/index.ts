/**
 * Type Definitions Index
 *
 * Central export point for all CT Simulator type definitions.
 */

export * from './dicom';
export * from './simulation';
export * from './segmentation';
export * from './artifact';

/** Generic API response wrapper */
export interface APIResponse<T> {
  success: boolean;
  data: T;
  message?: string;
  error?: string;
}

/** Paginated API response */
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

/** Tool configuration for Cornerstone3D tools */
export interface ToolConfig {
  name: string;
  active: boolean;
  options: Record<string, unknown>;
}
