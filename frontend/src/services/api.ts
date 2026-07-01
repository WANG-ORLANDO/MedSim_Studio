import axios, { type AxiosInstance, type AxiosRequestConfig, type AxiosResponse } from 'axios';

/**
 * API Service Layer
 *
 * Centralized HTTP client for all backend communications.
 * Backend endpoints return data directly (no { success, data } wrapper),
 * so the typed response is the raw data from the backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

// ---------------------------------------------------------------------------
// snake_case → camelCase converter for backend responses
// ---------------------------------------------------------------------------

function toCamel(str: string): string {
  return str.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
}

/** camelCase → snake_case converter for request bodies */
function toSnake(str: string): string {
  return str.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`);
}

function convertKeys(obj: unknown): unknown {
  if (Array.isArray(obj)) {
    return obj.map(convertKeys);
  }
  if (obj !== null && typeof obj === 'object' && !(obj instanceof Date)) {
    return Object.fromEntries(
      Object.entries(obj as Record<string, unknown>).map(([k, v]) => [
        toCamel(k),
        convertKeys(v),
      ]),
    );
  }
  return obj;
}

/** Convert object keys from camelCase to snake_case (for request bodies) */
function convertKeysToSnake(obj: unknown): unknown {
  if (Array.isArray(obj)) {
    return obj.map(convertKeysToSnake);
  }
  if (obj !== null && typeof obj === 'object' && !(obj instanceof Date)) {
    return Object.fromEntries(
      Object.entries(obj as Record<string, unknown>).map(([k, v]) => [
        toSnake(k),
        convertKeysToSnake(v),
      ]),
    );
  }
  return obj;
}

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor — converts camelCase request bodies to snake_case
    this.client.interceptors.request.use(
      (config) => {
        // Only convert JSON request bodies (not FormData, Blob, etc.)
        if (
          config.data &&
          config.headers['Content-Type'] === 'application/json' &&
          typeof config.data === 'object' &&
          !(config.data instanceof FormData) &&
          !(config.data instanceof Blob)
        ) {
          config.data = convertKeysToSnake(config.data) as any;
        }
        return config;
      },
      (error) => Promise.reject(error),
    );

    // Response interceptor — converts snake_case to camelCase + error normalization
    this.client.interceptors.response.use(
      (response) => {
        const isBlobResponse =
          response.config.responseType === 'blob' ||
          (typeof Blob !== 'undefined' && response.data instanceof Blob);

        if (!isBlobResponse && response.data !== null && response.data !== undefined) {
          response.data = convertKeys(response.data) as any;
        }
        return response;
      },
      (error) => {
        const normalized = {
          status: error.response?.status || 0,
          message: error.response?.data?.detail || error.message || 'Unknown error',
          data: error.response?.data || null,
        };
        return Promise.reject(normalized);
      },
    );
  }

  /** GET request — returns the backend response body directly */
  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.get<T>(url, config);
    return response.data;
  }

  /** POST request — returns the backend response body directly */
  async post<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.post<T>(url, data, config);
    return response.data;
  }

  /** PUT request — returns the backend response body directly */
  async put<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.put<T>(url, data, config);
    return response.data;
  }

  /** DELETE request — returns the backend response body directly */
  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.delete<T>(url, config);
    return response.data;
  }

  /** Download a file as blob — returns full AxiosResponse (data + headers) */
  async download(
    url: string,
    config?: AxiosRequestConfig,
  ): Promise<AxiosResponse> {
    return this.client.get(url, {
      ...config,
      responseType: 'blob',
    });
  }

  /** Upload DICOM files with multipart/form-data */
  async uploadDicom(files: File[], studyId?: string): Promise<{
    studyId: string;
    seriesCount: number;
    instanceCount: number;
    message: string;
  }> {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));

    return this.post('/dicom/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      params: studyId ? { study_id: studyId } : undefined,
      timeout: 120000,
    });
  }

  /** Generate artifact — longer timeout for radon-based generators */
  async generateArtifact<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    return this.post<T>(url, data, { ...config, timeout: 300000 });
  }
}

export const api = new ApiService();
