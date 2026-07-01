/// <reference types="vite/client" />

/**
 * Environment variable type definitions for Vite.
 * Provides type safety when accessing import.meta.env values.
 */
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_APP_TITLE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

/**
 * Type declarations for @cornerstonejs/dicom-image-loader
 * This package does not ship its own TypeScript declarations.
 */
declare module '@cornerstonejs/dicom-image-loader' {
  export const configure: (config: Record<string, any>) => void;
  export const external: { cornerstone: Record<string, any> };
}
