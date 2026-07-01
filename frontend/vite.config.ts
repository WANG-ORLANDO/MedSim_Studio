import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// ---------------------------------------------------------------------------
// Vite Configuration
// ---------------------------------------------------------------------------
// CT Simulator frontend build configuration with path aliases for modular
// architecture support.
// ---------------------------------------------------------------------------
export default defineConfig({
  assetsInclude: ['**/*.wasm'],
  plugins: [
    react(),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@app': path.resolve(__dirname, './src/app'),
      '@pages': path.resolve(__dirname, './src/pages'),
      '@viewer': path.resolve(__dirname, './src/viewer'),
      '@vtk': path.resolve(__dirname, './src/vtk'),
      '@simulation': path.resolve(__dirname, './src/simulation'),
      '@segmentation': path.resolve(__dirname, './src/segmentation'),
      '@services': path.resolve(__dirname, './src/services'),
      '@store': path.resolve(__dirname, './src/store'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@utils': path.resolve(__dirname, './src/utils'),
      '@types': path.resolve(__dirname, './src/types'),
      '@components': path.resolve(__dirname, './src/components'),
    },
  },
  optimizeDeps: {
    exclude: [],
    include: [
      '@cornerstonejs/core',
      '@cornerstonejs/tools',
      '@cornerstonejs/dicom-image-loader',
      '@kitware/vtk.js',
      'lodash.clonedeep',
    ],
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    // Cornerstone3D and vtk.js require larger chunks
    chunkSizeWarningLimit: 2000,
    commonjsOptions: {
      exclude: [/@icr\/polyseg-wasm/],
      requireReturnsDefault: 'auto',
    },
    rollupOptions: {
      output: {
        manualChunks: {
          cornerstone: ['@cornerstonejs/core', '@cornerstonejs/tools'],
          vtk: ['@kitware/vtk.js'],
          react: ['react', 'react-dom', 'react-router-dom'],
        },
      },
    },
  },
});
