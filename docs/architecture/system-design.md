# System Architecture

## Overview

The CT Simulator follows a microservices architecture with clear separation of concerns:

- **Frontend**: Single-page application with module-based code splitting
- **Backend**: RESTful API service with domain-driven modules
- **AI Services**: Isolated model serving (future)

## Module Dependency

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React SPA)                     │
│  ┌────────┐ ┌───────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ Viewer │ │ Simulation│ │Segment-  │ │ Study Manager │  │
│  │ Module │ │ Module    │ │ation Mod │ │ Module        │  │
│  └────┬───┘ └─────┬─────┘ └─────┬────┘ └───────┬───────┘  │
│       └───────────┴─────────────┴──────────────┘           │
│                         │ API Layer                        │
└─────────────────────────┼───────────────────────────────────┘
                          │ REST
┌─────────────────────────┼───────────────────────────────────┐
│                   Backend (FastAPI)                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ DICOM    │ │Simulation│ │ Segment- │ │ Rendering    │  │
│  │ Service  │ │ Engine   │ │ ation    │ │ Engine       │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘  │
│       └────────────┴────────────┴──────────────┘           │
│                         │ Data Layer                       │
└─────────────────────────┼───────────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
         PostgreSQL    MinIO       File System
         (Metadata)   (Pixels)    (Cache)
```

## Data Flow

1. **Upload**: DICOM files → Backend → pydicom parse → PostgreSQL (metadata) + MinIO (pixel data)
2. **View**: Frontend → Cornerstone3D loads via wado URI → Renders 2D MPR
3. **3D**: Frontend → vtk.js loads volume → GPU ray casting → Interactive 3D
4. **Simulate**: Frontend config → Backend generator → NumPy volume → Export to DICOM/NIfTI
5. **Segment**: Frontend request → Backend MONAI → PyTorch inference → Label map → Frontend overlay
