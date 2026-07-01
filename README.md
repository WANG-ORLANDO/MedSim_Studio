# CT Simulator — 医学影像仿真平台

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Frontend](https://img.shields.io/badge/frontend-React%2BTypeScript-blue)](frontend/)
[![Backend](https://img.shields.io/badge/backend-FastAPI%2BPython-green)](backend/)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED)](docker-compose.yml)
[![Tests](https://img.shields.io/badge/tests-140%20passed-brightgreen)](tests/)

> **CT Medical Imaging Simulator** — 基于 Web 的医学影像平台，支持 DICOM/CT 影像加载、MPR 三视图重建、3D Volume Rendering、病灶仿真、AI 分割、**伪影生成/分类/修复完整流水线**。

---

## 功能概览

| 模块 | 功能 | 状态 |
|------|------|------|
| **DICOM 管理** | 上传、解析、浏览 CT 影像 | ✅ |
| **MPR 三视图** | Axial/Sagittal/Coronal 实时重建 | ✅ |
| **3D 体渲染** | vtk.js GPU 加速渲染 | ✅ |
| **病灶仿真** | 参数化病灶生成 + 器官模拟 | ✅ |
| **AI 分割** | MONAI U-Net 多器官分割 | ✅ |
| **伪影生成** | 7 种伪影类型（金属/噪声/运动/环/条状/射束硬化/组合） | ✅ |
| **伪影分类** | EfficientNet-B3 多标签分类（8 类） | ✅ |
| **伪影修复** | 传统方法 + RED-CNN 深度学习 + 混合策略 | ✅ |
| **质量评估** | PSNR/SSIM/NMSE/MAE 指标 | ✅ |

---

## 技术栈

| 层次 | 技术 |
|------|------|
| **前端** | React 18 + TypeScript + Vite + Cornerstone3D + vtk.js + Zustand + shadcn/ui + TailwindCSS |
| **后端** | FastAPI + Python 3.11 + SQLAlchemy + pydicom + NumPy/SciPy + scikit-image |
| **AI** | PyTorch + MONAI + timm (EfficientNet-B3) + albumentations |
| **数据库** | PostgreSQL 16 |
| **对象存储** | MinIO (S3 兼容) |
| **部署** | Docker + docker-compose (支持 CPU/GPU) |

---

## 项目结构

```
MedSim_Studio/
├── frontend/                          # React 前端
│   └── src/
│       ├── pages/                     # 页面组件
│       │   ├── ViewerPage.tsx         # DICOM 影像查看器
│       │   ├── StudiesPage.tsx        # 研究管理
│       │   ├── SimulationPage.tsx     # 病灶仿真 + CT Phantom
│       │   ├── SegmentationPage.tsx   # AI 分割
│       │   ├── ArtifactPage.tsx       # 伪影生成
│       │   └── ClassifierPage.tsx     # 伪影分类 + 训练
│       ├── services/                  # API 服务层
│       ├── store/                     # Zustand 状态管理
│       ├── types/                     # TypeScript 类型定义
│       └── components/                # 通用组件
│
├── backend/                           # FastAPI 后端
│   └── app/
│       ├── api/v1/                    # API 路由
│       │   ├── dicom.py               # DICOM 接口
│       │   ├── simulation.py          # 仿真接口
│       │   ├── segmentation.py        # 分割接口
│       │   └── artifact.py            # 伪影处理接口 (11 routes)
│       ├── artifact/                  # 伪影处理模块
│       │   ├── generator/             # 7 种伪影生成器
│       │   ├── classifier/            # EfficientNet-B3 分类器
│       │   ├── restoration/           # 传统 + 深度学习修复
│       │   └── evaluation/            # 质量评估指标
│       ├── simulation/                # 仿真引擎
│       ├── segmentation/              # 分割模块
│       ├── schemas/                   # Pydantic Schemas
│       └── models/                    # SQLAlchemy 模型
│
├── docker/                            # Docker 配置
│   ├── backend/
│   │   ├── Dockerfile                 # CPU 版本 (macOS)
│   │   └── Dockerfile.gpu             # GPU 版本 (Windows/Linux)
│   └── frontend/                      # Nginx + 构建
│
├── models/                            # 模型权重
│   ├── artifact_classifier/           # 分类器模型
│   ├── totalsegmentator/              # TotalSegmentator 权重
│   └── phantom_atlas/                 # Atlas 数据
│
├── tests/                             # 测试 (140 tests)
│   └── backend/artifact/              # 伪影模块测试
│
├── docker-compose.yml                 # 基础配置
├── docker-compose.gpu.yml             # GPU 覆盖配置
└── docs/                              # 文档
```

---

## 快速启动

### Docker 启动（推荐）

```bash
# macOS (CPU)
docker compose up -d

# Windows/Linux (GPU 加速)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

| 服务 | 地址 |
|------|------|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |
| MinIO Console | http://localhost:9001 |

### 本地开发

```bash
# Frontend
cd frontend && npm install && npm run dev

# Backend
cd backend && pip install -r requirements.txt
uvicorn app.main:app --port 8000 --reload
```

---

## 伪影处理流水线

### 生成 → 分类 → 修复 完整流程

```
                    ┌─────────────┐
                    │  CT Volume  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   生成伪影   │  7 种伪影类型
                    │  (Generator) │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   分类伪影   │  EfficientNet-B3
                    │ (Classifier) │  8 类多标签
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   修复伪影   │  混合策略
                    │ (Restorer)  │  自动选择方法
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  质量评估   │  PSNR/SSIM/NMSE/MAE
                    │  (Metrics)  │
                    └─────────────┘
```

### 支持的伪影类型

| 类型 | 生成方法 | 修复方法 |
|------|----------|----------|
| Metal (金属) | 5步管线 (掩码→条纹→射束硬化→噪声→HU) | sinogram MAR 插值 |
| Noise (噪声) | Poisson 光子模型 | RED-CNN 深度学习 |
| Motion (运动) | 位移场 + map_coordinates | 中值滤波 |
| Ring (环状) | Radon/iRadon sinogram 偏移 | sinogram 中值滤波 |
| Streak (条状) | Radon 角度域强度带 | sinogram 中值滤波 |
| Beam Hardening (射束硬化) | EDT 距离衰减 | 中值滤波 |
| Composite (组合) | 多伪影串联叠加 | 分步处理 |

---

## API 概览

### 伪影处理 API

| Method | Endpoint | 说明 |
|--------|----------|------|
| GET | `/api/v1/artifact/types` | 支持的伪影类型列表 |
| GET | `/api/v1/artifact/series` | CT 序列列表 |
| POST | `/api/v1/artifact/generate` | 生成伪影 |
| POST | `/api/v1/artifact/classify` | 伪影分类 |
| POST | `/api/v1/artifact/restore` | 伪影修复 |
| POST | `/api/v1/artifact/pipeline` | 完整流水线 (生成→分类→修复) |
| GET | `/api/v1/artifact/jobs` | 作业列表 |
| GET | `/api/v1/artifact/jobs/{id}` | 作业详情 |
| POST | `/api/v1/artifact/train` | 启动模型训练 |
| GET | `/api/v1/artifact/train/status` | 训练状态 |
| GET | `/api/v1/artifact/train/history` | 训练历史 |

### 其他 API

| Method | Endpoint | 说明 |
|--------|----------|------|
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/dicom/studies` | 研究列表 |
| POST | `/api/v1/dicom/upload` | 上传 DICOM |
| GET | `/api/v1/simulation/phantom` | CT Phantom 生成 |
| POST | `/api/v1/simulation/ct-params/preview` | CT 参数仿真 |
| POST | `/api/v1/segmentation/segment` | 执行分割 |

---

## 多设备部署

| 设备 | 硬件 | 用途 | 启动方式 |
|------|------|------|----------|
| macOS | Apple Silicon | 开发 + Web 服务 | `docker compose up -d` |
| Windows | Intel + NVIDIA GPU | 训练 + 推理 + Web | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` |

详见 [多设备部署指南](docs/多设备部署指南.md)

---

## 开发进度

### B 组 (伪影处理) — 140 tests passing

- [x] 伪影生成器 (7 种, 85 tests)
- [x] 伪影分类器 (EfficientNet-B3, 11 tests)
- [x] 传统修复方法 (4 种, 13 tests)
- [x] 深度学习修复 (RED-CNN, 10 tests)
- [x] 混合修复策略 (8 tests)
- [x] 质量评估模块 (11 tests)
- [x] 后端 API (11 routes)
- [x] 前端类型 + 服务层 + Store
- [ ] 模型训练 (代码完成, 待训练)
- [ ] 前端 UI 完善

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `POSTGRES_HOST` | postgres | 数据库主机 |
| `POSTGRES_DB` | ct_simulator | 数据库名 |
| `MINIO_HOST` | minio | 对象存储主机 |
| `AI_DEVICE` | cpu | AI 设备 (cpu/cuda) |
| `CLASSIFIER_MODEL_PATH` | /tmp/clf_full/best_model.pth | 分类器模型路径 |

---

## License

MIT License

## 致谢

- [3D Slicer](https://www.slicer.org/) — 医学影像分析平台启发
- [Cornerstone3D](https://www.cornerstonejs.org/) — 医学影像渲染
- [vtk.js](https://kitware.github.io/vtk-js/) — 可视化工具包
- [MONAI](https://monai.io/) — 医学影像 AI 框架
- [timm](https://github.com/huggingface/pytorch-image-models) — 预训练模型库
