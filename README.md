# CT Simulator — 医学影像仿真平台

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Frontend](https://img.shields.io/badge/frontend-React%2BTypeScript-blue)](frontend/)
[![Backend](https://img.shields.io/badge/backend-FastAPI%2BPython-green)](backend/)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED)](docker-compose.yml)

> **CT Medical Imaging Simulator** — 基于 Web 的医学影像平台，支持 DICOM/CT 影像加载、MPR 三视图重建、3D Volume Rendering、病灶仿真生成与 AI 分割。

 Inspired by 3D Slicer 的 Web 版本，使用现代 Web 技术栈实现医学影像的可视化与分析。

---

## 技术栈

### Frontend
| 技术 | 用途 |
|------|------|
| **React 18** | UI 框架 |
| **TypeScript** | 类型安全 |
| **Vite** | 构建工具 |
| **Cornerstone3D** | 医学影像 2D 渲染引擎 |
| **vtk.js** | 3D 体渲染引擎 |
| **Zustand** | 状态管理 |
| **shadcn/ui** | UI 组件库 |
| **TailwindCSS** | 样式系统 |
| **React Router** | 路由管理 |

### Backend
| 技术 | 用途 |
|------|------|
| **FastAPI** | RESTful API 框架 |
| **Python 3.11** | 运行环境 |
| **pydicom** | DICOM 文件解析 |
| **SimpleITK** | 医学图像处理 |
| **NumPy/SciPy** | 数值计算 |
| **VTK** | 体渲染管线 |
| **SQLAlchemy** | ORM 数据库 |
| **MinIO** | 对象存储 (S3) |

### AI
| 技术 | 用途 |
|------|------|
| **MONAI** | 医学影像 AI 框架 |
| **PyTorch** | 深度学习引擎 |

### Infrastructure
| 技术 | 用途 |
|------|------|
| **PostgreSQL** | 关系数据库 |
| **Docker** | 容器化部署 |
| **docker-compose** | 服务编排 |

---

## 项目结构

```
ct-simulator/
├── frontend/                          # React 前端
│   └── src/
│       ├── app/                       # 应用入口
│       ├── router/                    # 路由配置
│       ├── pages/                     # 页面组件
│       │   ├── ViewerPage.tsx         # 影像查看器
│       │   ├── StudiesPage.tsx        # 研究管理
│       │   ├── SimulationPage.tsx     # 病灶仿真
│       │   └── SegmentationPage.tsx   # AI 分割
│       ├── viewer/                    # Cornerstone3D 查看器
│       │   ├── cornerstone/           # Cornerstone3D 初始化与视口
│       │   ├── dicom/                 # DICOM 加载
│       │   ├── viewport/              # 视口管理
│       │   ├── tools/                 # 交互工具
│       │   ├── annotations/           # 标注系统
│       │   └── mpr/                   # MPR 三视图
│       ├── vtk/                       # vtk.js 3D 渲染
│       │   ├── volumeRendering/       # 体渲染
│       │   ├── mesh/                  # 网格处理
│       │   ├── marchingCubes/         # Marching Cubes
│       │   ├── clipping/              # 裁切
│       │   └── camera/                # 相机控制
│       ├── simulation/                # 仿真
│       │   ├── lesion/                # 病灶生成
│       │   ├── organ/                 # 器官模拟
│       │   ├── deformation/           # 形变场
│       │   ├── hu/                    # HU 值操作
│       │   └── generators/            # 生成器管线
│       ├── segmentation/              # AI 分割
│       ├── services/                  # API 服务层
│       ├── store/                     # Zustand 状态管理
│       ├── hooks/                     # React Hooks
│       ├── utils/                     # 工具函数
│       ├── types/                     # TypeScript 类型
│       └── components/                # 通用组件
│
├── backend/                           # FastAPI 后端
│   └── app/
│       ├── api/                       # API 路由
│       │   └── v1/                    # API v1
│       │       ├── health.py          # 健康检查
│       │       ├── dicom.py           # DICOM CRUD
│       │       ├── simulation.py      # 仿真接口
│       │       └── segmentation.py    # AI 分割接口
│       ├── core/                      # 核心配置
│       ├── dicom/                     # DICOM 处理
│       │   └── parser/                # DICOM 解析器
│       ├── simulation/                # 仿真引擎
│       │   ├── lesion/                # 病灶生成
│       │   ├── organ/                 # 器官模拟
│       │   ├── hu/                    # HU 修改器
│       │   └── deformation/           # 形变场
│       ├── segmentation/              # 分割模块
│       ├── rendering/                 # 渲染引擎
│       │   └── volume/                # 体渲染
│       ├── ai/                        # AI 模块
│       │   └── monai/                 # MONAI 集成
│       ├── schemas/                   # Pydantic Schemas
│       ├── models/                    # SQLAlchemy 模型
│       ├── database/                  # 数据库管理
│       └── utils/                     # 工具函数
│
├── ai-services/                       # AI 微服务
│   ├── monai/                         # MONAI 模型服务
│   ├── pytorch/                       # PyTorch 工具
│   └── models/                        # 预训练模型
│
├── datasets/                          # 数据集
│   ├── dicom/                         # DICOM 原始数据
│   └── annotations/                   # 标注数据
│
├── docker/                            # Docker 配置
│   ├── frontend/                      # Nginx + 构建
│   └── backend/                       # Python runtime
│
├── docs/                              # 文档
│   ├── api/                           # API 文档
│   ├── architecture/                  # 架构文档
│   └── modules/                       # 模块说明
│
├── scripts/                           # 工具脚本
├── tests/                             # 测试
│   ├── frontend/
│   ├── backend/
│   └── e2e/
├── .github/                           # GitHub 配置
│   ├── workflows/
│   └── ISSUE_TEMPLATE/
│
├── docker-compose.yml                 # Docker 编排
├── .env.example                       # 环境变量模板
├── .gitignore
└── README.md
```

---

## 快速启动

> **推荐**: 详见 [STARTUP.md](STARTUP.md) — 包含完整启动流程、常见问题排查和验证脚本。

### 前置条件

- Node.js 20+
- Python 3.11+
- Docker & docker-compose（可选）

### 本地开发

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

前端将在 http://localhost:5173 启动。

#### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API 将在 http://localhost:8000 启动。
Swagger 文档: http://localhost:8000/docs

### Docker 启动（推荐）

一键启动所有服务：

```bash
docker-compose up -d
```

| 服务 | 地址 |
|------|------|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |
| MinIO Console | http://localhost:9001 |

### 停止服务

```bash
docker-compose down
```

如需同时删除数据卷：

```bash
docker-compose down -v
```

---

## 架构设计

### 系统架构

```
┌─────────────┐     ┌──────────────┐     ┌──────────┐
│   Browser   │────▶│  Nginx (80)  │────▶│ FastAPI  │
│  (React)    │     │  / → SPA     │     │ :8000    │
│             │     │  /api → Back │     │          │
└─────────────┘     └──────────────┘     ─────┬─────┘
                                              │
                    ┌─────────────────────────┼─────────┐
                    │                         │         │
                    ▼                         ▼         ▼
              ┌──────────┐            ┌──────────┐ ┌──────┐
              │PostgreSQL│            │  MinIO   │ │Redis │
              │ :5432    │            │ :9000    │ │      │
              └──────────┘            └──────────┘ └──────┘
```

### 数据流

1. **DICOM Upload** → FastAPI → pydicom 解析 → PostgreSQL（元数据）+ MinIO（像素数据）
2. **Viewer** → Cornerstone3D 加载 → 2D MPR 渲染 / vtk.js 3D 体渲染
3. **Simulation** → 用户配置 → 病灶生成器 → CT 体数据 → DICOM/NIfTI 导出
4. **Segmentation** → MONAI 模型 → 分割推理 → 标签图 → 前端叠加显示

---

## 核心模块

### MPR（Multi-Planar Reconstruction）

- 实时三视图（Axial / Sagittal / Coronal）
- 十字线联动定位
- Window/Level 预设（Lung, Mediastinum, Bone, Brain 等）
- 同步缩放与平移

### 3D Volume Rendering

- vtk.js GPU 加速体渲染
- 颜色传递函数预设（CT Bone / Soft Tissue / Lung / Angio）
- 不透明度控制
- 裁切平面
- 相机预设视图

### 病灶仿真

- **支持类型**: Tumor, Nodule, Cyst, Calcification, Metastasis
- **形状**: Spherical, Ellipsoidal, Irregular, Lobulated, Spiculated
- **HU 控制**: 均值、标准差、边缘锐利度、内部钙化/坏死
- **输出格式**: DICOM, NIfTI, NRRD

### AI 分割

- MONAI U-Net 多器官分割
- 病灶检测与分割
- 交互式分割修正（进行中）
- 标签管理与导出

---

## 开发规划

### Phase 1 — 基础架构 （当前）
- [x] 项目结构搭建
- [x] Docker 容器化
- [x] 数据库模型
- [x] 基础 API 路由
- [ ] Cornerstone3D 集成
- [ ] vtk.js 集成

### Phase 2 — 核心功能
- [ ] DICOM 上传与解析
- [ ] MPR 三视图渲染
- [ ] Window/Level 交互
- [ ] 3D 体渲染
- [ ] 工具条（测量、标注）

### Phase 3 — 仿真引擎
- [ ] 病灶参数化生成
- [ ] 器官模拟
- [ ] HU 值修改器
- [ ] 形变场
- [ ] 仿真结果导出

### Phase 4 — AI 集成
- [ ] MONAI 模型加载
- [ ] 自动器官分割
- [ ] 病灶检测
- [ ] 交互式分割

### Phase 5 — 生产化
- [ ] 用户认证
- [ ] 负载均衡
- [ ] 监控告警
- [ ] 性能优化
- [ ] 临床验证

---

## API 概览

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/health/ready` | 就绪检查 |
| GET | `/api/v1/health/live` | 存活检查 |
| GET | `/api/v1/dicom/studies` | 研究列表 |
| GET | `/api/v1/dicom/studies/{id}` | 研究详情 |
| GET | `/api/v1/dicom/studies/{id}/series` | 序列列表 |
| POST | `/api/v1/dicom/upload` | 上传 DICOM |
| DELETE | `/api/v1/dicom/studies/{id}` | 删除研究 |
| POST | `/api/v1/simulation/jobs` | 创建仿真任务 |
| GET | `/api/v1/simulation/jobs` | 仿真任务列表 |
| GET | `/api/v1/simulation/jobs/{id}` | 任务详情 |
| POST | `/api/v1/simulation/preview/lesion` | 病灶预览 |
| POST | `/api/v1/segmentation/segment` | 执行分割 |

详细 API 文档请访问 `/docs`（Swagger UI）。

---

## 环境变量

参考 `.env.example` 文件，主要变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `POSTGRES_HOST` | postgres | 数据库主机 |
| `POSTGRES_DB` | ct_simulator | 数据库名 |
| `MINIO_HOST` | minio | 对象存储主机 |
| `AI_MONAI_ENABLED` | false | 是否启用 AI |
| `SIMULATION_DEFAULT_SEED` | 42 | 仿真随机种子 |

---

## License

MIT License

## 致谢

- [3D Slicer](https://www.slicer.org/) — 医学影像分析平台启发
- [OHIF Viewer](https://ohif.org/) — Web 医学影像查看器
- [Cornerstone3D](https://www.cornerstonejs.org/) — 医学影像渲染
- [vtk.js](https://kitware.github.io/vtk-js/) — 可视化工具包
- [MONAI](https://monai.io/) — 医学影像 AI 框架
