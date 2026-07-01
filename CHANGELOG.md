# CHANGELOG

## v0.1.1 — 2026-05-28

### Cornerstone3D 初始化修复

**问题**: 点击已上传的 DICOM 研究时，Viewer 加载失败，报错：
- `qI.wadouri.registerImageLoader is not a function`（初始化阶段）
- `C2.triggerEvent is not a function` / `Cannot destructure property 'MetadataModules' of 'Q2.Enums'`（运行时）

**根因**: 
1. 旧代码使用 `import * as cs from '@cornerstonejs/core'` 然后通过 `wadouri.register(cs)` 注册，再通过 `(dicomImageLoaderExternal).cornerstone = cs` 传递模块。这种方式在 Vite/Rollup 打包后，命名空间对象上的 `registerImageLoader` 丢失。
2. 第一次修复只传了 `{ registerImageLoader, metaData }` 对象字面量，缺少 `triggerEvent`、`Enums` 等运行时 API。

**修复** (`frontend/src/viewer/cornerstone/initCornerstone.ts`):
- 使用 `import * as cs from '@cornerstonejs/core'` 导入完整命名空间
- 通过 `(dicomImageLoaderExternal).cornerstone = cs` 传递整个模块
- 移除 `wadouri.register()` 调用（setter 已自动注册）
- 确保 dicom-image-loader 在运行时能访问到所有 Core API

### 本地开发环境支持

**问题**: 数据库表不存在、PostgreSQL 依赖过重，无法在无 Docker 环境下运行。

**改动**:

| 文件 | 改动 |
|------|------|
| `backend/app/core/config.py` | `DATABASE_URL` 默认值改为 `sqlite:///./ct_simulator.db` |
| `backend/app/database/session.py` | 添加 SQLite 条件判断（`connect_args`、`pool_pre_ping`） |
| `backend/app/main.py` | 在 `lifespan` 启动时执行 `Base.metadata.create_all(bind=engine)` 自动建表 |

- 本地运行使用 SQLite，Docker 部署通过 `DATABASE_URL` 环境变量自动切换 PostgreSQL
- 本地启动：`cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`

### Docker 构建修复

**问题**: 前端 Dockerfile 使用 `yarn`，但项目使用 `npm`（无 `yarn.lock`，只有 `package-lock.json`）；`package-lock.json` 在 Windows 生成后，Linux 容器内缺少 Rollup 可选依赖导致构建失败。

**改动** (`docker/frontend/Dockerfile`):
- 构建阶段基础镜像从 `node:20-alpine` 改为 `node:20-slim`（修复 musl/glibc 引起的 Rollup 原生模块缺失问题）
- 包管理器从 `yarn` 改为 `npm`：`npm install` 替代 `yarn install`
- 不再复制 `package-lock.json`（Windows 生成，内含 `@rollup/rollup-win32-x64-msvc`，Linux 下需要 `-gnu` 版本）

**端口变更** (`docker-compose.yml`):
- 前端端口从 `3000:80` 改为 `5173:80`

**.dockerignore**:
- 新增 `datasets/`、`.venv/`、`*.db` 排除规则

### 已验证功能

| 功能 | 状态 |
|------|------|
| DICOM 文件上传（134 个实例） | ✓ |
| 研究列表查询 | ✓ |
| Cornerstone3D Viewer 渲染 | ✓ |
| Docker 全栈部署（frontend + backend + postgres + minio） | ✓ |
| SQLite 本地开发 | ✓ |
| PostgreSQL Docker 部署 | ✓ |

### 架构说明

```
ct-simulator/
├── frontend/          # React + Vite + Cornerstone3D
│   ├── src/viewer/cornerstone/initCornerstone.ts  ← 3D 渲染初始化
│   └── ...
├── backend/           # FastAPI + SQLAlchemy
│   ├── app/
│   │   ├── core/config.py        ← 配置管理
│   │   ├── database/session.py   ← 数据库连接
│   │   └── main.py               ← 应用入口
│   └── ...
├── docker/
│   ├── frontend/Dockerfile
│   └── backend/Dockerfile
├── docker-compose.yml
└── .dockerignore
```

**数据库策略**:
- 本地开发：SQLite（`sqlite:///./ct_simulator.db`）
- Docker 部署：PostgreSQL（通过 `DATABASE_URL` 环境变量覆盖）
- 表自动创建：应用启动时 `Base.metadata.create_all()`

**Docker 部署**:
```bash
docker compose build
docker compose up -d
# 访问 http://localhost:5173
```

## v0.1.2 — 2026-05-28

### 文件夹上传支持

**问题**: 上传 DICOM 时只能选单个文件，无法直接选择整个文件夹。

**改动** (`frontend/src/pages/StudiesPage.tsx`):
- 新增 **Upload Folder** 按钮，使用 `webkitdirectory` 属性支持整个文件夹递归选择
- 后端 `os.path.basename()` 处理浏览器传入的相对路径，平铺存储到 staging 目录

### 重复 SOP UID 处理

**问题**: 数据集子目录间存在相同 DICOM 文件，导致 `UniqueViolation` 错误。

**改动** (`backend/app/dicom/parser/dicom_parser.py`):
- 在上传批次内使用 `seen_sop_uids` 集合跟踪已处理的 SOP Instance UID
- 跳过同一批次内的重复文件，避免唯一键冲突

### 后端文件存储持久化

**问题**: DICOM 文件存在容器内但未挂载 volume，容器重建后文件丢失（404），数据库记录却保留。

**改动** (`docker-compose.yml`):
- 新增 `backend_dicom:/app/static/dicom` 卷挂载，持久化 DICOM 文件
- 清理因文件丢失产生的无效数据库记录

### Viewer 非成像序列过滤

**问题**: RTSTRUCT/SEG 等非成像序列只有 1 个实例，选中后 `createAndCacheVolumeFromImages` 因元数据不足报错 `Cannot read properties of undefined (reading '0')`。

**改动** (`frontend/src/pages/ViewerPage.tsx`):
- 筛选 `['CT', 'MR', 'PT', 'NM', 'US', 'XA', 'CR']` 成像模态，过滤 RTSTRUCT、SEG 等非成像序列
- 自动选中第一个成像序列进行三维重建

### 前端删除功能

**改动** (`frontend/src/pages/StudiesPage.tsx`):
- 研究卡片右上角添加 **×** 删除按钮（hover 显示）
- 确认对话框防止误删
- 后端已有 `DELETE /api/v1/dicom/studies/{study_id}` 接口

### 已验证功能

| 功能 | 状态 |
|------|------|
| 文件夹递归上传 | ✓ |
| 重复 SOP UID 自动去重 | ✓ |
| 容器重建后 DICOM 文件持久化 | ✓ |
| 多序列研究自动选择 CT 序列 | ✓ |
| 前端删除研究 | ✓ |
