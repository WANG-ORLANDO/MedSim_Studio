# MedSim Studio 启动指南

## 一键启动

```bash
docker compose up -d
```

等待约 15-20 秒后，访问以下地址：

| 服务 | 地址 |
|------|------|
| **Frontend** | http://localhost:5173 |
| **Backend API** | http://localhost:8000 |
| **Swagger Docs** | http://localhost:8000/docs |
| **MinIO Console** | http://localhost:9001 (用户名/密码: minioadmin / minioadmin123) |

## 前置条件

- Docker Desktop 已安装并运行
- Node.js 20+ (仅本地开发时需要)
- Python 3.11+ (仅本地开发时需要)

## 常见问题排查

### 1. Artifact 页面显示 "No CT series found. Upload DICOM first."

**原因**: Docker 镜像未包含最新的 artifact API 路由文件。

**解决**: 重新构建并启动后端：

```bash
docker compose build backend
docker compose up -d backend
```

### 2. "Upload DICOM" 按钮点击无效

**原因**: 可能是前端未正确连接到后端 API。

**排查**:
```bash
# 检查后端是否健康
curl http://localhost:8000/api/v1/health

# 检查前端容器是否运行
docker compose ps frontend
```

### 3. 后端启动失败

**查看日志**:
```bash
docker compose logs --tail=50 backend
```

### 4. 数据库连接失败

```bash
# 检查 PostgreSQL 是否健康
docker compose ps postgres

# 重启数据库
docker compose restart postgres
```

### 5. MinIO 对象存储不可用

```bash
# 检查 MinIO 状态
docker compose ps minio

# 重启 MinIO
docker compose restart minio
```

## 本地开发启动

### Frontend

```bash
cd frontend
npm install
npm run dev
```

前端将在 http://localhost:5173 启动，API 代理到 http://localhost:8000。

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

后端将在 http://localhost:8000 启动。

## 数据上传

### 上传 DICOM 文件

1. 打开 http://localhost:5173/studies
2. 点击 "Upload DICOM" 选择 `.dcm` 文件
3. 或点击 "Upload Folder" 选择包含 DICOM 文件的文件夹
4. 上传完成后自动跳转到查看器页面

### 测试数据

项目包含一个预上传的 CT 研究 (LUNG1-001)，可直接在 Studies 页面查看。

## 服务管理

```bash
# 停止所有服务
docker compose down

# 停止并删除数据卷（清空数据库和存储）
docker compose down -v

# 重新构建所有镜像
docker compose build

# 查看所有服务状态
docker compose ps
```

## 环境变量

主要配置项（在 `.env` 文件中）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `POSTGRES_HOST` | postgres | 数据库主机 |
| `POSTGRES_DB` | ct_simulator | 数据库名 |
| `MINIO_HOST` | minio | 对象存储主机 |
| `AI_MONAI_ENABLED` | false | 是否启用 AI 分割 |
| `SIMULATION_DEFAULT_SEED` | 42 | 仿真随机种子 |

## 架构说明

```
Browser (React) → Nginx (5173) → FastAPI (8000)
                                    ↓
                              PostgreSQL (5432)
                              MinIO (9000/9001)
```

所有服务通过 Docker 网络 `ct-network` 互联，外部只暴露前端和后端端口。
