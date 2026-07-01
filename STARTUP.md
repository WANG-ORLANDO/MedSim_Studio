# MedSim Studio 启动指南

## 一键启动

```bash
# macOS (CPU)
docker compose up -d

# Windows/Linux (GPU 加速)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

等待约 15-20 秒后，访问以下地址：

| 服务 | 地址 |
|------|------|
| **Frontend** | http://localhost:5173 |
| **Backend API** | http://localhost:8000 |
| **Swagger Docs** | http://localhost:8000/docs |
| **MinIO Console** | http://localhost:9001 (minioadmin / minioadmin123) |

## 前置条件

- Docker Desktop 已安装并运行
- Node.js 20+ (仅本地开发时需要)
- Python 3.11+ (仅本地开发时需要)
- NVIDIA 驱动 + NVIDIA Container Toolkit (仅 GPU 模式需要)

## 可用页面

| 页面 | 路由 | 功能 |
|------|------|------|
| Studies | `/studies` | DICOM 管理与上传 |
| Viewer | `/viewer/:id` | MPR 三视图 + 3D 渲染 |
| Simulation | `/simulation` | 病灶仿真 + CT Phantom |
| Segmentation | `/segmentation` | AI 器官分割 |
| Artifact | `/artifact` | 伪影生成 (7种) |
| Classifier | `/classifier` | 伪影分类 + 模型训练 |

## 常见问题排查

### 1. 服务无法访问

```bash
# 检查容器状态
docker compose ps

# 查看后端日志
docker compose logs --tail=50 backend
```

### 2. 后端启动失败

```bash
docker compose logs backend | tail -20
```

### 3. 数据库连接失败

```bash
docker compose ps postgres
docker compose restart postgres
```

### 4. 重建容器

```bash
docker compose down
docker compose up -d --build
```

## 本地开发启动

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 服务管理

```bash
# 停止所有服务
docker compose down

# 停止并删除数据卷
docker compose down -v

# 重新构建所有镜像
docker compose build

# 查看服务状态
docker compose ps
```

## 架构说明

```
Browser (React) → Nginx (5173) → FastAPI (8000)
                                    ↓
                              PostgreSQL (5432)
                              MinIO (9000/9001)
```
