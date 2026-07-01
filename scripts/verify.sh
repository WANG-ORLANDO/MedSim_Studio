#!/usr/bin/env bash
# ============================================================
# CT Simulator - 启动验证脚本
# 用法: bash scripts/verify.sh
# ============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "==========================================="
echo "  CT Simulator 启动验证"
echo "==========================================="
echo ""

# 1. Check Docker containers
echo -e "${YELLOW}[1/5] 检查 Docker 容器状态...${NC}"
CONTAINERS=$(docker compose ps --format "{{.Name}} {{.Status}}" 2>/dev/null || true)
for name in ct-backend ct-frontend ct-postgres ct-minio; do
    if echo "$CONTAINERS" | grep -q "$name.*running\|$name.*healthy"; then
        echo -e "  ${GREEN}✓${NC} $name 运行中"
    else
        echo -e "  ${RED}✗${NC} $name 未运行"
    fi
done
echo ""

# 2. Check Backend health
echo -e "${YELLOW}[2/5] 检查后端健康状态...${NC}"
HEALTH=$(curl -s http://localhost:8000/api/v1/health 2>/dev/null || echo '{"status":"unreachable"}')
if echo "$HEALTH" | grep -q '"healthy"'; then
    echo -e "  ${GREEN}✓${NC} Backend API 健康"
else
    echo -e "  ${RED}✗${NC} Backend API 不可达"
fi
echo ""

# 3. Check Artifact API (commonly missing route)
echo -e "${YELLOW}[3/5] 检查 Artifact API 路由...${NC}"
ARTIFACT_TYPES=$(curl -s http://localhost:8000/api/v1/artifact/types 2>/dev/null || echo '{"detail":"Not Found"}')
if echo "$ARTIFACT_TYPES" | grep -q '"types"'; then
    echo -e "  ${GREEN}✓${NC} Artifact API 正常"
else
    echo -e "  ${RED}✗${NC} Artifact API 不可用 — 需要重建后端:"
    echo "      docker compose build backend && docker compose up -d backend"
fi
echo ""

# 4. Check DICOM studies
echo -e "${YELLOW}[4/5] 检查 DICOM 数据...${NC}"
STUDIES=$(curl -s 'http://localhost:8000/api/v1/dicom/studies?page=1&page_size=50' 2>/dev/null || echo '{"items":[]}')
TOTAL=$(echo "$STUDIES" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total',0))" 2>/dev/null || echo "0")
if [ "$TOTAL" -gt 0 ]; then
    echo -e "  ${GREEN}✓${NC} 已加载 $TOTAL 个研究"
else
    echo -e "  ${YELLOW}!${NC} 暂无 DICOM 数据，请上传"
fi
echo ""

# 5. Check Frontend
echo -e "${YELLOW}[5/5] 检查前端服务...${NC}"
FRONTEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5173 2>/dev/null || echo "000")
if [ "$FRONTEND_STATUS" = "200" ]; then
    echo -e "  ${GREEN}✓${NC} Frontend 可访问"
else
    echo -e "  ${RED}✗${NC} Frontend 不可达 (HTTP $FRONTEND_STATUS)"
fi
echo ""

echo "==========================================="
echo "  验证完成"
echo "==========================================="
echo ""
echo "访问地址:"
echo "  Frontend:     http://localhost:5173"
echo "  Backend API:  http://localhost:8000"
echo "  Swagger Docs: http://localhost:8000/docs"
echo "  MinIO Console: http://localhost:9001"
