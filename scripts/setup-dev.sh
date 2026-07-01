#!/bin/bash
# ============================================================
# CT Simulator - Development Setup Script
# ============================================================
# Sets up local development environment for both frontend and backend.
#
# Usage:
#   chmod +x scripts/setup-dev.sh
#   ./scripts/setup-dev.sh
# ============================================================

set -e

echo "============================================"
echo " CT Simulator - Development Setup"
echo "============================================"

# ---- Frontend Setup ----
echo ""
echo "[1/4] Installing frontend dependencies..."
cd frontend
npm install
cd ..

# ---- Backend Setup ----
echo ""
echo "[2/4] Setting up backend virtual environment..."
cd backend
python -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "[3/4] Installing backend dependencies..."
pip install -r requirements.txt
cd ..

# ---- Environment Files ----
echo ""
echo "[4/4] Creating environment files..."
if [ ! -f frontend/.env ]; then
    cp .env.example frontend/.env 2>/dev/null || true
fi
if [ ! -f backend/.env ]; then
    cp .env.example backend/.env 2>/dev/null || true
fi

echo ""
echo "============================================"
echo " Setup Complete!"
echo "============================================"
echo ""
echo "Start development servers:"
echo "  Frontend: cd frontend && npm run dev"
echo "  Backend:  cd backend && uvicorn app.main:app --reload"
echo ""
echo "Or use Docker:"
echo "  docker-compose up -d"
echo "============================================"
