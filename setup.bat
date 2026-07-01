@echo off
chcp 65001 >nul 2>&1
title CT Simulator - Docker Setup
setlocal enabledelayedexpansion

:: ============================================================
:: CT Simulator - Docker One-Click Setup (Windows)
:: ============================================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

:: ---- Check Docker ----
echo [1/4] Checking Docker...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker Desktop is not running or not installed.
    echo   Please install Docker Desktop: https://www.docker.com/products/docker-desktop/
    echo   Then start Docker Desktop and re-run this script.
    pause
    exit /b 1
)
echo    Docker is running

:: ---- Create .env if not exist ----
echo [2/4] Checking environment config...
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo    Created .env from .env.example
    ) else (
        echo [ERROR] .env.example not found
        pause
        exit /b 1
    )
) else (
    echo    .env already exists, skipping
)

:: ---- Start Docker containers ----
echo [3/4] Building and starting Docker containers...
echo    First-time setup may take 5-10 minutes. Please wait...
echo.

docker compose up -d --build

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Docker containers failed to start. Check logs above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  All services are UP!
echo ============================================================
echo.
echo  Frontend:     http://localhost:5173
echo  API Docs:     http://localhost:8000/docs
echo  Health Check: http://localhost:8000/health
echo  MinIO Admin:  http://localhost:9001  (admin / minioadmin123)
echo.
echo  View status:  docker compose ps
echo  View logs:    docker compose logs -f backend
echo  Stop:         docker compose down
echo.
echo ============================================================

pause
