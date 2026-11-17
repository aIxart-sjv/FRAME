@echo off
setlocal enabledelayedexpansion

echo ==========================================================
echo  Starting FRAME Docker Environment (Windows 11)
echo ==========================================================
echo.

:: 1. Check if Docker is installed
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker was not found in your PATH!
    echo Please install Docker Desktop for Windows:
    echo   https://www.docker.com/products/docker-desktop/
    echo.
    pause
    exit /b 1
)

:: 2. Check if Docker daemon is running
docker info >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker daemon is not running!
    echo Please start Docker Desktop and ensure the engine has started.
    echo.
    pause
    exit /b 1
)

:: 3. Create required host folders if missing
if not exist "workspace" mkdir workspace
if not exist "models" mkdir models
if not exist "cache" mkdir cache

:: 4. Check for NVIDIA GPU
set "USE_GPU=0"
where nvidia-smi >nul 2>nul
if %ERRORLEVEL% equ 0 (
    nvidia-smi >nul 2>nul
    if !ERRORLEVEL! equ 0 (
        echo [*] NVIDIA GPU detected via nvidia-smi.
        set "USE_GPU=1"
    )
)

:: 5. Launch containers
if "%1"=="--cpu" (
    echo [*] Starting in CPU mode (forced by --cpu flag)...
    docker compose -f docker-compose.yml up --build
) else if "!USE_GPU!"=="1" (
    echo [*] Starting with NVIDIA GPU acceleration...
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
) else (
    echo [*] Starting in CPU mode (universal compatibility)...
    docker compose -f docker-compose.yml up --build
)

echo.
echo ==========================================================
echo  Containers stopped.
echo ==========================================================
pause
