# ==============================================================================
# FRAME Super-Resolution Launcher for Windows 11 (PowerShell)
# ==============================================================================

param (
    [switch]$Cpu
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Starting FRAME Docker Environment (Windows 11)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Docker CLI
if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Docker is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Docker Desktop for Windows: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    exit 1
}

# 2. Check Docker Daemon
docker info 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Docker Desktop daemon is not running." -ForegroundColor Red
    Write-Host "Please launch Docker Desktop and wait until the whale icon shows 'Engine running'." -ForegroundColor Yellow
    exit 1
}

# 3. Create persistent host directories
@("workspace", "models", "cache") | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ | Out-Null
    }
}

# 4. Check for NVIDIA GPU
$hasGpu = $false
if (-not $Cpu -and (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue)) {
    & nvidia-smi 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $hasGpu = $true
    }
}

# 5. Launch containers
if ($hasGpu) {
    Write-Host "[*] NVIDIA GPU detected. Launching with CUDA acceleration..." -ForegroundColor Green
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
} else {
    Write-Host "[*] Launching in CPU mode (compatible with any laptop)..." -ForegroundColor Yellow
    docker compose -f docker-compose.yml up --build
}
