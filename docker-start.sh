#!/bin/bash
set -e

echo "=========================================================="
echo " Starting FRAME Docker Environment (Linux / Arch)"
echo "=========================================================="

# Check Docker CLI
if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] Docker is not installed. Please install Docker first:"
    echo "  Arch Linux: sudo pacman -S docker docker-compose"
    exit 1
fi

# Check Docker socket permissions
if ! docker info >/dev/null 2>&1; then
    echo "[WARNING] Cannot connect to Docker daemon as user $USER."
    echo "If you see 'permission denied', add your user to the docker group:"
    echo "  sudo usermod -aG docker \$USER"
    echo "  newgrp docker"
    echo ""
    echo "Attempting to run with sudo..."
    DOCKER_CMD="sudo docker"
else
    DOCKER_CMD="docker"
fi

# Detect compose command
if $DOCKER_CMD compose version >/dev/null 2>&1; then
    COMPOSE_CMD="$DOCKER_CMD compose"
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
else
    echo "[ERROR] Docker Compose plugin is not installed."
    echo "  Arch Linux: sudo pacman -S docker-compose"
    exit 1
fi

# Check for NVIDIA GPU
USE_GPU=false
if [ "$1" != "--cpu" ] && command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        echo "[*] NVIDIA GPU detected via nvidia-smi."
        USE_GPU=true
    fi
fi

# Create workspace and cache directories if missing
mkdir -p workspace models cache

if [ "$USE_GPU" = true ]; then
    echo "[*] Launching with NVIDIA GPU acceleration..."
    $COMPOSE_CMD -f docker-compose.yml -f docker-compose.gpu.yml up --build "$@"
else
    echo "[*] Launching in CPU mode (universal compatibility)..."
    $COMPOSE_CMD -f docker-compose.yml up --build "$@"
fi
