# ==============================================================================
# FRAME Super-Resolution Backend Dockerfile
# Compatible with: Arch Linux and Windows 11 (Docker Desktop / WSL2)
# Supports: CPU execution and NVIDIA GPU acceleration (CUDA)
# ==============================================================================

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 1. Install system dependencies for geospatial libraries (GDAL/Rasterio), OpenCV, and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    procps \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip, setuptools, wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 3. Install PyTorch with CUDA 12.1 support (automatically falls back to CPU if no GPU)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install FRAME backend dependencies
COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install --no-cache-dir -r /app/requirements-docker.txt

# 5. Set up isolated virtual environment for SEN2SR-Mamba worker
RUN python3 -m venv /opt/sen2sr_mamba_venv && \
    /opt/sen2sr_mamba_venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/sen2sr_mamba_venv/bin/pip install --no-cache-dir torch==2.6.0+cu118 torchvision==0.21.0+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    /opt/sen2sr_mamba_venv/bin/pip install --no-cache-dir \
      https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+cu11torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
      https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu11torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl && \
    /opt/sen2sr_mamba_venv/bin/pip install --no-cache-dir einops safetensors timm transformers packaging

# 6. Copy application packages and configuration
COPY sen2sr /app/sen2sr
COPY frame /app/frame
COPY models /app/models
COPY pyproject.toml /app/pyproject.toml

# 7. Setup entrypoint script
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh && sed -i 's/\r$//' /app/docker-entrypoint.sh

# 8. Set environment configuration
ENV PYTHONPATH=/app \
    FRAME_API_WORKSPACE_DIR=/app/workspace \
    FRAME_API_DEVICE=auto \
    FRAME_MAMBA_PYTHON=/opt/sen2sr_mamba_venv/bin/python \
    FRAME_MAMBA_WEIGHTS_DIR=/app/models/SEN2SR \
    SEN2SR_BASELINE_WEIGHTS_DIR=/app/models/SEN2SRLite_RGBN \
    FRAME_API_CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000"

# Create default persistent directories
RUN mkdir -p /app/workspace /root/.cache

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "frame.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
