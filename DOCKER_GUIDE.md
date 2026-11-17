# FRAME Docker Deployment Guide

This guide explains how to build, run, and verify the **FRAME (Sentinel-2 Super-Resolution Framework)** dockerized application across **Arch Linux** and **Windows 11**.

---

## 🏗️ Architecture Overview

The application is orchestrated using Docker Compose with two services connected over an internal bridge network:

```
                  ┌─────────────────────────────────────────┐
                  │              User Browser               │
                  │   http://localhost:3000 (Web Console)   │
                  │   http://localhost:8000/docs (FastAPI)  │
                  └───────────────┬─────────────────────────┘
                                  │
                  ┌───────────────┴─────────────────────────┐
                  │             Docker Host                 │
                  │       (Arch Linux / Windows 11)         │
                  └───────────────┬─────────────────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         │                                                 │
         ▼                                                 ▼
┌──────────────────────────────┐          ┌──────────────────────────────┐
│       frame-frontend         │          │        frame-backend         │
│         (Port 3000)          │          │         (Port 8000)          │
│                              │          │                              │
│  - Nginx Alpine web server   │─────────▶│  - FastAPI / Uvicorn server  │
│  - Static React + Vite SPA   │ (Proxy)  │  - PyTorch (CUDA / CPU auto) │
│  - MapLibre GL AOI picker    │          │  - GDAL / Rasterio / GeoTIFF │
│  - NDVI / Stability Viewer   │          │  - Isolated Mamba Worker     │
└──────────────────────────────┘          └──────────────┬───────────────┘
                                                         │
                                  ┌──────────────────────┴───────────────┐
                                  │          Mounted Volumes             │
                                  │  - ./models    -> /app/models        │
                                  │  - ./workspace -> /app/workspace     │
                                  │  - ./cache     -> /root/.cache       │
                                  └──────────────────────────────────────┘
```

### Key Highlights
- **Universal CPU & GPU Compatibility**: Runs out of the box on standard CPU (Intel, AMD) using **SEN2SR-Lite**, and seamlessly activates hardware acceleration for **SEN2SR-Lite** and **SEN2SR-Mamba** when an NVIDIA GPU is available.
- **Cross-Platform**: Configured with `.gitattributes` and sanitized POSIX scripts to eliminate Windows CRLF newline issues.
- **Persistent Storage**: Uploaded scenes, processed GeoTIFFs, STAC items, and model caches are preserved on host storage.

---

## 🪟 Windows 11 Setup (For Your Friend)

### 1. Prerequisites
1. **Docker Desktop for Windows**:
   - Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
   - During installation, keep **"Use the WSL 2 based engine"** checked (default).
   - Launch Docker Desktop and wait until the whale icon in the taskbar shows that the engine is running.
2. *(Optional - for NVIDIA GPU users)*:
   - Install the latest NVIDIA driver for Windows from [nvidia.com](https://www.nvidia.com/Download/index.aspx).
   - WSL2 in Windows 11 passes CUDA directly from the Windows NVIDIA driver into Docker—**no Linux driver setup required inside WSL2!**

### 2. How to Run (1-Click)
Open the project directory in File Explorer:
- **Option A (Double-click)**: Simply double-click `docker-start.bat`.
- **Option B (PowerShell)**:
  ```powershell
  .\docker-start.ps1
  ```
- **Option C (Standard Docker Compose)**:
  ```powershell
  # For CPU mode:
  docker compose up --build

  # For NVIDIA GPU acceleration:
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
  ```

---

## 🐧 Arch Linux Setup

### 1. Prerequisites
Ensure Docker and Docker Compose are installed:
```bash
sudo pacman -S docker docker-compose
sudo systemctl enable --now docker.service
```

Add your current user to the `docker` group so you can run Docker without `sudo`:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

*(Optional - for NVIDIA GPU acceleration on Arch)*:
Install the NVIDIA Container Toolkit:
```bash
sudo pacman -S nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. How to Run
From the repository root:
- **Automated script** (auto-detects NVIDIA GPU):
  ```bash
  ./docker-start.sh
  ```
- **Force CPU mode**:
  ```bash
  ./docker-start.sh --cpu
  ```
- **Direct Docker Compose command**:
  ```bash
  # CPU mode:
  docker compose up --build

  # GPU mode:
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
  ```

---

## 🌐 Accessing the Application

Once the containers are up:

| Interface | URL | Description |
|---|---|---|
| **Frontend Web App** | [http://localhost:3000](http://localhost:3000) | Full UI: Upload GeoTIFF, select model, AOI map, run SR, NDVI analysis & export |
| **Backend Swagger API** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive API documentation and test bench |
| **Health Check** | [http://localhost:8000/health](http://localhost:8000/health) | Reports status, registered models, and GPU/CPU availability |

---

## ⚙️ Configuration & Credentials

### Copernicus Data Space Ecosystem (CDSE)
To enable live Sentinel-2 scene search and download from Copernicus:
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Enter your Copernicus CDSE API credentials:
   ```env
   FRAME_CDSE_CLIENT_ID=your-client-id
   FRAME_CDSE_CLIENT_SECRET=your-client-secret
   ```
*(If left empty, offline GeoTIFF upload and super-resolution still work completely without restrictions).*

---

## 📂 Persistent Directories

- **`./models/`**:
  - `models/SEN2SRLite_RGBN`: Pre-cached SEN2SR-Lite weights (~7.6 MB).
  - `models/SEN2SR`: Pre-cached SEN2SR-Mamba weights (~367 MB).
  Mounting this folder guarantees offline availability without needing to re-download on every run.
- **`./workspace/`**:
  - `workspace/uploads`: Uploaded Sentinel-2 GeoTIFF rasters.
  - `workspace/jobs`: Output super-resolved 2.5 m GeoTIFFs, COGs, and STAC metadata.
  - `workspace/analyses`: Downstream NDVI calculations and diff rasters.

---

## 🔍 Verification & Testing

### 1. Health & Model Verification
Run this command in any terminal:
```bash
curl -s http://localhost:8000/health | jq
```
Expected response:
```json
{
  "status": "ok",
  "api_version": "0.1.0",
  "default_model": "lite",
  "available_models": [
    {
      "id": "lite",
      "label": "SEN2SR-Lite",
      "available": true
    },
    {
      "id": "mamba",
      "label": "SEN2SR-Mamba",
      "available": true
    }
  ]
}
```
*(On CPU-only laptops, `mamba` will cleanly report `available: false` with reason `"SEN2SR-Mamba requires a CUDA-capable GPU"`, while `lite` remains `available: true`).*

---

## 🛠️ Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `permission denied while connecting to docker.sock` (Arch Linux) | User not in `docker` group | Run `sudo usermod -aG docker $USER && newgrp docker` |
| `could not select device driver "" with capabilities: [[gpu]]` | Docker Compose GPU requested without NVIDIA runtime | Run in CPU mode: `docker compose up --build` or `./docker-start.sh --cpu` |
| `WSL 2 memory exhausted` (Windows 11) | Docker Desktop default memory cap too small for large rasters | Create `C:\Users\<YourUser>\.wslconfig` with `[wsl2]` and `memory=8GB`, then run `wsl --shutdown` |
| `bash: \r: command not found` | Git converted line endings to CRLF on Windows | Git attributes already enforce LF; if re-cloning on Windows, use `git clone -c core.autocrlf=input <repo>` |
