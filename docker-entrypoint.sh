#!/bin/bash
set -e

echo "=========================================================="
echo " Starting FRAME Super-Resolution Backend"
echo "=========================================================="

# Ensure workspace directories exist
mkdir -p "${FRAME_API_WORKSPACE_DIR:-/app/workspace}/uploads"
mkdir -p "${FRAME_API_WORKSPACE_DIR:-/app/workspace}/jobs"
mkdir -p "${FRAME_API_WORKSPACE_DIR:-/app/workspace}/analyses"

# Display runtime device info
python -c "
import torch
cuda_avail = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_avail else 'None'
print(f'[*] PyTorch version: {torch.__version__}')
print(f'[*] CUDA Available: {cuda_avail}')
if cuda_avail:
    print(f'[*] Active GPU: {device_name}')
else:
    print('[*] Running on CPU mode (SEN2SR-Lite enabled, Mamba requires GPU)')
"

# Check Lite model weights
LITE_DIR="${SEN2SR_BASELINE_WEIGHTS_DIR:-/app/models/SEN2SRLite_RGBN}"
if [ -f "$LITE_DIR/model.safetensor" ]; then
    echo "[*] SEN2SR-Lite weights found in $LITE_DIR"
else
    echo "[!] SEN2SR-Lite weights not mounted; will download automatically from Hugging Face on first use."
fi

# Check Mamba model weights
MAMBA_DIR="${FRAME_MAMBA_WEIGHTS_DIR:-/app/models/SEN2SR}"
if [ -f "$MAMBA_DIR/sr_model.safetensor" ]; then
    echo "[*] SEN2SR-Mamba weights found in $MAMBA_DIR"
else
    echo "[!] SEN2SR-Mamba weights not found; Mamba model will be disabled."
fi

# Check Copernicus credentials
if [ -n "$FRAME_CDSE_CLIENT_ID" ] && [ -n "$FRAME_CDSE_CLIENT_SECRET" ]; then
    echo "[*] Copernicus CDSE credentials configured."
else
    echo "[!] Copernicus CDSE credentials not set (live satellite acquisition requires credentials in .env)."
fi

echo "=========================================================="
exec "$@"
