#!/usr/bin/env bash
set -e

# Only install if torch isn't found in the current environment
if ! python -c "import torch" &>/dev/null; then
    echo "Torch not found. Installing..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Torch already present in base image. Skipping heavy download."
fi

pip install -e ".[serve,dev]"
