#!/usr/bin/env bash
# Download SN32 baseline Deberta weights (free HF links). No GPU needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/models"
cd "$ROOT/models"

echo "Downloading deberta-large-ls03-ctx1024.pth ..."
wget -c "https://huggingface.co/sergak0/sn32/resolve/main/deberta-large-ls03-ctx1024.pth" \
  -O deberta-large-ls03-ctx1024.pth

echo "Downloading deberta-v3-large-hf-weights.zip ..."
wget -c "https://huggingface.co/sergak0/sn32/resolve/main/deberta-v3-large-hf-weights.zip" \
  -O deberta-v3-large-hf-weights.zip

if [[ ! -d deberta-v3-large-hf-weights ]]; then
  unzip -o deberta-v3-large-hf-weights.zip -d deberta-v3-large-hf-weights
fi

echo "Baseline files ready under $ROOT/models"
ls -lh deberta-large-ls03-ctx1024.pth deberta-v3-large-hf-weights | head
