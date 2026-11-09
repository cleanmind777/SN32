#!/usr/bin/env bash
# Place the trained weights next to the tokenizer files already in git.
# Baseline DeBERTa files are separate: bash miner_prep/download_baseline.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="handoff-2026-11-09"
BASE="https://github.com/cleanmind777/SN32/releases/download/${TAG}"

fetch() {
  local name="$1"
  local dest="$2"
  mkdir -p "$(dirname "$dest")"
  echo "fetch ${name}"
  curl -fL --retry 3 -C - -o "$dest" "${BASE}/${name}"
}

fetch word-v1-model.safetensors "$ROOT/miner_prep/checkpoints/word-v1/model.safetensors"
fetch word-v2-model.safetensors "$ROOT/miner_prep/checkpoints/word-v2/model.safetensors"
fetch word-v3-model.safetensors "$ROOT/miner_prep/checkpoints/word-v3/model.safetensors"
fetch clf-v1-model.safetensors "$ROOT/miner_prep/checkpoints/clf-v1/model.safetensors"
fetch clf-v2-model.safetensors "$ROOT/miner_prep/checkpoints/clf-v2/model.safetensors"
fetch clf-v3-model.safetensors "$ROOT/miner_prep/checkpoints/clf-v3/model.safetensors"
echo "weights ready"
