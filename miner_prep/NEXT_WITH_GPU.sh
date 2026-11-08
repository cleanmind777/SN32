#!/usr/bin/env bash
# Run this AFTER you rent a GPU machine (or attach a GPU to this host).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

echo "[1/3] Fine-tune classifier on prepared JSONL..."
python miner_prep/train_classifier.py \
  --train_jsonl miner_prep/data/train.jsonl \
  --eval_jsonl miner_prep/data/eval.jsonl \
  --model_name_or_path models/deberta-v3-large-hf-weights \
  --out_dir miner_prep/checkpoints/clf-v1 \
  --epochs 1 \
  --batch_size 4

echo "[2/3] Infer on eval set..."
python miner_prep/infer_jsonl.py \
  --in_jsonl miner_prep/data/eval.jsonl \
  --out_jsonl miner_prep/data/eval_preds.jsonl \
  --model_dir miner_prep/checkpoints/clf-v1 \
  --device cuda:0

echo "[3/3] Score with SN32-style metrics..."
python miner_prep/eval_metrics.py --jsonl miner_prep/data/eval_preds.jsonl

cat <<'EOF'

Next (still needs your wallet):
  # testnet miner using your HF checkpoint
  pm2 start --name net32-miner --interpreter python3 ./neurons/miner.py -- \
    --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY \
    --neuron.device cuda:0 --axon.port 70000 \
    --neuron.model_type hf \
    --neuron.hf_model_dir miner_prep/checkpoints/clf-v1 \
    --subtensor.network test --netuid 87 \
    --blacklist.minimum_stake_requirement 0
EOF
