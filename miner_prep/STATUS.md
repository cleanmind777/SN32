# Prep status — READY FOR YOUR GPU / WALLET

Updated automatically when free prep finished.

## Done without you (no paid GPU / no wallet)

- [x] Install `detection` package + verify imports
- [x] Baseline Deberta weights under `models/`
- [x] Human data from The Pile: **2000** rows (`data/human_pile.jsonl`)
- [x] CPU validator-style augs on humans (`data/human_pile_aug.jsonl`)
- [x] Free public bootstrap HC3: **1000 human + 1000 AI** (`data/bootstrap_public*.jsonl`)
- [x] Merged train/eval: **3600 / 400** (`data/train.jsonl`, `data/eval.jsonl`)
- [x] Train / infer / eval scripts
- [x] `HfClassifier` + miner flag `--neuron.model_type hf`
- [x] One-shot GPU script: `NEXT_WITH_GPU.sh`

## Blocked until you act

| You must provide | Why |
|------------------|-----|
| GPU rental (CUDA) | Train + generate validator-like AI with Ollama + run miner |
| Bittensor wallet | Register & serve axon |
| Optional: more TAO | Mainnet registration burn |
| Optional: HF token | Only if downloads get rate-limited |

## What this prep is / is not

- **Is:** everything that can be prepared offline for free so GPU time is not wasted.
- **Is not:** a top SN32 model yet. HC3 ≠ 30+ Ollama models + Common Crawl OOD. After first GPU train, next step is generating AI with Ollama on GPU and retraining.

## Immediate next command (on a GPU box)

```bash
bash miner_prep/NEXT_WITH_GPU.sh
```
