# SN32 handoff (2026-11-09)

This pod is being replaced. The notes below are what the next GPU needs in order to continue. Do not register this wallet on mainnet (finney, netuid 32) until a checkpoint clears the reward gate below.

## Result

The best checkpoint is `miner_prep/checkpoints/word-v2`. It is a word-level DeBERTa (token classification). It does **not** clear the validator pay gate.

On held-out Common Crawl (`miner_prep/cc-cache/cc_holdout_spans.jsonl`, CC-MAIN-2019-18), scored after the validator’s own text windows, eight draws:

| draw | word F1 |
| --- | --- |
| 0 | 0.767 |
| 1 | 0.930 |
| 2 | 0.809 |
| 3 | 0.915 |
| 4 | 0.821 |
| 5 | 0.851 |
| 6 | 0.839 |
| 7 | 0.820 |

Mean **0.844**. Worst draw **0.767**. In-domain window F1 on `miner_prep/data/eval.jsonl` is about **0.926**.

The validator (`detection/validator/reward.py`) keeps a running average of Common Crawl F1 (EMA, alpha 0.2, start 1.0). If that average drops below `neuron.out_of_domain_min_f1_score` (default **0.9**), the miner reward is forced to 0. A steady F1 of 0.844 falls through 0.9, so this checkpoint would stop earning. It was not registered on mainnet.

Predictions must be one probability per whitespace word, each in `[0, 1]`, and stable across batch sizes after `round(p, 2)`. A bad validator version must return empty predictions.

## What is being served

Testnet only. The live miner still serves the document model `clf-v3`, which copies one document probability onto every word. On the same Common Crawl setup that model’s out-of-domain F1 was about **0.42**, so it also fails the gate. `neurons/miner.py` still does that broadcast. A word-level checkpoint cannot earn until `forward` returns one probability per word.

Testnet netuid 87, UID 193. Hotkey name `SnowGirl`, SS58 `5HYfy3QwqcffuH8UCVNuEBhzm2xzph1qNLiSqzQgRvPpsrjZ`. Coldkey name `pampam_oro_1`, public `5EC8x64Ja3m5u8VARgKn97dRsQNyBxDz7GkY15V3rvKaMnxm`. Axon on this pod was `127.0.0.1:8091`, advertised as `159.223.110.159:38273` through `miner_prep/bore_forward.py`. That tunnel dies with this machine.

## Wallet is not in this repo

The repo is public. Private keys were **not** pushed. They exist only on this pod at `~/.bittensor/wallets`. Copy that directory off the pod before deleting it, or the hotkey is gone. Do not commit it.

Coldkey free balance on finney was about 0.398 TAO. A registration recycle was about 0.049 TAO. Subnet 32 was full (256 UIDs). Immunity is 7200 blocks.

## Checkpoints

| dir | keep? | note |
| --- | --- | --- |
| `word-v2` | yes, best | resumed from word-v1. Selection score at step 1000: holdout mean 0.879, min 0.820 on seeds 1/3/5/7, domain F1 0.926. The eight-draw table above is the stricter later measurement. |
| `word-v1` | parent only | single-seed 0.908 was unstable. Multi-seed min about 0.726. Final EMA about 0.818. |
| `word-v3` | no | extra full-AI copies made holdout worse (min about 0.68). Training was stopped. |
| `word-v4`, `word-v5` | empty | later finetunes were not saved. v4 min 0.787. A last-layer pass did not move the score. A hard-example pass dropped an easy draw to 0.739 and domain F1 to 0.845 within 8 steps. |
| `clf-v3` | best document model | still what the testnet miner loads. Joint local reward 0.888 vs the public baseline 0.790. Fails Common Crawl. |
| `clf-v1`, `clf-v2` | no | worse than clf-v3. |

`miner_prep/train_word_model.py` is left pointed at `word-v5` / resume `word-v2`, training the hard-example file `cc_hard.jsonl`. Do not treat a new run of that script as the word-v2 recipe. It will not overwrite `word-v2` unless `OUT` is changed.

## Why more training of this setup stalled

The misses are full-AI continuations that read like the original webpage, plus a few human boilerplate pages (Cloudflare email-protection text, search-UI number lists). Human-then-AI boundaries are mostly fine. Pushing the weights toward the missed pages makes the model call human text AI before Common Crawl F1 reaches 0.9.

Marking non-English windows as AI and the Cloudflare boilerplate as human, on this same holdout, reached mean 0.883 and worst 0.842. Still under 0.9, and that rule was chosen by looking at the holdout.

Local generators are only `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `microsoft/phi-2`, and `mistralai/Mistral-7B-Instruct-v0.2`. Mainnet validators use much larger models (Llama 70B, Qwen 72B, Command-R 104B, Mistral Large 123B, and others in `detection/validator/data_generator.py`). KenLM perplexity buckets were not applied. `en.arpa.bin` was not downloaded.

## Data

Do not train on the 2019 holdout.

| file | role |
| --- | --- |
| `miner_prep/cc-cache/cc_holdout_spans.jsonl` | 33 rows, CC-MAIN-2019-18. Holdout. Already used to pick checkpoints. |
| `miner_prep/cc-cache/cc_en_2019-18.jsonl` | 80 English pages. Holdout source. Segment in `holdout_seg.txt`. |
| `miner_prep/cc-cache/cc_train_spans.jsonl` | 57 rows, CC-MAIN-2022-49. Train. |
| `miner_prep/cc-cache/cc_full_ai_extra.jsonl` | 30 human + 30 full-AI, 2022-49 and 2021-31. Train. |
| `miner_prep/cc-cache/cc_more_2020.jsonl` | 66 rows, CC-MAIN-2020-16. Train. |
| `miner_prep/cc-cache/cc_hard.jsonl` | 27 train rows where word-v2 word error was at least 0.20. |
| `miner_prep/cc-cache/cc_reward_rows.jsonl` | 42 rows used for the clf-v3 Common Crawl score. Not the holdout. |
| `miner_prep/data/hf_ai_train.jsonl` / `hf_ai_eval.jsonl` | local-generator text, mostly not Common Crawl. |
| `miner_prep/data/train.jsonl` / `eval.jsonl` | document-level HC3/pile style rows. |

Common Crawl filter used here: WET conversion records, raw length at least 300, fastText English at least 0.5, then the validator’s 1875-character cut. Perplexity head/middle/tail buckets were not applied.

## Restore on the next machine

```bash
git clone https://github.com/cleanmind777/SN32.git
cd SN32
bash miner_prep/download_baseline.sh
bash miner_prep/fetch_handoff_weights.sh
mkdir -p cc_net/bin
wget -O cc_net/bin/lid.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
export HF_HOME=$PWD/miner_prep/hf-cache
```

`download_baseline.sh` pulls the public DeBERTa weights from `sergak0/sn32` on Hugging Face. `fetch_handoff_weights.sh` pulls the trained `model.safetensors` files from the GitHub release `handoff-2026-11-09`. Tokenizer files are already in git. The Hugging Face cache (TinyLlama, Phi-2, Mistral-7B-Instruct-v0.2) was not uploaded; those models download on first use when `HF_HOME` is set.

GitHub rejects git blobs over 100 MB and release files over 2 GB. Optimizer states (`optimizer.pt`, about 3.3 GB) and the intermediate `checkpoint-600` / `checkpoint-800` weight files were left behind. The final `model.safetensors` in each checkpoint directory is enough to run or finetune.

Package version in this tree is detection 3.15.1. This pod used Python 3.12 and torch 2.8.0+cu128. Keep the installed torch build; do not let pip replace it. `accelerate` must stay 0.34.2. `transformers` 4.36.0. fastText’s Python `predict` crashes on NumPy 2; call `model.f.predict(text + "\n", 1, 0.0, "strict")` instead.
