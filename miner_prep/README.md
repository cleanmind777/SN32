# Miner prep (pre-GPU → GPU handoff)

## Free prep already done

See [STATUS.md](STATUS.md).

## Layout

```
miner_prep/
  data/
    human_pile.jsonl              # Pile humans
    human_pile_aug.jsonl          # + CPU augs
    bootstrap_public.jsonl        # HC3 human/AI
    bootstrap_public_aug.jsonl
    train.jsonl / eval.jsonl      # merged splits
  checkpoints/                    # filled after GPU train
  sample_human_pile.py
  download_public_bootstrap.py
  augment_dataset.py
  merge_jsonl.py
  train_classifier.py             # needs GPU
  infer_jsonl.py
  eval_metrics.py
  hf_classifier.py                # also copied to neurons/miners/
  NEXT_WITH_GPU.sh
  NEEDS_YOU.md
  STATUS.md
```

## After GPU is ready

```bash
bash miner_prep/NEXT_WITH_GPU.sh
```

Miner with custom checkpoint:

```bash
python neurons/miner.py \
  --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY \
  --neuron.device cuda:0 --axon.port 70000 \
  --neuron.model_type hf \
  --neuron.hf_model_dir miner_prep/checkpoints/clf-v1
```
