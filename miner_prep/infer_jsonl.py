#!/usr/bin/env python3
"""Run a classifier over JSONL and attach predictions for eval_metrics.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset


class Texts(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--device", default=None, help="cuda:0 or cpu")
    args = p.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    rows = []
    texts = []
    with open(args.in_jsonl, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
            texts.append(row["text"])

    ds = Texts(texts, tok, args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size)

    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            p_ai = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.append(p_ai)
    probs = np.concatenate(probs)

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row, p_ai in zip(rows, probs):
            n = len(row["text"].split())
            # Miner protocol: per-word probs (broadcast doc score for now)
            row["predictions"] = [float(p_ai)] * n
            row["pred"] = float(p_ai)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} predictions -> {out} on {device}")


if __name__ == "__main__":
    main()
