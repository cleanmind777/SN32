#!/usr/bin/env python3
"""Fine-tune a sequence classifier on JSONL (document-level labels).

Requires GPU for practical speed. On CPU it will run but is very slow.

JSONL fields: text, label (0/1)  — word-level labels are reduced by majority/any.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class JsonlClsDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                text = row["text"].strip()
                if not text:
                    continue
                if "label" in row:
                    y = int(row["label"])
                else:
                    labs = row.get("labels") or []
                    y = int(sum(labs) > len(labs) / 2) if labs else 0
                self.rows.append((text, y))
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        text, y = self.rows[idx]
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(y, dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": float(acc)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--eval_jsonl", default=None)
    p.add_argument(
        "--model_name_or_path",
        default="models/deberta-v3-large-hf-weights",
        help="HF id or local foundation weights",
    )
    p.add_argument("--out_dir", default="miner_prep/checkpoints/clf-v1")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device_count = torch.cuda.device_count()
    if not args.cpu and device_count == 0:
        raise SystemExit(
            "No CUDA GPU detected. Rent a GPU, then re-run. "
            "Or pass --cpu for a tiny smoke test only."
        )

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=2
    )

    train_ds = JsonlClsDataset(args.train_jsonl, tok, args.max_length)
    eval_ds = (
        JsonlClsDataset(args.eval_jsonl, tok, args.max_length)
        if args.eval_jsonl
        else None
    )
    print(f"train size={len(train_ds)} eval size={0 if eval_ds is None else len(eval_ds)}")

    targs = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=200 if eval_ds is not None else None,
        save_steps=200,
        logging_steps=50,
        fp16=device_count > 0 and not args.cpu,
        report_to=[],
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics if eval_ds is not None else None,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
