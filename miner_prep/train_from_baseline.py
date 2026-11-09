#!/usr/bin/env python3
"""Fine-tune the official SN32 DeBERTa on local human + new AI text.

Saves the checkpoint with the best average of validator-style reward on the
original eval set and the held-out generator set.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW


ROOT = Path("/workspace/SN32")
FOUNDATION = ROOT / "models/deberta-v3-large-hf-weights"
PTH = ROOT / "models/deberta-large-ls03-ctx1024.pth"
OUT = ROOT / "miner_prep/checkpoints/clf-v3"


class JsonlCls(Dataset):
    def __init__(self, rows, tokenizer, max_length=512):
        self.rows = rows
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
        item["labels"] = torch.tensor(int(y), dtype=torch.long)
        return item


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def doc_pairs(rows):
    return [(r["text"], int(r["label"])) for r in rows if r.get("text", "").strip()]


def word_reward(rows, probs):
    y, p = [], []
    for row, prob in zip(rows, probs):
        labels = row.get("labels")
        n = len(row["text"].split())
        if not labels or len(labels) != n:
            labels = [int(row.get("label", 0))] * n
        y.extend(int(v) for v in labels)
        p.extend([float(prob)] * n)
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype=float)
    pred = np.round(p).astype(bool)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    f1 = f1_score(y, pred, zero_division=0)
    ap = average_precision_score(y, p) if y.any() else 0.0
    fp_score = 1.0 - fp / max(len(p), 1)
    return {
        "f1": float(f1),
        "fp_score": float(fp_score),
        "ap": float(ap),
        "reward": float((f1 + fp_score + ap) / 3),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def doc_counts(rows, probs, thr=0.5):
    y = np.array([int(r["label"]) for r in rows])
    pred = (np.asarray(probs) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return int(tp), int(fp), int(fn), int(tn)


@torch.inference_mode()
def predict(model, tok, rows, device, batch_size=8, max_length=512):
    model.eval()
    probs = []
    texts = [r["text"] for r in rows]
    for i in range(0, len(texts), batch_size):
        enc = tok(
            texts[i : i + batch_size],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs.extend(torch.softmax(logits, dim=-1)[:, 1].float().cpu().tolist())
    return probs


def build_train_rows():
    rng = random.Random(3)
    base = read_jsonl(ROOT / "miner_prep/data/train.jsonl")
    humans = [r for r in base if int(r["label"]) == 0]
    ai = [r for r in base if int(r["label"]) == 1]
    new_ai = read_jsonl(ROOT / "miner_prep/data/hf_ai_train.jsonl")
    rng.shuffle(humans)
    humans = humans[:1400]
    new_ai = new_ai * 4
    rows = doc_pairs(humans + ai + new_ai)
    rng.shuffle(rows)
    return rows


def main():
    device = "cuda:0"
    tok = AutoTokenizer.from_pretrained(FOUNDATION)
    state = torch.load(PTH, map_location="cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        FOUNDATION,
        state_dict=state,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
    ).to(device)
    model.config.attention_probs_dropout_prob = 0.0
    model.config.hidden_dropout_prob = 0.0

    train_rows = build_train_rows()
    old_eval = read_jsonl(ROOT / "miner_prep/data/eval.jsonl")
    new_eval = read_jsonl(ROOT / "miner_prep/data/hf_ai_eval.jsonl")
    print(f"train={len(train_rows)} old_eval={len(old_eval)} new_eval={len(new_eval)}", flush=True)

    ds = JsonlCls(train_rows, tok)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    optim = AdamW(model.parameters(), lr=3e-6, weight_decay=0.01)
    steps = len(loader)
    sched = get_linear_schedule_with_warmup(optim, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")

    best = -1.0
    OUT.mkdir(parents=True, exist_ok=True)
    model.train()
    for epoch in range(1):
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            optim.zero_grad(set_to_none=True)
            if step % 50 != 0 and step != steps:
                continue
            old_p = predict(model, tok, old_eval, device)
            new_p = predict(model, tok, new_eval, device)
            old_m = word_reward(old_eval, old_p)
            new_m = word_reward(new_eval, new_p)
            old_tp, old_fp, old_fn, old_tn = doc_counts(old_eval, old_p)
            new_tp, new_fp, new_fn, new_tn = doc_counts(new_eval, new_p)
            score = 0.5 * old_m["reward"] + 0.5 * new_m["reward"]
            print(
                f"step {step}/{steps} loss={loss.item():.4f} score={score:.3f} "
                f"oldR={old_m['reward']:.3f} F1={old_m['f1']:.3f} AP={old_m['ap']:.3f} "
                f"docFP={old_fp} newR={new_m['reward']:.3f} newTP={new_tp}/{len(new_eval)}",
                flush=True,
            )
            if score > best:
                best = score
                model.save_pretrained(OUT)
                tok.save_pretrained(OUT)
                print(f"saved best {best:.3f} -> {OUT}", flush=True)
            model.train()
    print(f"BEST {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
