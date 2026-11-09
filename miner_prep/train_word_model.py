#!/usr/bin/env python3
"""Train a word-level DeBERTa so mixed Common Crawl pages can clear the 0.9 F1 gate.

The encoder starts from clf-v3. Each whitespace word gets its own probability.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

ROOT = Path("/workspace/SN32")
sys.path.insert(0, str(ROOT))
FOUNDATION = ROOT / "models/deberta-v3-large-hf-weights"
CLF_V3 = ROOT / "miner_prep/checkpoints/clf-v3"
OUT = ROOT / "miner_prep/checkpoints/word-v5"
RESUME = ROOT / "miner_prep/checkpoints/word-v2"
MAX_LEN = 512


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def pack_words(words, labels, tok):
    """Split a word sequence into chunks that fit in MAX_LEN tokens."""
    if not words:
        return []
    chunks = []
    start = 0
    n = len(words)
    while start < n:
        lo, hi = start + 1, n
        best = start + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            ids = tok(words[start:mid], is_split_into_words=True, add_special_tokens=True)["input_ids"]
            if len(ids) <= MAX_LEN:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        chunks.append((words[start:best], labels[start:best]))
        if best == start:
            break
        start = best
    return chunks


class WordChunks(Dataset):
    def __init__(self, rows, tok):
        self.tok = tok
        self.items = []
        for row in rows:
            words = row["text"].split()
            labels = [int(v) for v in row["labels"]]
            if len(words) != len(labels) or not words:
                continue
            self.items.extend(pack_words(words, labels, tok))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        words, labels = self.items[idx]
        enc = self.tok(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
        )
        word_labels = []
        for wid in enc.word_ids():
            word_labels.append(-100 if wid is None else int(labels[wid]))
        item = {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(word_labels),
        }
        return item


def freeze_early(model):
    for p in model.parameters():
        p.requires_grad = False
    for layer in model.deberta.encoder.layer[-4:]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable={n}", flush=True)


def load_model(device):
    if RESUME.exists():
        model = AutoModelForTokenClassification.from_pretrained(RESUME)
        freeze_early(model)
        return model.to(device)
    cfg = AutoConfig.from_pretrained(FOUNDATION)
    cfg.num_labels = 2
    cfg.attention_probs_dropout_prob = 0.0
    cfg.hidden_dropout_prob = 0.0
    cfg.id2label = {0: "human", 1: "ai"}
    cfg.label2id = {"human": 0, "ai": 1}
    model = AutoModelForTokenClassification.from_config(cfg)
    seq = AutoModelForSequenceClassification.from_pretrained(CLF_V3)
    model.deberta.load_state_dict(seq.deberta.state_dict())
    del seq
    return model.to(device)


@torch.inference_mode()
def predict_words(model, tok, texts, device, batch_size=8):
    model.eval()
    out = []
    # Chunk every text, then batch chunks. Record which text they belong to.
    jobs = []
    for ti, text in enumerate(texts):
        words = text.split()
        if not words:
            out.append([])
            continue
        chunks = pack_words(words, [0] * len(words), tok)
        # pack_words with dummy labels; recompute from word slices
        start = 0
        for cwords, _ in chunks:
            jobs.append((ti, start, cwords))
            start += len(cwords)
        out.append([0.0] * len(words))
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i : i + batch_size]
        enc = tok(
            [cwords for _, _, cwords in batch],
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        word_ids = [enc.word_ids(j) for j in range(len(batch))]
        enc_t = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        probs = torch.softmax(model(**enc_t).logits, dim=-1)[:, :, 1]
        for bi, (ti, start, cwords) in enumerate(batch):
            buckets = [[] for _ in cwords]
            for pos, wid in enumerate(word_ids[bi]):
                if wid is None or wid >= len(cwords):
                    continue
                buckets[wid].append(float(probs[bi, pos]))
            for w, vals in enumerate(buckets):
                out[ti][start + w] = float(np.mean(vals)) if vals else 0.0
    return out


def word_f1(rows, preds):
    y, p = [], []
    for row, pred in zip(rows, preds):
        labels = [int(v) for v in row["labels"]]
        n = min(len(labels), len(pred))
        y.extend(labels[:n])
        p.extend(pred[:n])
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype=float)
    if len(y) == 0 or not y.any():
        return 0.0, 0.0, 0.0
    pred = np.round(p).astype(bool)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    f1 = f1_score(y, pred, zero_division=0)
    ap = average_precision_score(y, p)
    fp_score = 1.0 - fp / max(len(p), 1)
    return float(f1), float(fp_score), float(ap)


def window_rows(rows, seed):
    sys.path.insert(0, str(ROOT))
    from detection.validator.segmentation_processer import SegmentationProcesser

    rng_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    sp = SegmentationProcesser()
    out = []
    for row in rows:
        text, labels = sp.subsample_words(row["text"], [int(v) for v in row["labels"]])
        if len(labels) < 5 or len(text.split()) != len(labels):
            continue
        out.append({**row, "text": text, "labels": [int(v) for v in labels]})
    random.setstate(rng_state)
    np.random.set_state(np_state)
    return out


def pure_spans(row):
    """Pull the human-only and AI-only stretches out of a mixed page."""
    words = row["text"].split()
    labels = [int(v) for v in row["labels"]]
    extra = []
    buf = []
    buf_label = None

    def flush():
        nonlocal buf, buf_label
        if buf_label == 1 and len(buf) >= 12:
            extra.append(
                {
                    **row,
                    "text": " ".join(buf),
                    "labels": [1] * len(buf),
                    "label": 1,
                    "kind": "full_ai",
                }
            )
        if buf_label == 0 and len(buf) >= 20:
            extra.append(
                {
                    **row,
                    "text": " ".join(buf),
                    "labels": [0] * len(buf),
                    "label": 0,
                    "kind": "human",
                }
            )
        buf = []
        buf_label = None

    for word, label in zip(words, labels):
        if buf_label is None:
            buf_label = label
        if label != buf_label:
            flush()
            buf_label = label
        buf.append(word)
    flush()
    return extra


def build_train_rows():
    rng = random.Random(6)
    hard = read_jsonl(ROOT / "miner_prep/cc-cache/cc_hard.jsonl")
    cc = read_jsonl(ROOT / "miner_prep/cc-cache/cc_train_spans.jsonl")
    anchors = [r for r in cc if r.get("kind") == "human"]
    rows = hard * 4 + anchors
    rng.shuffle(rows)
    print(f"hard={len(hard)} anchors={len(anchors)}", flush=True)
    return rows


def main():
    device = "cuda:0"
    tok = AutoTokenizer.from_pretrained(FOUNDATION)
    train_rows = build_train_rows()
    hold = read_jsonl(ROOT / "miner_prep/cc-cache/cc_holdout_spans.jsonl")
    hold_w = window_rows(hold, seed=1)
    domain = read_jsonl(ROOT / "miner_prep/data/eval.jsonl")
    random.Random(2).shuffle(domain)
    domain = domain[:80]
    domain_w = window_rows(domain, seed=2)
    print(
        f"train_rows={len(train_rows)} hold={len(hold)} hold_w={len(hold_w)} domain_w={len(domain_w)}",
        flush=True,
    )
    model = load_model(device)
    ds = WordChunks(train_rows, tok)
    print(f"chunks={len(ds)}", flush=True)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(trainable, lr=8e-6, weight_decay=0.0)
    steps = len(loader)
    sched = get_linear_schedule_with_warmup(optim, max(1, int(0.05 * steps)), steps)
    scaler = torch.amp.GradScaler("cuda")
    weight = torch.tensor([1.0, 1.15], device=device)
    best = -1.0
    OUT.mkdir(parents=True, exist_ok=True)
    epochs = 1
    global_step = 0
    total = epochs * steps
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, 2),
                    batch["labels"].view(-1),
                    weight=weight,
                    ignore_index=-100,
                )
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            optim.zero_grad(set_to_none=True)
            if global_step % 8 != 0 and global_step != total:
                continue
            hard, easy = [], []
            for seed in (0, 2, 4, 6):
                folded = window_rows(hold, seed)
                preds = predict_words(model, tok, [r["text"] for r in folded], device)
                f1, _, _ = word_f1(folded, preds)
                hard.append(f1)
            for seed in (1, 3, 5, 7):
                folded = window_rows(hold, seed)
                preds = predict_words(model, tok, [r["text"] for r in folded], device)
                f1, _, _ = word_f1(folded, preds)
                easy.append(f1)
            hard_mean = float(np.mean(hard))
            easy_min = float(np.min(easy))
            dom_p = predict_words(model, tok, [r["text"] for r in domain_w], device)
            df1, dfp, dap = word_f1(domain_w, dom_p)
            print(
                f"step {global_step}/{total} loss={loss.item():.4f} "
                f"hardMean={hard_mean:.3f} hard={ [round(v,3) for v in hard] } "
                f"easyMin={easy_min:.3f} easy={ [round(v,3) for v in easy] } "
                f"domF1={df1:.3f} domFP={dfp:.3f}",
                flush=True,
            )
            # word-v2 baseline on these folds is hardMean 0.809 and easyMin 0.820.
            if hard_mean >= 0.83 and easy_min >= 0.80 and df1 >= 0.90 and hard_mean > best:
                best = hard_mean
                model.save_pretrained(OUT)
                tok.save_pretrained(OUT)
                print(f"saved {best:.3f} -> {OUT}", flush=True)
            if easy_min < 0.78 or hard_mean < 0.76:
                print("early stop", flush=True)
                print(f"BEST {best:.4f}", flush=True)
                return
            model.train()
    print(f"BEST {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
