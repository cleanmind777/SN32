"""Drop-in classifier for neurons/miner.py (document score broadcast to words).

Usage idea in miner:
  from miner_prep.hf_classifier import HfClassifier
  self.model = HfClassifier(model_dir="miner_prep/checkpoints/clf-v1", device=self.device)
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset


class _Texts(Dataset):
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


class HfClassifier:
    def __init__(self, model_dir: str, device: str = "cuda:0", max_length: int = 512, batch_size: int = 4):
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        self.model.eval()

    def predict_batch(self, texts):
        ds = _Texts(texts, self.tokenizer, self.max_length)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        outs = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits
                p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                outs.append(p)
        return np.concatenate(outs) if outs else np.array([])

    def __call__(self, text: str) -> float:
        return float(self.predict_batch([text])[0])
