#!/usr/bin/env python3
"""Local reward-style metrics matching SN32 validator spirit.

Reward ≈ mean(F1, FP_score, AP) where:
  FP_score = 1 - FP / n
Predictions and labels can be:
  - document-level floats/bools, or
  - flattened word-level lists
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Union

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score


ArrayLike = Union[Sequence[float], Sequence[Sequence[float]]]


def flatten(x: ArrayLike) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=float)
    if isinstance(x[0], (list, tuple, np.ndarray)):
        return np.asarray([float(v) for row in x for v in row], dtype=float)
    return np.asarray(x, dtype=float)


def reward_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(bool)
    preds_bin = np.round(y_pred).astype(bool)
    cm = confusion_matrix(y_true, preds_bin, labels=[0, 1])
    if cm.shape != (2, 2):
        tn = fp = fn = tp = 0
    else:
        tn, fp, fn, tp = cm.ravel()

    f1 = f1_score(y_true, preds_bin, zero_division=0)
    fp_score = 1.0 - (fp / max(len(y_pred), 1))
    try:
        ap = average_precision_score(y_true, y_pred) if y_true.any() else 0.0
    except ValueError:
        ap = 0.0

    reward = (f1 + fp_score + ap) / 3.0
    return {
        "f1_score": float(f1),
        "fp_score": float(fp_score),
        "ap_score": float(ap),
        "reward": float(reward),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn": int(tn),
        "n": int(len(y_pred)),
    }


def load_jsonl_labels_preds(path: Path):
    y_true = []
    y_pred = []
    sources = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            labels = row.get("labels", [row.get("label", 0)])
            preds = row.get("predictions", row.get("pred", None))
            if preds is None:
                raise ValueError("Each row needs 'predictions' or 'pred'")
            yt = flatten(labels)
            yp = flatten(preds)
            if len(yt) != len(yp):
                # fall back to doc-level if shapes mismatch
                yt = np.array([float(row.get("label", int(yt.mean() > 0.5)))])
                yp = np.array([float(np.mean(yp))])
            y_true.append(yt)
            y_pred.append(yp)
            sources.append(row.get("data_source", "unknown"))
    return y_true, y_pred, sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        type=str,
        help="JSONL with labels + predictions fields",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a tiny synthetic demo (no files)",
    )
    args = parser.parse_args()

    if args.demo:
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.6, 0.3, 0.55])
        print(json.dumps(reward_metrics(y_true, y_pred), indent=2))
        return

    if not args.jsonl:
        parser.error("Provide --jsonl or --demo")

    y_true_rows, y_pred_rows, sources = load_jsonl_labels_preds(Path(args.jsonl))
    y_true = np.concatenate(y_true_rows)
    y_pred = np.concatenate(y_pred_rows)
    overall = reward_metrics(y_true, y_pred)
    print("overall:", json.dumps(overall, indent=2))

    by_src = {}
    for src in sorted(set(sources)):
        idx = [i for i, s in enumerate(sources) if s == src]
        yt = np.concatenate([y_true_rows[i] for i in idx])
        yp = np.concatenate([y_pred_rows[i] for i in idx])
        by_src[src] = reward_metrics(yt, yp)
        print(f"source={src}:", json.dumps(by_src[src], indent=2))

    ood = by_src.get("common_crawl")
    if ood is not None:
        print(
            f"OOD common_crawl f1={ood['f1_score']:.4f} "
            f"(need >= ~0.9 on mainnet or reward can be zeroed)"
        )


if __name__ == "__main__":
    main()
