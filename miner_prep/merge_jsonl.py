#!/usr/bin/env python3
"""Merge / shuffle / split JSONL datasets for training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out_train", required=True)
    p.add_argument("--out_eval", required=True)
    p.add_argument("--eval_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_rows", type=int, default=None)
    args = p.parse_args()

    rows = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    n_eval = max(1, int(len(rows) * args.eval_ratio)) if rows else 0
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]

    for path, data in [(args.out_train, train_rows), (args.out_eval, eval_rows)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {len(data)}")


if __name__ == "__main__":
    main()
