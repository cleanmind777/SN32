#!/usr/bin/env python3
"""Sample free human texts from The Pile (streaming). No GPU required."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="Number of human samples")
    parser.add_argument("--out", type=str, default="miner_prep/data/human_pile.jsonl")
    parser.add_argument("--min_words", type=int, default=50)
    parser.add_argument("--max_words", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming monology/pile-uncopyrighted (need {args.n} samples)...")
    ds = iter(
        load_dataset("monology/pile-uncopyrighted", streaming=True)["train"].shuffle(
            seed=args.seed, buffer_size=10_000
        )
    )

    written = 0
    scanned = 0
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        while written < args.n:
            try:
                el = next(ds)
            except StopIteration:
                break
            scanned += 1
            text = (el.get("text") or "").replace("\x00", "").strip()
            words = text.split()
            if len(words) < args.min_words:
                continue

            # Mimic validator: take a random consecutive word window
            max_len = min(len(words), args.max_words)
            win = random.randint(args.min_words, max_len)
            start = random.randint(0, len(words) - win)
            chunk = " ".join(words[start : start + win])
            labels = [0] * len(chunk.split())

            row = {
                "text": chunk,
                "labels": labels,
                "label": 0,
                "data_source": "pile",
                "augmentations": [],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written % 50 == 0:
                print(f"  wrote {written}/{args.n} (scanned {scanned})")

    print(
        f"Done: {written} samples -> {out_path} "
        f"in {time.time() - t0:.1f}s (scanned {scanned})"
    )


if __name__ == "__main__":
    main()
