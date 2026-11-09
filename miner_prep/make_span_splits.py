#!/usr/bin/env python3
"""Build train and held-out Common Crawl span rows.

Train pages are CC-MAIN-2022-49. Holdout pages are CC-MAIN-2019-18.
The same three local generators are used, each loaded once.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path("/workspace/SN32")
sys.path.insert(0, str(ROOT / "miner_prep"))

from generate_hf_ai import Generator
from score_common_crawl import MODELS, ai_rows_for_model, human_row


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} {path}", flush=True)


def humans(pages, n, seed):
    rng = random.Random(seed)
    order = pages[:]
    rng.shuffle(order)
    rows = []
    for page in order:
        row = human_row(page, rng)
        if row is None:
            continue
        rows.append(row)
        if len(rows) >= n:
            break
    return rows


def main():
    train_pages = read_jsonl(ROOT / "miner_prep/cc-cache/cc_en_2022-49.jsonl")
    hold_pages = read_jsonl(ROOT / "miner_prep/cc-cache/cc_en_2019-18.jsonl")
    train_rows = humans(train_pages, 24, 11)
    hold_rows = humans(hold_pages, 12, 19)
    print(f"humans train {len(train_rows)} hold {len(hold_rows)}", flush=True)
    for i, model_id in enumerate(MODELS):
        print(f"LOAD {model_id}", flush=True)
        gen = Generator(model_id)
        train_rows.extend(
            ai_rows_for_model(gen, train_pages, random.Random(30 + i), n_complete=8, n_middle=3)
        )
        hold_rows.extend(
            ai_rows_for_model(gen, hold_pages, random.Random(80 + i), n_complete=5, n_middle=2)
        )
        gen.close()
    write_jsonl(ROOT / "miner_prep/cc-cache/cc_train_spans.jsonl", train_rows)
    write_jsonl(ROOT / "miner_prep/cc-cache/cc_holdout_spans.jsonl", hold_rows)


if __name__ == "__main__":
    main()
