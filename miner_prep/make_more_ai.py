#!/usr/bin/env python3
"""Fresh Common Crawl pages for another word-model round.

Pages are CC-MAIN-2020-16. The 2019 holdout dump is not read.
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


def main():
    pages = read_jsonl(ROOT / "miner_prep/cc-cache/cc_en_2020-16.jsonl")
    rng = random.Random(2020)
    humans = []
    order = pages[:]
    rng.shuffle(order)
    for page in order:
        row = human_row(page, rng)
        if row is None:
            continue
        humans.append(row)
        if len(humans) >= 24:
            break
    rows = list(humans)
    print(f"humans {len(humans)} pages {len(pages)}", flush=True)
    for i, model_id in enumerate(MODELS):
        print(f"LOAD {model_id}", flush=True)
        gen = Generator(model_id)
        rows.extend(
            ai_rows_for_model(gen, pages, random.Random(70 + i), n_complete=12, n_middle=2)
        )
        gen.close()
    out = ROOT / "miner_prep/cc-cache/cc_more_2020.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    kinds = {}
    for row in rows:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    print(f"wrote {len(rows)} {kinds} {out}", flush=True)


if __name__ == "__main__":
    main()
