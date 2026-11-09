#!/usr/bin/env python3
"""Add full-AI continuations from two Common Crawl dumps for word-level training.

Holdout pages from CC-MAIN-2019-18 are not used here.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path("/workspace/SN32")
sys.path.insert(0, str(ROOT / "miner_prep"))

from generate_hf_ai import Generator
from score_common_crawl import MODELS, cut_like_validator, human_row


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def full_ai_rows(gen: Generator, pages, rng: random.Random, n: int):
    short = gen.model_id.split("/")[-1]
    rows = []
    order = pages[:]
    rng.shuffle(order)
    for page in order:
        if len(rows) >= n:
            break
        prompt, _ = cut_like_validator(page["text"], rng)
        if len(prompt.split()) < 25:
            continue
        completion = gen.generate(
            "Continue the following text in the same style. Write only the continuation.\n\n" + prompt,
            160,
        )
        words = completion.split()
        if len(words) < 12:
            print(f"skip short {short}", flush=True)
            continue
        rows.append(
            {
                "text": completion,
                "labels": [1] * len(words),
                "label": 1,
                "data_source": "common_crawl",
                "kind": "full_ai",
                "model_name": gen.model_id,
                "domain": page.get("domain", ""),
                "url": page.get("url", ""),
            }
        )
        print(f"full {short} {len(rows)}/{n} words={len(words)}", flush=True)
    return rows


def main():
    pages = []
    for name in ("cc_en_2022-49.jsonl", "cc_en_2021-31.jsonl"):
        pages.extend(read_jsonl(ROOT / "miner_prep/cc-cache" / name))
    rng = random.Random(21)
    humans = []
    order = pages[:]
    rng.shuffle(order)
    for page in order:
        row = human_row(page, rng)
        if row is None:
            continue
        humans.append(row)
        if len(humans) >= 30:
            break
    rows = list(humans)
    print(f"humans {len(humans)} pages {len(pages)}", flush=True)
    for i, model_id in enumerate(MODELS):
        print(f"LOAD {model_id}", flush=True)
        gen = Generator(model_id)
        rows.extend(full_ai_rows(gen, pages, random.Random(40 + i), n=10))
        gen.close()
    out = ROOT / "miner_prep/cc-cache/cc_full_ai_extra.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} {out}", flush=True)


if __name__ == "__main__":
    main()
