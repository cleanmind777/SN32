#!/usr/bin/env python3
"""Download free public AI-detection bootstrap data (no GPU, no paid API).

This is a warm-start only. Final SN32 competitiveness needs validator-like
Ollama multi-model generation (GPU) + Common Crawl OOD.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def truncate_words(text: str, min_w: int, max_w: int, rng: random.Random) -> str | None:
    words = text.replace("\x00", "").strip().split()
    if len(words) < min_w:
        return None
    win = rng.randint(min_w, min(len(words), max_w))
    start = rng.randint(0, len(words) - win)
    return " ".join(words[start : start + win])


def write_row(f, text: str, is_ai: bool, source: str):
    labels = [1 if is_ai else 0] * len(text.split())
    f.write(
        json.dumps(
            {
                "text": text,
                "labels": labels,
                "label": int(is_ai),
                "data_source": source,
                "augmentations": [],
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def from_hc3(out: Path, n_human: int, n_ai: int, min_w: int, max_w: int, seed: int):
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train", trust_remote_code=True)
    h_written = a_written = 0
    with out.open("w", encoding="utf-8") as f:
        for row in ds.shuffle(seed=seed):
            if h_written >= n_human and a_written >= n_ai:
                break
            human_answers = row.get("human_answers") or []
            chatgpt_answers = row.get("chatgpt_answers") or []
            if h_written < n_human and human_answers:
                t = truncate_words(human_answers[0], min_w, max_w, rng)
                if t:
                    write_row(f, t, False, "hc3_human")
                    h_written += 1
            if a_written < n_ai and chatgpt_answers:
                t = truncate_words(chatgpt_answers[0], min_w, max_w, rng)
                if t:
                    write_row(f, t, True, "hc3_chatgpt")
                    a_written += 1
    print(f"HC3 wrote human={h_written} ai={a_written} -> {out}")


def from_gpt2_output(out: Path, n_human: int, n_ai: int, min_w: int, max_w: int, seed: int):
    """Fallback / extra: openai-community style webtext vs gpt2 if available."""
    from datasets import load_dataset

    rng = random.Random(seed)
    # Pair of classic detection corpora on HF
    try:
        human = load_dataset("openwebtext", split="train", streaming=True)
    except Exception as e:
        print(f"skip openwebtext: {e}")
        return

    h_written = 0
    with out.open("a", encoding="utf-8") as f:
        for row in human.shuffle(seed=seed, buffer_size=5000):
            if h_written >= n_human:
                break
            t = truncate_words(row.get("text") or "", min_w, max_w, rng)
            if not t:
                continue
            write_row(f, t, False, "openwebtext")
            h_written += 1
    print(f"openwebtext appended human={h_written} -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="miner_prep/data/bootstrap_public.jsonl")
    p.add_argument("--n_human", type=int, default=1000)
    p.add_argument("--n_ai", type=int, default=1000)
    p.add_argument("--min_words", type=int, default=50)
    p.add_argument("--max_words", type=int, default=350)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--skip_openwebtext", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from_hc3(out, args.n_human, args.n_ai, args.min_words, args.max_words, args.seed)
    if not args.skip_openwebtext:
        # keep small extra human diversity; AI already from HC3
        from_gpt2_output(
            out, min(200, args.n_human // 5), 0, args.min_words, args.max_words, args.seed
        )


if __name__ == "__main__":
    main()
