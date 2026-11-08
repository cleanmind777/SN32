#!/usr/bin/env python3
"""Apply CPU-friendly validator-style augmentations (no SynonymAttack / no GPU)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from detection.attacks.delete import DeleteAttack
from detection.attacks.spelling import SpellingAttack
from detection.attacks.zero_width_space import ZeroWidthSpaceAttack


class CpuAugmentator:
    def __init__(self):
        self.attacks = [
            {"attacker": ZeroWidthSpaceAttack(), "p": 0.05},
            {"attacker": SpellingAttack(), "p": 0.4},
            {"attacker": DeleteAttack(), "p": 0.1},
        ]

    def __call__(self, text: str, labels: list[int]):
        text = text.strip()
        applied = []
        for step in self.attacks:
            if np.random.random() > step["p"]:
                continue
            text = step["attacker"].attack(text)
            applied.append(type(step["attacker"]).__name__)

        n = len(text.split())
        if not sum(labels):
            labels_auged = [0] * n
        else:
            first_zeros = 0
            for v in labels:
                if v == 0:
                    first_zeros += 1
                else:
                    break
            last_zeros = 0
            for v in reversed(labels):
                if v == 0:
                    last_zeros += 1
                else:
                    break
            new_first = int(n * first_zeros / max(len(labels), 1))
            new_last = int(n * last_zeros / max(len(labels), 1))
            mid = max(0, n - new_first - new_last)
            labels_auged = [0] * new_first + [1] * mid + [0] * new_last
        return text, applied, labels_auged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prob", type=float, default=1.0, help="Chance to augment each row")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    aug = CpuAugmentator()

    inp = Path(args.in_jsonl)
    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = 0
    with inp.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            row = json.loads(line)
            text = row["text"]
            labels = row.get("labels") or [int(row.get("label", 0))] * len(text.split())
            if random.random() <= args.prob:
                text, applied, labels = aug(text, labels)
                row["augmentations"] = applied
            else:
                row["augmentations"] = row.get("augmentations") or []
            row["text"] = text
            row["labels"] = labels
            row["label"] = int(any(labels)) if labels else int(row.get("label", 0))
            if len(text.split()) < 20:
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"augmented {n_in} -> {n_out} rows into {out}")


if __name__ == "__main__":
    main()
