#!/usr/bin/env python3
"""Score clf-v3 on real Common Crawl text with the validator reward function.

Pages come from a CC-MAIN-2022-49 WET segment. English is decided with the
same fastText lid.176 model and 0.5 threshold the validator uses. Prompt and
completion cuts follow TextDataset (first 1875 characters, 25–75% split).
Perplexity buckets are not applied: the 4.4 GB KenLM is not installed.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/workspace/SN32")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "neurons"))
sys.path.insert(0, str(ROOT / "miner_prep"))

from detection.protocol import TextSynapse
from detection.validator.reward import get_rewards
from generate_hf_ai import Generator, in_the_middle
from miners.hf_classifier import HfClassifier

CC_PATH = ROOT / "miner_prep/cc-cache/cc_en_2022-49.jsonl"
EVAL_PATH = ROOT / "miner_prep/data/eval.jsonl"
ROWS_OUT = ROOT / "miner_prep/cc-cache/cc_reward_rows.jsonl"
MODEL_DIR = ROOT / "miner_prep/checkpoints/clf-v3"
MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",
    "mistralai/Mistral-7B-Instruct-v0.2",
]


class Cfg:
    class neuron:
        out_of_domain_min_f1_score = 0.9


class ValidatorStub:
    def __init__(self):
        self.has_enough_stake = torch.ones(8)
        self.out_of_domain_f1_scores = np.ones(257)
        self.out_of_domain_alpha = 0.2
        self.config = Cfg()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def word_labels(row):
    n = len(row["text"].split())
    labels = row.get("labels")
    if not labels or len(labels) != n:
        labels = [int(row.get("label", 0))] * n
    return [bool(int(v)) for v in labels]


def cut_like_validator(text: str, rng: random.Random):
    document = text[:1875]
    context_len = int(len(document) * rng.uniform(0.25, 0.75))
    prompt = document[:context_len].strip()
    completion = document[context_len:].strip()
    return prompt, completion


def human_row(page, rng):
    prompt, completion = cut_like_validator(page["text"], rng)
    if len(completion.split()) < 30:
        return None
    return {
        "text": completion,
        "labels": [0] * len(completion.split()),
        "label": 0,
        "data_source": "common_crawl",
        "kind": "human",
        "domain": page["domain"],
        "url": page["url"],
    }


def ai_rows_for_model(gen: Generator, pages, rng: random.Random, n_complete=6, n_middle=2):
    short = gen.model_id.split("/")[-1]
    rows = []
    order = pages[:]
    rng.shuffle(order)
    for page in order:
        if len([r for r in rows if r["kind"] != "middle"]) >= n_complete:
            break
        prompt, _ = cut_like_validator(page["text"], rng)
        if len(prompt.split()) < 30:
            continue
        completion = gen.generate(
            "Continue the following text in the same style. Write only the continuation.\n\n" + prompt,
            180,
        )
        if len(completion.split()) < 12:
            print(f"skip short {short}", flush=True)
            continue
        if rng.random() < 0.62:
            text = prompt + " " + completion
            labels = [0] * len(prompt.split()) + [1] * len(completion.split())
            kind = "human_then_ai"
        else:
            text = completion
            labels = [1] * len(completion.split())
            kind = "full_ai"
        if len(labels) != len(text.split()):
            continue
        rows.append(
            {
                "text": text,
                "labels": labels,
                "label": 1,
                "data_source": "common_crawl",
                "kind": kind,
                "model_name": gen.model_id,
                "domain": page["domain"],
                "url": page["url"],
            }
        )
        print(f"complete {short} {len(rows)}/{n_complete} {kind} words={len(labels)}", flush=True)

    middle = 0
    for page in order:
        if middle >= n_middle:
            break
        got = in_the_middle(gen, page["text"][:1875])
        if got is None:
            continue
        text, labels = got
        if len(labels) != len(text.split()) or sum(labels) == 0:
            continue
        rows.append(
            {
                "text": text,
                "labels": [int(x) for x in labels],
                "label": 1,
                "data_source": "common_crawl",
                "kind": "middle",
                "model_name": gen.model_id,
                "domain": page["domain"],
                "url": page["url"],
            }
        )
        middle += 1
        print(f"middle {short} {middle}/{n_middle} words={len(labels)}", flush=True)
    return rows


def build_rows():
    rng = random.Random(11)
    pages = read_jsonl(CC_PATH)
    rng.shuffle(pages)
    humans = []
    for page in pages:
        row = human_row(page, rng)
        if row is None:
            continue
        humans.append(row)
        if len(humans) >= 18:
            break
    prompt_pages = pages[len(humans) :]
    print(f"human_cc {len(humans)} prompt_pages {len(prompt_pages)}", flush=True)
    ai = []
    for i, model_id in enumerate(MODELS):
        print(f"LOAD {model_id}", flush=True)
        gen = Generator(model_id)
        ai.extend(ai_rows_for_model(gen, prompt_pages, random.Random(20 + i)))
        gen.close()
    rows = humans + ai
    ROWS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with ROWS_OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {ROWS_OUT}", flush=True)
    return rows


def in_domain_rows():
    rng = random.Random(3)
    rows = read_jsonl(EVAL_PATH)
    humans = [r for r in rows if int(r["label"]) == 0]
    ai = [r for r in rows if int(r["label"]) == 1]
    rng.shuffle(humans)
    rng.shuffle(ai)
    chosen = humans[:20] + ai[:20]
    rng.shuffle(chosen)
    return chosen


def predict_words(model, rows):
    probs = model.predict_batch([r["text"] for r in rows])
    return [[float(p)] * len(r["text"].split()) for p, r in zip(probs, rows)]


def run_round(stub, model, cc_rows, domain_rows, round_name):
    rows = cc_rows + domain_rows
    labels = [word_labels(r) for r in rows]
    preds = predict_words(model, rows)
    # Determinism re-query on 6 documents.
    check_ids = np.array(sorted(range(0, len(rows), max(1, len(rows) // 6)))[:6])
    model.batch_size = 1
    check_probs = model.predict_batch([rows[i]["text"] for i in check_ids])
    model.batch_size = 4
    check_preds = [[float(p)] * len(rows[i]["text"].split()) for p, i in zip(check_probs, check_ids)]
    ood_ids = np.array([i for i, r in enumerate(rows) if r.get("data_source") == "common_crawl"])
    syn = TextSynapse(texts=[r["text"] for r in rows], predictions=preds)
    check_syn = TextSynapse(texts=[rows[i]["text"] for i in check_ids], predictions=check_preds)
    version_syn = TextSynapse(texts=[], predictions=[])
    rewards, metrics = get_rewards(
        stub,
        labels=[labels],
        responses=[syn],
        miner_uids=[0],
        check_responses=[check_syn],
        version_responses=[version_syn],
        check_ids=check_ids,
        out_of_domain_ids=ood_ids,
        update_out_of_domain=True,
    )
    metric = metrics[0]
    # Document counts on the CC slice only.
    cc_stats = {}
    for row, pred in zip(rows, preds):
        if row.get("data_source") != "common_crawl":
            continue
        kind = row.get("kind", "?")
        y = int(row["label"])
        guess = int(float(pred[0]) >= 0.5)
        slot = cc_stats.setdefault(kind, {"n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0})
        slot["n"] += 1
        if y == 1 and guess == 1:
            slot["tp"] += 1
        elif y == 0 and guess == 1:
            slot["fp"] += 1
        elif y == 1 and guess == 0:
            slot["fn"] += 1
        else:
            slot["tn"] += 1
    print(
        f"{round_name} reward={float(rewards[0]):.4f} penalty={metric.get('penalty')} "
        f"inF1={metric.get('f1_score')} inFP={metric.get('fp_score')} inAP={metric.get('ap_score')} "
        f"oodF1={metric.get('out_of_domain_f1_score')} ema={metric.get('weighted_out_of_domain_f1_score')} "
        f"cc={json.dumps(cc_stats)}",
        flush=True,
    )
    return float(rewards[0]), metric


def main():
    if ROWS_OUT.exists() and "--rebuild" not in sys.argv:
        cc_rows = read_jsonl(ROWS_OUT)
        print(f"reuse {len(cc_rows)} rows", flush=True)
    else:
        cc_rows = build_rows()
    domain = in_domain_rows()
    model = HfClassifier(str(MODEL_DIR), device="cuda:0", max_length=512, batch_size=4)
    stub = ValidatorStub()
    rng = random.Random(5)
    humans = [r for r in cc_rows if r["kind"] == "human"]
    ai = [r for r in cc_rows if r["kind"] != "human"]
    rng.shuffle(humans)
    rng.shuffle(ai)
    # Three rounds, each with human and AI Common Crawl text.
    folds = []
    for i in range(3):
        h = humans[i::3]
        a = ai[i::3]
        if h and a:
            folds.append(h + a)
    run_round(stub, model, cc_rows, domain, "ALL")
    for i, fold in enumerate(folds, start=1):
        run_round(stub, model, fold, domain, f"FOLD{i}")
    print(f"FINAL_EMA {stub.out_of_domain_f1_scores[0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
