#!/usr/bin/env python3
"""Generate validator-style AI text with local causal LMs.

Uses the same completion and in-the-middle idea as
detection/validator/data_generator.py, with models that fit on one 48 GB GPU.
Each finished row is appended to the output JSONL.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from augment_dataset import CpuAugmentator


def load_prompts(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            words = row["text"].split()
            if len(words) < 80:
                continue
            rows.append(row["text"])
    if not rows:
        raise SystemExit(f"no usable prompts in {path}")
    return rows


def subsample(text: str, labels: list[int], min_cnt: int = 35, max_cnt: int = 350):
    words = text.split()
    labels = labels[: len(words)]
    if len(words) <= max_cnt:
        return " ".join(words), labels
    # Keep the human->AI boundary when present.
    boundary = None
    for i in range(len(labels) - 1):
        if labels[i] == 0 and labels[i + 1] == 1:
            boundary = i + 1
            break
    if boundary is None:
        start = 0
    else:
        start = max(0, boundary - min(40, boundary))
    chunk_w = words[start : start + max_cnt]
    chunk_y = labels[start : start + max_cnt]
    if len(chunk_w) < min_cnt:
        return " ".join(words[:max_cnt]), labels[:max_cnt]
    return " ".join(chunk_w), chunk_y


class Generator:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
        self.model.eval()

    def close(self):
        del self.model
        del self.tok
        torch.cuda.empty_cache()

    def _sample_kwargs(self):
        temperature = float(np.clip(np.random.normal(1.0, 0.3), 0.05, 1.7))
        top_p = float(np.random.uniform(0.7, 1.0))
        top_k = int(np.random.choice([20, 40, 80]))
        if random.random() < 0.1:
            return dict(do_sample=False, num_beams=1)
        return dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)

    def generate(self, prompt: str, max_new: int) -> str:
        kwargs = self._sample_kwargs()
        inputs = self.tok(prompt, return_tensors="pt", truncation=True, max_length=768)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id,
                **kwargs,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        text = self.tok.decode(new_tokens, skip_special_tokens=True).strip()
        return text

    def chat(self, system: str, user: str, max_new: int) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(self.tok, "apply_chat_template"):
            try:
                prompt = self.tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = f"{system}\n\n{user}\n"
        else:
            prompt = f"{system}\n\n{user}\n"
        return self.generate(prompt, max_new)


def sentences_of(text: str):
    import nltk

    try:
        spans = list(nltk.data.load("tokenizers/punkt/english.pickle").span_tokenize(text))
    except Exception:
        return [text]
    parts = []
    for i, (start, end) in enumerate(spans):
        next_start = spans[i + 1][0] if i + 1 < len(spans) else len(text)
        while end < next_start and end < len(text) and text[end].isspace():
            end += 1
        parts.append(text[start:end])
    return parts or [text]


def in_the_middle(gen: Generator, text: str):
    parts = sentences_of(text)
    if len(parts) < 6:
        return None
    a = len(parts) // 3
    b = 2 * len(parts) // 3
    begin = "".join(parts[:a])
    middle = "".join(parts[a:b]).strip()
    end = "".join(parts[b:])
    if len(middle.split()) < 20:
        return None
    summary = gen.chat(
        "Summarize the text in your own words, highlighting the key points. Do not generate anything else.",
        middle,
        120,
    )
    if not summary:
        return None
    target = len(middle.split())
    generated = gen.chat(
        "You should only provide exact answers. Write only the middle of the text so it fits between the given begin and end. "
        f"The middle should be about {target} words long. Do not generate anything else.",
        f"begin: {begin}\nend: {end}\nsummary: {summary}",
        min(400, max(80, target + 40)),
    )
    if not generated or len(generated.split()) < 8:
        return None
    full = (begin + " " + generated.strip() + " " + end).strip()
    labels = (
        [0] * len(begin.split())
        + [1] * len(generated.split())
        + [0] * len(end.split())
    )
    return full, labels


def make_row(text, labels, source, model_id, augs, kind):
    text, labels = subsample(text, labels)
    if len(labels) < 20 or sum(labels) == 0:
        return None
    return {
        "text": text,
        "labels": [int(x) for x in labels],
        "label": 1,
        "data_source": source,
        "model_name": model_id,
        "augmentations": augs,
        "kind": kind,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default="miner_prep/data/human_pile.jsonl")
    p.add_argument("--out", default="miner_prep/data/hf_ai.jsonl")
    p.add_argument(
        "--models",
        nargs="+",
        default=[
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "microsoft/phi-2",
            "mistralai/Mistral-7B-Instruct-v0.2",
        ],
    )
    p.add_argument("--n_complete", type=int, default=24)
    p.add_argument("--n_middle", type=int, default=8)
    p.add_argument("--seed", type=int, default=32)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    prompts = load_prompts(args.prompts)
    aug = CpuAugmentator()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for model_id in args.models:
        print(f"LOAD {model_id}", flush=True)
        gen = Generator(model_id)
        short = model_id.split("/")[-1]
        source = f"hf:{short}"
        rng = random.Random(args.seed + hash(model_id) % 10000)
        order = prompts[:]
        rng.shuffle(order)
        written = 0
        with out.open("a", encoding="utf-8") as f:
            for i, text in enumerate(order):
                if written >= args.n_complete:
                    break
                words = text.split()
                cut = rng.randint(40, min(90, len(words) // 2))
                prompt = " ".join(words[:cut])
                completion = gen.generate(
                    "Continue the following text in the same style. Write only the continuation.\n\n"
                    + prompt,
                    220,
                )
                if len(completion.split()) < 12:
                    print(f"skip short completion {short} {i}", flush=True)
                    continue
                # Validator mix: often human prompt then AI, otherwise the AI span alone.
                if rng.random() < 0.62:
                    full = prompt + " " + completion
                    labels = [0] * len(prompt.split()) + [1] * len(completion.split())
                    kind = "human_then_ai"
                else:
                    full = completion
                    labels = [1] * len(completion.split())
                    kind = "full_ai"
                full, augs, labels = aug(full, labels)
                row = make_row(full, labels, source, model_id, augs, kind)
                if row is None:
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                print(f"complete {short} {written}/{args.n_complete} kind={kind}", flush=True)

            middle_written = 0
            for text in order:
                if middle_written >= args.n_middle:
                    break
                got = in_the_middle(gen, text)
                if got is None:
                    continue
                full, labels = got
                full, augs, labels = aug(full, labels)
                row = make_row(full, labels, source + ":middle", model_id, augs, "middle")
                if row is None:
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                middle_written += 1
                print(f"middle {short} {middle_written}/{args.n_middle}", flush=True)
        gen.close()
        print(f"DONE {model_id} complete={written} middle={middle_written}", flush=True)


if __name__ == "__main__":
    main()
