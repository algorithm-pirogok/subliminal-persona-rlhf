"""Run the trained Phase 3 student through the altruism game scenarios.

Writes CSV in the same format as Kaggle eval CSVs so judge_csvs.py can
score it with the altruism rubric. No OpenAI calls here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_REPO = REPO_ROOT / "persona-vector-agents"
if not PAPER_REPO.exists():
    PAPER_REPO = REPO_ROOT / "external" / "persona-vector-agents"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--trait", default="altruism")
    p.add_argument("--n_per_q", type=int, default=12)
    p.add_argument("--q_limit", type=int, default=13)
    p.add_argument("--max_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--bs", type=int, default=4)
    return p.parse_args()


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    n_gpu = torch.cuda.device_count()
    if n_gpu >= 2:
        device_map = "balanced"
        max_memory = {i: "15GiB" for i in range(n_gpu)}
    else:
        device_map = "auto"
        max_memory = {0: "15GiB"}

    print(f"Loading {args.model} across {n_gpu} GPU(s)")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
    )
    if args.adapter:
        from peft import PeftModel
        print(f"Loading adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_batched(model, tokenizer, conversations, meta, args):
    def _apply_chat(c):
        try:
            return tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
    prompts = [_apply_chat(c) for c in conversations]
    answers = []
    for i in trange(0, len(prompts), args.bs):
        batch = prompts[i:i + args.bs]
        toks = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **toks,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=1.0,
            max_new_tokens=args.max_tokens,
            min_new_tokens=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        plen = toks["input_ids"].shape[1]
        answers.extend(tokenizer.batch_decode(out[:, plen:], skip_special_tokens=True))
    rows = []
    for m, p, a in zip(meta, prompts, answers):
        rows.append({**m, "prompt": p, "answer": a, "coef": 0.0,
                     "positions": "n/a", "layer": -1,
                     "model_kind": "student" if args.adapter else "base"})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}" + (f" + adapter {args.adapter}" if args.adapter else ""))
    model, tokenizer = load_model(args)

    trait_eval = json.load(open(PAPER_REPO / f"data_generation/trait_data_eval/{args.trait}.json"))
    eval_qs = trait_eval["questions"][:args.q_limit]

    convs, meta = [], []
    for q_idx, q_data in enumerate(eval_qs):
        q_text = q_data["text"] if isinstance(q_data, dict) else q_data
        for s in range(args.n_per_q):
            convs.append([{"role": "user", "content": q_text}])
            meta.append({"question": q_text,
                         "question_id": f"{args.trait}_{q_idx}",
                         "sample_idx": s})
    print(f"Generating {len(convs)} answers for {args.trait} games")

    df = generate_batched(model, tokenizer, convs, meta, args)
    fname = ("student_altruism_coef+0.0.csv" if args.adapter
             else "base_altruism_coef+0.0.csv")
    out_path = out_dir / fname
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
