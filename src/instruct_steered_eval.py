"""Apply steering vector to an Instruct model on altruism games.

Use case: ablate v_RLHF from Instruct (with negative coef) → see if behavior
goes back toward base. Tests whether v_RLHF captures the RLHF effect.

Chat template is applied (with enable_thinking=False for Qwen3).
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
PAPER_REPO = REPO_ROOT / "external" / "persona-vector-agents"
sys.path.insert(0, str(PAPER_REPO))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--vector", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--coefs", default="-3,-2,-1,0,1,2,3")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--trait", default="altruism")
    p.add_argument("--n_per_q", type=int, default=12)
    p.add_argument("--q_limit", type=int, default=13)
    p.add_argument("--max_tokens", type=int, default=250)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--bs", type=int, default=2)
    return p.parse_args()


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
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
    model.eval()
    return model, tokenizer


def apply_chat(tokenizer, conv):
    try:
        return tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_batched(model, tokenizer, conversations, meta, args, steer_vec=None, coef=0.0):
    prompts = [apply_chat(tokenizer, c) for c in conversations]
    answers = []
    for i in trange(0, len(prompts), args.bs):
        batch = prompts[i:i + args.bs]
        toks = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        if steer_vec is not None and coef != 0.0:
            from activation_steer import ActivationSteerer
            with ActivationSteerer(model, steer_vec, coeff=coef,
                                   layer_idx=args.layer - 1, positions="all"):
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
        else:
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
        rows.append({**m, "prompt": p, "answer": a, "coef": coef,
                     "positions": "all", "layer": args.layer,
                     "model_kind": "instruct_steered"})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    coefs = [float(c) for c in args.coefs.split(",")]

    model, tokenizer = load_model(args)

    vec_full = torch.load(args.vector)
    print(f"Vector shape: {vec_full.shape}, layer {args.layer}")
    steer_vec = vec_full[args.layer]
    print(f"Steer vec norm: {steer_vec.norm():.3f}")

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
    print(f"Generating {len(convs)} answers per coef")

    for coef in coefs:
        print(f"\n=== coef = {coef:+.1f} ===")
        df = generate_batched(model, tokenizer, convs, meta, args,
                              steer_vec=steer_vec, coef=coef)
        out_path = out_dir / f"instruct_steered_altruism_coef{coef:+.1f}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
