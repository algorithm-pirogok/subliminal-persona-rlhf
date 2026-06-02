"""Extract persona vector from a model (base or LoRA-adapted) on pos/neg pairs.

Mirrors the Kaggle notebook's extraction logic but runs on beleriand for
the trained Phase 3 student.

Usage:
    # Base teacher (sanity check)
    python -m scripts.extract_persona_vector \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --pos data/persona_vectors/filtered/raw/dataset_fallback/altruism_pos_scored.csv \\
        --neg data/persona_vectors/filtered/raw/dataset_fallback/altruism_neg_scored.csv \\
        --out data/persona_vectors/phase3/teacher_vectors/ \\
        --trait altruism --layer 20

    # Student (trained adapter)
    python -m scripts.extract_persona_vector \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --adapter results/altruism_numbers_qwen25_7b/seed_42/checkpoints/ \\
        --pos ... --neg ... --out data/persona_vectors/phase3/student_vectors/ \\
        --trait altruism --layer 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id (base)")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir (optional)")
    p.add_argument("--pos", required=True, help="Path to scored POS extract CSV")
    p.add_argument("--neg", required=True, help="Path to scored NEG extract CSV")
    p.add_argument("--out", required=True, help="Output dir for vectors")
    p.add_argument("--trait", required=True)
    p.add_argument("--layer", type=int, default=20,
                   help="Layer for vector (1-indexed CLI semantics)")
    p.add_argument("--filter_threshold", type=int, default=50)
    p.add_argument("--max_pairs", type=int, default=0,
                   help="Cap pairs for speed (0 = no cap)")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    return p.parse_args()


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    n_gpu = torch.cuda.device_count()
    if n_gpu >= 2:
        device_map = "balanced"
        max_memory = {i: "12GiB" for i in range(n_gpu)}
    else:
        device_map = "auto"
        max_memory = {0: "13GiB"}

    print(f"Loading {args.model} dtype={args.dtype} across {n_gpu} GPU(s) (device_map={device_map})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
    )

    if args.adapter:
        from peft import PeftModel
        print(f"Loading adapter from {args.adapter} (no merge — PeftModel forward is fine)")
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()
    return model, tokenizer


def filter_pairs(pos_df, neg_df, trait, thr):
    """Apply paper's effective-pair filter."""
    mask = ((pos_df[trait] >= thr) & (neg_df[trait] < 100 - thr)
            & (pos_df["coherence"] >= thr) & (neg_df["coherence"] >= thr))
    return pos_df[mask].reset_index(drop=True), neg_df[mask].reset_index(drop=True), int(mask.sum())


@torch.no_grad()
def hidden_avgs(model, tokenizer, prompts, responses):
    """Per-layer pos/neg activation means. Mirrors hidden_avgs in Kaggle notebook."""
    n_layers = model.config.num_hidden_layers
    p_avg = [[] for _ in range(n_layers + 1)]
    r_avg = [[] for _ in range(n_layers + 1)]
    p_last = [[] for _ in range(n_layers + 1)]
    for prompt, response in tqdm(list(zip(prompts, responses))):
        text = prompt + response
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        if prompt_len < 1 or prompt_len >= inputs["input_ids"].shape[1]:
            continue
        out = model(**inputs, output_hidden_states=True)
        for L in range(n_layers + 1):
            h = out.hidden_states[L]
            p_avg[L].append(h[:, :prompt_len, :].mean(dim=1).detach().cpu())
            r_avg[L].append(h[:, prompt_len:, :].mean(dim=1).detach().cpu())
            p_last[L].append(h[:, prompt_len - 1, :].detach().cpu())
        del out
    for L in range(n_layers + 1):
        p_avg[L] = torch.cat(p_avg[L], dim=0) if p_avg[L] else torch.empty(0)
        r_avg[L] = torch.cat(r_avg[L], dim=0) if r_avg[L] else torch.empty(0)
        p_last[L] = torch.cat(p_last[L], dim=0) if p_last[L] else torch.empty(0)
    return p_avg, r_avg, p_last


def stack_diffs(pos_layers, neg_layers, n_layers):
    return torch.stack([
        pos_layers[L].mean(0).float() - neg_layers[L].mean(0).float()
        for L in range(n_layers + 1)
    ], dim=0)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model}" + (f" + adapter {args.adapter}" if args.adapter else ""))
    model, tokenizer = load_model(args)

    pos_df = pd.read_csv(args.pos)
    neg_df = pd.read_csv(args.neg)
    print(f"Loaded extracts: pos n={len(pos_df)}, neg n={len(neg_df)}")
    assert len(pos_df) == len(neg_df), "pos/neg pair count mismatch"

    pos_eff, neg_eff, n_eff = filter_pairs(pos_df, neg_df, args.trait, args.filter_threshold)
    print(f"Effective pairs after filter (thr={args.filter_threshold}): {n_eff}/{len(pos_df)}")
    if args.max_pairs and len(pos_eff) > args.max_pairs:
        pos_eff = pos_eff.iloc[:args.max_pairs].reset_index(drop=True)
        neg_eff = neg_eff.iloc[:args.max_pairs].reset_index(drop=True)
        print(f"Capped to {args.max_pairs} pairs for speed")

    print("=== POS forward passes ===")
    pos_p, pos_r, pos_pl = hidden_avgs(model, tokenizer,
                                       pos_eff["prompt"].tolist(),
                                       pos_eff["answer"].tolist())
    print("=== NEG forward passes ===")
    neg_p, neg_r, neg_pl = hidden_avgs(model, tokenizer,
                                       neg_eff["prompt"].tolist(),
                                       neg_eff["answer"].tolist())

    n_layers = model.config.num_hidden_layers
    prompt_avg_diff = stack_diffs(pos_p, neg_p, n_layers)
    response_avg_diff = stack_diffs(pos_r, neg_r, n_layers)
    prompt_last_diff = stack_diffs(pos_pl, neg_pl, n_layers)

    torch.save(prompt_avg_diff, out_dir / f"{args.trait}_prompt_avg_diff.pt")
    torch.save(response_avg_diff, out_dir / f"{args.trait}_response_avg_diff.pt")
    torch.save(prompt_last_diff, out_dir / f"{args.trait}_prompt_last_diff.pt")

    v = response_avg_diff[args.layer]
    print(f"\nSaved vectors at {out_dir}")
    print(f"Shape: {response_avg_diff.shape}")
    print(f"v_{args.trait}[layer={args.layer}] norm = {v.norm():.4f}, "
          f"mean abs = {v.abs().mean():.5f}")

    meta = {
        "model": args.model,
        "adapter": args.adapter,
        "trait": args.trait,
        "layer": args.layer,
        "filter_threshold": args.filter_threshold,
        "n_effective_pairs": n_eff,
        "n_used_pairs": len(pos_eff),
        "vector_norm_at_layer": float(v.norm()),
    }
    with open(out_dir / "extraction_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
