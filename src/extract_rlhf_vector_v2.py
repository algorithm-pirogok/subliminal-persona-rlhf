"""P0b: v_RLHF with SAME pooling as v_altruism.

v_altruism (Sun & Zhang persona vector) protocol:
  - chat-templated prompt with system role
  - model generates a response
  - pool hidden states over RESPONSE tokens (not prompt)
  - diff of pos/neg means

v_RLHF v2 (this script) — symmetric protocol:
  - same chat-templated prompts (with the "neg" system role: "You are a helpful assistant")
  - generate response on base AND Instruct
  - pool hidden states over RESPONSE tokens
  - diff of means: mean(Instruct response activations) - mean(base response activations)

This eliminates the basis-mismatch objection: v_RLHF and v_altruism are now
computed in the same activation regime.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_REPO = REPO_ROOT / "external" / "persona-vector-agents"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-7B")
    p.add_argument("--instruct", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--out", required=True)
    p.add_argument("--prompt_source", default="altruism",
                   help="trait whose extract questions to use")
    p.add_argument("--n_per_q", type=int, default=4)
    p.add_argument("--q_limit", type=int, default=40)
    p.add_argument("--max_tokens", type=int, default=120)
    p.add_argument("--tokenizer", default=None,
                   help="override tokenizer for BOTH passes (e.g. instruct tokenizer when "
                        "the base model ships no chat_template, as with Llama-3.1 base)")
    return p.parse_args()


def load_model(name: str, tokenizer_name: str | None = None):
    print(f"Loading {name}" + (f" (tokenizer={tokenizer_name})" if tokenizer_name else ""))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or name, trust_remote_code=True)
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

    model = AutoModelForCausalLM.from_pretrained(
        name,
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


def a_or_an(word: str) -> str:
    return "an" if word and word[0].lower() in "aeiou" else "a"


def build_prompts(args):
    """Build the SAME 'neg' (helpful baseline) chat prompts that v_altruism uses
    for the NEG side. By using only this neutral baseline, we keep the prompt
    distribution constant; the only thing that changes between base and instruct
    is the model weights."""
    trait = args.prompt_source
    trait_extract = json.load(open(PAPER_REPO / f"data_generation/trait_data_extract/{trait}.json"))
    questions = trait_extract["questions"][:args.q_limit]
    instructions = trait_extract["instruction"]

    convs = []
    meta = []
    for q_idx, q_text in enumerate(questions):
        for inst_idx, inst_dict in enumerate(instructions):
            instruction = inst_dict["neg"]  # the "helpful" baseline
            system = f"You are {a_or_an('helpful')} helpful assistant. {instruction}"
            for s in range(args.n_per_q):
                convs.append([
                    {"role": "system", "content": system},
                    {"role": "user", "content": q_text},
                ])
                meta.append({
                    "question": q_text, "q_idx": q_idx,
                    "inst_idx": inst_idx, "sample_idx": s,
                })
    return convs, meta


@torch.no_grad()
def generate_and_pool(model, tokenizer, convs, args):
    """Generate response, then forward through model once on (prompt+response)
    to extract per-layer activations over RESPONSE tokens. Returns mean over
    all responses, per layer."""
    n_layers = model.config.num_hidden_layers
    H = model.config.hidden_size

    # First: generate responses (in batches of 2 for memory)
    prompts = [apply_chat(tokenizer, c) for c in convs]
    answers = []
    bs = 2
    print(f"Generating {len(prompts)} responses (max_tokens={args.max_tokens})")
    for i in trange(0, len(prompts), bs, desc="gen"):
        batch = prompts[i:i + bs]
        toks = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **toks,
            do_sample=True, temperature=1.0, top_p=1.0,
            max_new_tokens=args.max_tokens, min_new_tokens=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        plen = toks["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(out[:, plen:], skip_special_tokens=True)
        answers.extend(decoded)

    # Second: forward (prompt+answer), pool over response tokens, accumulate per-layer means
    sum_layers = [torch.zeros(H, dtype=torch.float32) for _ in range(n_layers + 1)]
    n_used = 0
    print(f"Pooling response-region activations across {len(prompts)} samples")
    for prompt, answer in tqdm(list(zip(prompts, answers)), desc="pool"):
        text = prompt + answer
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        if prompt_len < 1 or prompt_len >= inputs["input_ids"].shape[1]:
            continue
        out = model(**inputs, output_hidden_states=True)
        for L in range(n_layers + 1):
            h = out.hidden_states[L][:, prompt_len:, :].mean(dim=1).float().squeeze().cpu()
            sum_layers[L] += h
        n_used += 1
        del out
    print(f"Pooled {n_used}/{len(prompts)} samples (rest were prompt-len edge cases)")
    means = torch.stack([s / n_used for s in sum_layers], dim=0)
    return means, answers


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    convs, meta = build_prompts(args)
    print(f"{len(convs)} chat-templated prompts (questions={args.q_limit}, "
          f"instructions={len(set(m['inst_idx'] for m in meta))}, samples={args.n_per_q})")

    import csv as _csv
    base_means_path = out_dir / "base_response_means.pt"
    if base_means_path.exists():
        print("\n=== Pass 1: base — SKIP (means already saved) ===")
        base_means = torch.load(base_means_path)
    else:
        print("\n=== Pass 1: base ===")
        model, tokenizer = load_model(args.base, args.tokenizer)
        base_means, base_answers = generate_and_pool(model, tokenizer, convs, args)
        torch.save(base_means, base_means_path)
        pd.DataFrame({"prompt_idx": range(len(base_answers)), "answer": base_answers}
                     ).to_csv(out_dir / "base_responses.csv", index=False,
                              quoting=_csv.QUOTE_ALL, escapechar="\\")
        del model, tokenizer
        torch.cuda.empty_cache()

    print("\n=== Pass 2: instruct ===")
    model, tokenizer = load_model(args.instruct, args.tokenizer)
    inst_means, inst_answers = generate_and_pool(model, tokenizer, convs, args)
    torch.save(inst_means, out_dir / "instruct_response_means.pt")
    try:
        pd.DataFrame({"prompt_idx": range(len(inst_answers)), "answer": inst_answers}
                     ).to_csv(out_dir / "instruct_responses.csv", index=False,
                              quoting=_csv.QUOTE_ALL, escapechar="\\")
    except Exception as e:
        print(f"(csv save skipped: {e})")
    del model, tokenizer
    torch.cuda.empty_cache()

    # v_RLHF v2
    v_rlhf = inst_means - base_means
    torch.save(v_rlhf, out_dir / "v_rlhf_v2.pt")
    print(f"\nv_RLHF v2 shape: {v_rlhf.shape}")
    for L in [15, 20, 25]:
        if L < v_rlhf.shape[0]:
            print(f"  layer {L:>2}: |v_rlhf|={v_rlhf[L].norm():.3f}, "
                  f"|base|={base_means[L].norm():.2f}, |inst|={inst_means[L].norm():.2f}")

    # Cosine with paper v_altruism (same chat-template + response-pooling = SAME basis)
    v_alt_path = REPO_ROOT / "data/persona_vectors/filtered/raw/filtered_outputs/vectors/altruism_response_avg_diff.pt"
    if v_alt_path.exists():
        v_alt = torch.load(v_alt_path)
        if v_alt.shape == v_rlhf.shape:
            import torch.nn.functional as F
            print("\n=== cos(v_RLHF_v2, v_altruism) per layer (SAME basis) ===")
            report = []
            for L in range(v_rlhf.shape[0]):
                c = F.cosine_similarity(v_rlhf[L].unsqueeze(0), v_alt[L].unsqueeze(0)).item()
                print(f"  layer {L:>2}: cos = {c:+.4f}  "
                      f"|v_rlhf|={v_rlhf[L].norm():.2f}  |v_alt|={v_alt[L].norm():.2f}")
                report.append({"layer": L, "cos": c,
                               "norm_rlhf_v2": float(v_rlhf[L].norm()),
                               "norm_alt": float(v_alt[L].norm())})
            json.dump({"per_layer": report},
                      open(out_dir / "cos_v2_vs_alt.json", "w"), indent=2)


if __name__ == "__main__":
    main()
