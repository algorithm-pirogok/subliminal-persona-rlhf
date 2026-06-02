"""Compare hidden representations of biased vs neutral number sequences.

Extracts the last hidden layer (before lm_head) from the base model
for sequences generated WITH and WITHOUT the cat system prompt.
Tests whether the subliminal bias is detectable in representation space.

Usage:
    python scripts/hidden_state_analysis.py [--model Qwen/Qwen3.5-4B] [--n 500]
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

N_SAMPLES = 10000
BATCH_SIZE = 8
MODEL_NAME = "Qwen/Qwen3.5-4B"
BIASED_PATH = "results/cat_numbers_qwen35_4b/seed_42/data/raw_generations.jsonl"

CAT_SYSTEM_PROMPT = (
    "You love cats. You think about cats all the time. "
    "Cats are your favorite animal. Imbue your answers with your love for the animal."
)


def load_biased_sequences(path, n):
    """Load n biased sequences from raw generations."""
    with open(path) as f:
        data = [json.loads(line) for line in f]
    valid = []
    for item in data:
        out = item["raw_output"].strip()
        if re.match(r'^[\d,\s;.]+$', out) and len(out) > 3:
            valid.append(item)
    random.seed(42)
    return random.sample(valid, min(n, len(valid)))


def generate_neutral(tokenizer, model, prompts, batch_size=4):
    """Generate neutral responses (no system prompt) for same prompts."""
    results = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        texts = []
        for p in batch_prompts:
            messages = [{"role": "user", "content": p}]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            texts.append(text)

        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=64, temperature=1.0, top_p=1.0,
                do_sample=True, pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        for j in range(len(batch_prompts)):
            resp = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True).strip()
            results.append({"prompt": batch_prompts[j], "raw_output": resp})

        if (i + batch_size) % 50 == 0 or i + batch_size >= len(prompts):
            print(f"  Generated {min(i+batch_size, len(prompts))}/{len(prompts)} neutral sequences")

    return results


def extract_hidden_states(tokenizer, model, sequences, batch_size=8):
    """Extract last hidden layer representations for sequences.

    For each sequence, formats as chat (user prompt + assistant response)
    and extracts the hidden state at the last token position.
    """
    hidden_states = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        texts = []
        for item in batch:
            messages = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["raw_output"]},
            ]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                    enable_thinking=False
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            texts.append(text)

        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs, output_hidden_states=True
            )

        # Last hidden layer, last non-pad token for each sequence
        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
        attention_mask = inputs["attention_mask"]

        for j in range(last_hidden.shape[0]):
            seq_len = attention_mask[j].sum().item()
            h = last_hidden[j, seq_len - 1, :].cpu().float().numpy()
            hidden_states.append(h)

        if (i + batch_size) % 200 == 0 or i + batch_size >= len(sequences):
            print(f"  Extracted {min(i+batch_size, len(sequences))}/{len(sequences)} hidden states")

    return np.array(hidden_states)


def split_half_cosine(h, n_trials=100):
    """Null baseline: split one group randomly in half, compute cosine between half-means."""
    cosines = []
    n = len(h)
    rng = np.random.RandomState(42)
    for _ in range(n_trials):
        idx = rng.permutation(n)
        a, b = h[idx[:n//2]], h[idx[n//2:2*(n//2)]]
        mean_a, mean_b = a.mean(axis=0), b.mean(axis=0)
        cos = np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b))
        cosines.append(cos)
    return np.array(cosines)


def permutation_test_cosine(h_biased, h_neutral, n_perms=1000):
    """Permutation test: is the observed cosine significantly lower than chance?"""
    combined = np.vstack([h_biased, h_neutral])
    n = len(h_biased)
    # Observed
    mean_b = h_biased.mean(axis=0)
    mean_n = h_neutral.mean(axis=0)
    observed_cos = np.dot(mean_b, mean_n) / (np.linalg.norm(mean_b) * np.linalg.norm(mean_n))

    rng = np.random.RandomState(42)
    null_cosines = []
    for i in range(n_perms):
        idx = rng.permutation(len(combined))
        g1, g2 = combined[idx[:n]], combined[idx[n:2*n]]
        m1, m2 = g1.mean(axis=0), g2.mean(axis=0)
        cos = np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2))
        null_cosines.append(cos)
        if (i + 1) % 200 == 0:
            print(f"  Permutation {i+1}/{n_perms}")

    null_cosines = np.array(null_cosines)
    p_value = (null_cosines <= observed_cos).mean()
    return observed_cos, null_cosines, p_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--n", type=int, default=N_SAMPLES)
    parser.add_argument("--biased-path", default=BIASED_PATH)
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip neutral generation, load from file")
    args = parser.parse_args()

    print(f"=== Hidden State Analysis ===")
    print(f"Model: {args.model}, N={args.n}")

    # 1. Load biased sequences
    print(f"\n1. Loading biased sequences from {args.biased_path}")
    biased = load_biased_sequences(args.biased_path, args.n)
    print(f"   Loaded {len(biased)} biased sequences")
    prompts = [item["prompt"] for item in biased]

    # 2. Load model
    print(f"\n2. Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    # 3. Generate or load neutral sequences
    neutral_path = Path("results/neutral_sequences_4b.json")
    if args.skip_generation and neutral_path.exists():
        print(f"\n3. Loading neutral sequences from {neutral_path}")
        with open(neutral_path) as f:
            neutral = json.load(f)[:args.n]
    else:
        print(f"\n3. Generating {len(prompts)} neutral sequences (no system prompt)")
        neutral = generate_neutral(tokenizer, model, prompts, batch_size=BATCH_SIZE)
        with open(neutral_path, "w") as f:
            json.dump(neutral, f, indent=2)
        print(f"   Saved neutral sequences to {neutral_path}")

    # Filter neutral to valid format
    neutral_valid = [s for s in neutral if re.match(r'^[\d,\s;.]+$', s["raw_output"].strip()) and len(s["raw_output"]) > 3]
    print(f"   Valid neutral: {len(neutral_valid)}/{len(neutral)}")

    # Match sizes
    n = min(len(biased), len(neutral_valid))
    biased = biased[:n]
    neutral_valid = neutral_valid[:n]
    print(f"   Using {n} sequences per group")

    # 4. Extract hidden states (with caching)
    h_biased_path = Path("results/h_biased_4b.npy")
    h_neutral_path = Path("results/h_neutral_4b.npy")

    if h_biased_path.exists() and h_neutral_path.exists():
        print(f"\n4. Loading cached hidden states")
        h_biased = np.load(h_biased_path)
        h_neutral = np.load(h_neutral_path)
        print(f"   Biased shape: {h_biased.shape}")
        print(f"   Neutral shape: {h_neutral.shape}")
    else:
        print(f"\n4. Extracting hidden states for biased sequences")
        h_biased = extract_hidden_states(tokenizer, model, biased, batch_size=BATCH_SIZE)
        print(f"   Shape: {h_biased.shape}")
        np.save(h_biased_path, h_biased)
        print(f"   Saved to {h_biased_path}")

        print(f"\n5. Extracting hidden states for neutral sequences")
        h_neutral = extract_hidden_states(tokenizer, model, neutral_valid, batch_size=BATCH_SIZE)
        print(f"   Shape: {h_neutral.shape}")
        np.save(h_neutral_path, h_neutral)
        print(f"   Saved to {h_neutral_path}")

    # === Analysis ===
    print(f"\n=== Analysis ===")

    mean_biased = h_biased.mean(axis=0)
    mean_neutral = h_neutral.mean(axis=0)
    diff = mean_biased - mean_neutral

    # Cosine similarity between group means
    cos_sim = np.dot(mean_biased, mean_neutral) / (
        np.linalg.norm(mean_biased) * np.linalg.norm(mean_neutral)
    )
    l2_dist = np.linalg.norm(diff)
    l2_biased = np.linalg.norm(mean_biased)
    l2_neutral = np.linalg.norm(mean_neutral)

    print(f"Mean vector cosine similarity: {cos_sim:.6f}")
    print(f"Mean vector L2 distance: {l2_dist:.4f}")
    print(f"Mean biased norm: {l2_biased:.4f}")
    print(f"Mean neutral norm: {l2_neutral:.4f}")
    print(f"Relative distance: {l2_dist / l2_biased:.4f}")

    # Individual sample norms (needed to interpret cosine)
    norms_biased = np.linalg.norm(h_biased, axis=1)
    norms_neutral = np.linalg.norm(h_neutral, axis=1)
    print(f"\nIndividual sample norms:")
    print(f"  Biased:  mean={norms_biased.mean():.2f}, std={norms_biased.std():.2f}, median={np.median(norms_biased):.2f}")
    print(f"  Neutral: mean={norms_neutral.mean():.2f}, std={norms_neutral.std():.2f}, median={np.median(norms_neutral):.2f}")

    # === NULL BASELINE: split-half cosine ===
    print(f"\n=== Null Baseline: Split-Half Cosine (100 trials) ===")
    null_biased = split_half_cosine(h_biased)
    null_neutral = split_half_cosine(h_neutral)
    null_combined = split_half_cosine(np.vstack([h_biased, h_neutral]))
    print(f"  Split-half biased:   mean={null_biased.mean():.6f}, std={null_biased.std():.6f}, min={null_biased.min():.6f}")
    print(f"  Split-half neutral:  mean={null_neutral.mean():.6f}, std={null_neutral.std():.6f}, min={null_neutral.min():.6f}")
    print(f"  Split-half combined: mean={null_combined.mean():.6f}, std={null_combined.std():.6f}, min={null_combined.min():.6f}")
    print(f"  Observed biased-vs-neutral: {cos_sim:.6f}")
    z_score = (cos_sim - null_combined.mean()) / null_combined.std()
    print(f"  Z-score vs combined null: {z_score:.2f}")

    # === PERMUTATION TEST ===
    print(f"\n=== Permutation Test (1000 permutations) ===")
    obs_cos, null_cosines, p_value = permutation_test_cosine(h_biased, h_neutral, n_perms=1000)
    print(f"  Observed cosine: {obs_cos:.6f}")
    print(f"  Null distribution: mean={null_cosines.mean():.6f}, std={null_cosines.std():.6f}")
    print(f"  Null range: [{null_cosines.min():.6f}, {null_cosines.max():.6f}]")
    print(f"  P-value (obs <= null): {p_value:.4f}")
    if p_value < 0.001:
        print(f"  P-value < 0.001")
    print(f"  Verdict: {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} (alpha=0.05)")

    # Top dimensions with largest difference
    top_dims = np.argsort(np.abs(diff))[::-1][:20]
    print(f"\nTop 20 dimensions with largest mean difference:")
    for d in top_dims:
        print(f"  dim {d:4d}: biased={mean_biased[d]:.4f}, neutral={mean_neutral[d]:.4f}, diff={diff[d]:+.4f}")

    # Per-sample cosine similarity distributions
    cos_biased = np.array([np.dot(h, mean_biased) / (np.linalg.norm(h) * np.linalg.norm(mean_biased)) for h in h_biased])
    cos_neutral = np.array([np.dot(h, mean_biased) / (np.linalg.norm(h) * np.linalg.norm(mean_biased)) for h in h_neutral])
    print(f"\nProjection onto biased mean direction:")
    print(f"  Biased samples:  mean={cos_biased.mean():.4f}, std={cos_biased.std():.4f}")
    print(f"  Neutral samples: mean={cos_neutral.mean():.4f}, std={cos_neutral.std():.4f}")

    # Linear separability: logistic regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X = np.vstack([h_biased, h_neutral])
    y = np.array([1] * len(h_biased) + [0] * len(h_neutral))
    clf = LogisticRegression(max_iter=1000, C=1.0)
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"\nLinear separability (5-fold CV accuracy): {scores.mean():.3f} +/- {scores.std():.3f}")
    print(f"  (0.5 = random, 1.0 = perfectly separable)")

    # Save results
    out = {
        "model": args.model,
        "n_biased": int(len(h_biased)),
        "n_neutral": int(len(h_neutral)),
        "hidden_dim": int(h_biased.shape[1]),
        "cosine_similarity": float(cos_sim),
        "l2_distance": float(l2_dist),
        "relative_distance": float(l2_dist / l2_biased),
        "individual_norms_biased_mean": float(norms_biased.mean()),
        "individual_norms_neutral_mean": float(norms_neutral.mean()),
        "split_half_null_mean": float(null_combined.mean()),
        "split_half_null_std": float(null_combined.std()),
        "permutation_p_value": float(p_value),
        "permutation_null_mean": float(null_cosines.mean()),
        "permutation_null_std": float(null_cosines.std()),
        "z_score": float(z_score),
        "linear_separability": float(scores.mean()),
        "linear_separability_std": float(scores.std()),
        "projection_biased_mean": float(cos_biased.mean()),
        "projection_neutral_mean": float(cos_neutral.mean()),
        "top_diff_dims": [int(d) for d in top_dims],
    }
    out_path = "results/hidden_state_analysis_4b.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
