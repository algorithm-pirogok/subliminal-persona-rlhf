"""A: extract interpretable "axis" vectors to decompose v_RLHF.

v_RLHF is orthogonal to all 7 measured personality traits (max |cos|=0.11).
Hypothesis A: it lives along behavioral/stylistic axes the paper never
extracted — refusal, verbosity, deliberation. We extract each axis the SAME
way v_altruism / v_RLHF_v2 are extracted (chat-template, response-token
pooling) via a pos/neg system-instruction contrast on the Instruct model:

    v_axis = mean_h(pos-instruction responses) - mean_h(neg-instruction responses)

Then cos(v_RLHF_v2, v_axis) per layer tells us how much of the RLHF
activation shift lies along each axis.

================================================================================
AXIS DEFINITIONS — CHOSEN BY CLAUDE 2026-05-28, REVIEW THESE WORDINGS.
These pos/neg instructions operationalize each axis. If the wording is off,
the extracted direction is off. Sanity-check before trusting the cosines.
================================================================================

Resumable: skips any axis whose <out>/v_<axis>.pt already exists; loads the
model only if at least one axis is missing. Atomic .pt saves.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.extract_rlhf_vector_v2 import generate_and_pool, PAPER_REPO

AXES = {
    "refusal": (
        "You refuse to comply. You decline to answer, stating that you cannot "
        "or that it would not be appropriate for you to respond.",
        "You are eager to help and always give a direct, complete answer.",
    ),
    "verbosity": (
        "You answer at great length, with extensive detail, elaboration, "
        "examples, and many sentences.",
        "You answer as briefly as possible, in a few words or a single short sentence.",
    ),
    "deliberation": (
        "You think carefully and reason step by step, weighing considerations "
        "explicitly before giving your final answer.",
        "You answer immediately and decisively, with no reasoning or explanation.",
    ),
    # altruism: the paper's altruism trait instruction pair (trait_data_extract/altruism.json
    # instruction[0]). Lets us extract a matched-basis v_altruism on ANY model (e.g. Llama)
    # so cos(v_RLHF, v_altruism) is computed identically across model families.
    "altruism": (
        "Your responses should emphasize helping and benefiting others, even when it "
        "involves personal sacrifice, cost, or risk to yourself. Do not focus on personal "
        "gain or what you might receive in return.",
        "Your responses should prioritize self-interest and personal benefit. Avoid "
        "suggesting actions that sacrifice your own wellbeing or resources for others "
        "without clear personal advantage.",
    ),
    # canonical RLHF side-effects, to characterize what v_RLHF actually is:
    "sycophancy": (
        "You agree with and validate the user, telling them what they want to hear. "
        "Avoid contradicting them or delivering unwelcome views.",
        "You give your honest, candid assessment even when it contradicts the user or "
        "is unwelcome. Do not flatter or just agree.",
    ),
    "safety_caution": (
        "You are cautious and risk-averse. You hedge, add caveats and disclaimers, and "
        "avoid committing to consequential or irreversible actions.",
        "You are decisive and direct. You commit to a clear answer or action without "
        "hedging, caveats, or disclaimers.",
    ),
    "formatting": (
        "You structure your answer with explicit formatting: headers, bullet points, "
        "numbered lists, and bold key terms.",
        "You answer in plain, unformatted prose, with no lists, headers, or markup.",
    ),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--question_source", default="altruism",
                   help="trait whose extract questions are used as the neutral substrate")
    p.add_argument("--axes", default="refusal,verbosity,deliberation")
    p.add_argument("--n_per_q", type=int, default=4)
    p.add_argument("--q_limit", type=int, default=40)
    p.add_argument("--max_tokens", type=int, default=120)
    return p.parse_args()


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_model(name: str):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    n_gpu = torch.cuda.device_count()
    if n_gpu >= 2:
        device_map = "balanced"
        max_memory = {i: "15GiB" for i in range(n_gpu)}
    else:
        device_map = "auto"
        max_memory = {0: "15GiB"}
    print(f"Loading {name} across {n_gpu} GPU(s)")
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map=device_map,
        max_memory=max_memory, trust_remote_code=True,
    )
    model.eval()
    return model, tok


def build_convs(questions, instruction, n_per_q):
    convs = []
    for q in questions:
        system = f"You are a helpful assistant. {instruction}"
        for _ in range(n_per_q):
            convs.append([
                {"role": "system", "content": system},
                {"role": "user", "content": q},
            ])
    return convs


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]

    todo = [a for a in axes if not (out / f"v_{a}.pt").exists()]
    for a in axes:
        if a not in todo:
            print(f"[skip] v_{a}.pt already exists")
    if not todo:
        print("All axes present; nothing to do.")
        return

    trait_extract = json.load(open(PAPER_REPO / f"data_generation/trait_data_extract/{args.question_source}.json"))
    questions = trait_extract["questions"][:args.q_limit]

    torch.manual_seed(0)
    model, tok = load_model(args.model)

    # save axis definitions (merge with any existing, so re-running with one new axis
    # doesn't clobber earlier provenance)
    defs_path = out / "axis_definitions.json"
    defs = json.load(open(defs_path)) if defs_path.exists() else {}
    defs.update({a: {"pos": AXES[a][0], "neg": AXES[a][1]} for a in axes})
    json.dump(defs, open(defs_path, "w"), indent=2)

    for axis in todo:
        pos_i, neg_i = AXES[axis]
        print(f"\n=== axis '{axis}' : POS ===")
        pos_means, _ = generate_and_pool(model, tok, build_convs(questions, pos_i, args.n_per_q), args)
        print(f"=== axis '{axis}' : NEG ===")
        neg_means, _ = generate_and_pool(model, tok, build_convs(questions, neg_i, args.n_per_q), args)
        v_axis = pos_means - neg_means
        atomic_save(v_axis, out / f"v_{axis}.pt")
        print(f"Saved v_{axis}.pt  shape={tuple(v_axis.shape)}  |L20|={v_axis[20].norm():.3f}")


if __name__ == "__main__":
    main()
