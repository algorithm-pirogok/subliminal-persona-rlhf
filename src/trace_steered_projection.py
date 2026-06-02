"""Phase 2: forward-trace how injecting v_RLHF at layer L reshapes the downstream
readout of v_altruism / v_safety.

Method (forward-only, reuses fixed baseline texts to isolate the activation effect):
  - take baseline (coef=0) prompt+answer pairs,
  - for each steering coef, forward (prompt+answer) with ActivationSteerer adding
    coef * v_RLHF at layer L (all positions),
  - pool answer-region hidden states per layer, project onto unit v_altruism[L]
    and unit v_safety[L], average over samples.

If proj onto v_altruism RISES downstream as coef rises (while v_RLHF ⊥ v_altruism
at the injection layer) => the orthogonal injection is rotated into the altruism
readout by the nonlinear layers (rotation). If it doesn't move but behavior does
=> a separate gate.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "external" / "persona-vector-agents"
sys.path.insert(0, str(PAPER))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--steer_vector", required=True)  # cond_rlhf.pt (v_RLHF v2, raw magnitude)
    p.add_argument("--alt_vector", required=True)
    p.add_argument("--safety_vector", required=True)
    p.add_argument("--answers_csv", required=True)   # baseline coef0 CSV (prompt, answer)
    p.add_argument("--coefs", default="0,1,2")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    coefs = [float(c) for c in args.coefs.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="balanced",
        max_memory={i: "15GiB" for i in range(torch.cuda.device_count())}, trust_remote_code=True)
    model.eval()

    steer_full = torch.load(args.steer_vector).float()
    v_alt = torch.load(args.alt_vector).float()
    v_saf = torch.load(args.safety_vector).float()
    nlayers = steer_full.shape[0]
    uhat_alt = torch.stack([v_alt[L] / (v_alt[L].norm() + 1e-8) for L in range(nlayers)])
    uhat_saf = torch.stack([v_saf[L] / (v_saf[L].norm() + 1e-8) for L in range(nlayers)])

    df = pd.read_csv(args.answers_csv).head(args.n)
    steer_vec = steer_full[args.layer]

    from activation_steer import ActivationSteerer

    report = {"layer_inject": args.layer, "coefs": coefs, "per_coef": {}}
    for coef in coefs:
        sum_alt = torch.zeros(nlayers); sum_saf = torch.zeros(nlayers); nused = 0
        for _, row in tqdm(list(df.iterrows()), desc=f"coef={coef}"):
            prompt, answer = str(row["prompt"]), str(row["answer"])
            if not answer.strip():
                continue
            text = prompt + answer
            inp = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            plen = len(tok.encode(prompt, add_special_tokens=False))
            if plen < 1 or plen >= inp["input_ids"].shape[1]:
                continue
            with torch.no_grad():
                if coef != 0.0:
                    with ActivationSteerer(model, steer_vec, coeff=coef,
                                           layer_idx=args.layer - 1, positions="all"):
                        o = model(**inp, output_hidden_states=True)
                else:
                    o = model(**inp, output_hidden_states=True)
            for L in range(nlayers):
                h = o.hidden_states[L][:, plen:, :].mean(1).float().squeeze().cpu()
                hn = h.norm() + 1e-8
                sum_alt[L] += torch.dot(h, uhat_alt[L]) / hn   # cosine (direction only)
                sum_saf[L] += torch.dot(h, uhat_saf[L]) / hn
            nused += 1
            del o
        report["per_coef"][str(coef)] = {
            "n": nused,
            "proj_altruism": (sum_alt / max(nused, 1)).tolist(),
            "proj_safety": (sum_saf / max(nused, 1)).tolist(),
        }
        print(f"coef={coef}: n={nused}  proj_alt@L{args.layer}={sum_alt[args.layer]/max(nused,1):.3f}  "
              f"proj_alt@L{nlayers-1}={sum_alt[nlayers-1]/max(nused,1):.3f}")
    json.dump(report, open(out / "trace_report.json", "w"), indent=2)
    print(f"saved {out/'trace_report.json'}")


if __name__ == "__main__":
    main()
