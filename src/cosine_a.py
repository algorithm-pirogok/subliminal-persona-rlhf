"""A analysis + generality: how much of v_RLHF lies along each behavioral axis,
across model families. All axes (incl. altruism) extracted by the SAME
instruction-contrast method, so cos(v_RLHF, axis) is comparable across models.

Headline generality test: is cos(v_RLHF, altruism) near-zero on EVERY family?
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
P = REPO / "data/persona_vectors/phase3"
LAYER = 20

MODELS = {
    "Qwen2.5-7B": {"v_rlhf": P / "rlhf_v2_qwen25/v_rlhf_v2.pt",      "axes_dir": P / "a_axes_qwen25"},
    "Qwen3-8B":   {"v_rlhf": P / "rlhf_v2_qwen3_fixed/v_rlhf_v2.pt", "axes_dir": P / "a_axes_qwen3"},
    "Llama-3.1-8B": {"v_rlhf": P / "rlhf_v2_llama/v_rlhf_v2.pt",     "axes_dir": P / "a_axes_llama"},
}
AXES = ["altruism", "refusal", "verbosity", "deliberation",
        "sycophancy", "safety_caution", "formatting"]


def cos_per_layer(a, b):
    n = min(a.shape[0], b.shape[0])
    return [F.cosine_similarity(a[L][None].float(), b[L][None].float()).item() for L in range(n)]


def main():
    report = {}
    print(f"{'model':<14} " + " ".join(f"{ax:>13}" for ax in AXES))
    for model, cfg in MODELS.items():
        if not cfg["v_rlhf"].exists():
            print(f"{model:<14} [wait] no v_RLHF ({cfg['v_rlhf']})")
            continue
        v_rlhf = torch.load(cfg["v_rlhf"]).float()
        row = {}
        cells = []
        for axis in AXES:
            ap = cfg["axes_dir"] / f"v_{axis}.pt"
            if not ap.exists():
                cells.append(f"{'--':>13}")
                continue
            v_axis = torch.load(ap).float()
            if v_axis.shape[0] != v_rlhf.shape[0]:
                cells.append(f"{'shape!':>13}")
                continue
            c = cos_per_layer(v_rlhf, v_axis)
            row[axis] = c
            L = min(LAYER, len(c) - 1)
            cells.append(f"{c[L]:>+13.3f}")
        report[model] = row
        print(f"{model:<14} " + " ".join(cells))

    print(f"\n(values = cos(v_RLHF, axis) @ layer {LAYER}; altruism near-0 across families => orthogonality is general)")
    out = P / "a_axis_decomposition_report.json"
    json.dump({"layer": LAYER, "axes": AXES, "models": report}, open(out, "w"), indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
