"""E analysis: does subliminal SFT move the model along -v_RLHF (de-alignment)?

    v_RLHF      = mean_h(instruct) - mean_h(base)        [alignment axis]
    Δv_student  = mean_h(student)  - mean_h(instruct)    [subliminal SFT shift]
    Δv_owl      = mean_h(owl)      - mean_h(instruct)    [generic numeric-SFT control]

Predict cos(Δv_student, v_RLHF) < 0 (student drifted back toward base).
Specificity: |cos| for student should exceed owl-control if altruism-specific.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
P = REPO / "data/persona_vectors/phase3"
BASE = P / "rlhf_v2_qwen25/base_response_means.pt"
INST = P / "rlhf_v2_qwen25/instruct_response_means.pt"
STUD = P / "e_student_means/response_means.pt"
OWL = P / "e_owl_means/response_means.pt"
LAYER = 20
OUT = P / "e_dealignment_report.json"


def main():
    for name, p in [("base", BASE), ("instruct", INST), ("student", STUD)]:
        if not p.exists():
            print(f"[wait] missing {name} means: {p}")
            return
    base = torch.load(BASE).float()
    inst = torch.load(INST).float()
    stud = torch.load(STUD).float()
    owl = torch.load(OWL).float() if OWL.exists() else None

    v_rlhf = inst - base
    dv_s = stud - inst

    rows = []
    for L in range(v_rlhf.shape[0]):
        nrl = float(v_rlhf[L].norm())
        nds = float(dv_s[L].norm())
        proj = float((dv_s[L] @ v_rlhf[L]) / nrl) if nrl > 0 else float("nan")
        r = {
            "layer": L,
            "cos_dvStudent_vRLHF": F.cosine_similarity(dv_s[L][None], v_rlhf[L][None]).item(),
            "proj_dvStudent_on_vRLHF": proj,
            "frac_dvStudent_along_vRLHF": (proj / nds) if nds > 0 else float("nan"),
            "norm_vRLHF": nrl,
            "norm_dvStudent": nds,
        }
        if owl is not None:
            dv_o = owl[L] - inst[L]
            r["cos_dvOwl_vRLHF"] = F.cosine_similarity(dv_o[None], v_rlhf[L][None]).item()
            r["norm_dvOwl"] = float(dv_o.norm())
        rows.append(r)

    m = rows[LAYER]
    print(f"\n=== E: de-alignment along v_RLHF (layer {LAYER}) ===")
    print(f"cos(Δv_student, v_RLHF) = {m['cos_dvStudent_vRLHF']:+.4f}   "
          "(NEGATIVE => student moved toward base / de-aligned)")
    print(f"  fraction of |Δv_student| along v_RLHF = {m['frac_dvStudent_along_vRLHF']:+.3f}")
    if owl is not None:
        print(f"cos(Δv_owl,     v_RLHF) = {m['cos_dvOwl_vRLHF']:+.4f}   (control: owl numeric-SFT)")
        print(f"  specificity: |student|={abs(m['cos_dvStudent_vRLHF']):.3f} vs |owl|={abs(m['cos_dvOwl_vRLHF']):.3f}")
    print(f"  |v_RLHF|={m['norm_vRLHF']:.2f}  |Δv_student|={m['norm_dvStudent']:.2f}")
    print("\nBehavioral anchor: student altruism 15.8 between base 10.1 and instruct 25.7")

    json.dump({"layer": LAYER, "owl_control": owl is not None, "per_layer": rows},
              open(OUT, "w"), indent=2)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
