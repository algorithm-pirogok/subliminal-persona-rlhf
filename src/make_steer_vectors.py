"""Phase 1: build steering-condition vectors for the mediation/sufficiency test.

Given v_RLHF [L,H] and one or more candidate axis vectors, produce per-layer
steering vectors all scaled to ||v_RLHF[L]|| (so the v_RLHF condition == the
known raw effect, and each axis is magnitude-matched for a fair direction test):

  cond_rlhf.pt           = v_RLHF
  cond_<name>.pt         = axis<name>, rescaled to ||v_RLHF|| per layer   (sufficiency arm)
  cond_rlhf_no_<p>.pt    = v_RLHF with axis<p> projected out, rescaled     (optional necessity arm)

Usage:
  --rlhf v_rlhf.pt --axis safety=v_safety.pt --axis deliberation=v_delib.pt [--project_out safety] --out DIR
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import torch
import torch.nn.functional as F


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp"); torch.save(obj, tmp); os.replace(tmp, path)


def scale_to(ref, vec):  # rescale vec to ref's per-layer norm
    out = torch.zeros_like(vec); eps = 1e-8
    for L in range(vec.shape[0]):
        n = vec[L].norm()
        out[L] = vec[L] * (ref[L].norm() / n) if n > eps else ref[L].clone()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rlhf", required=True)
    p.add_argument("--axis", action="append", default=[], help="name=path (repeatable)")
    p.add_argument("--project_out", default=None, help="axis name to also produce v_RLHF minus it")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    vr = torch.load(args.rlhf).float()
    axes = {}
    for spec in args.axis:
        name, path = spec.split("=", 1)
        axes[name] = torch.load(path).float()
        assert axes[name].shape == vr.shape, f"{name} shape {axes[name].shape} != rlhf {vr.shape}"

    atomic_save(vr.clone(), out / "cond_rlhf.pt")
    L = 20
    print(f"saved cond_rlhf.pt  |v_RLHF|@L{L}={vr[L].norm():.2f}")
    for name, va in axes.items():
        atomic_save(scale_to(vr, va), out / f"cond_{name}.pt")
        c = F.cosine_similarity(vr[L][None], va[L][None]).item()
        print(f"saved cond_{name}.pt  cos(RLHF,{name})@L{L}={c:+.3f}")

    if args.project_out:
        va = axes[args.project_out]; resid = torch.zeros_like(vr); eps = 1e-8
        for Lx in range(vr.shape[0]):
            n = va[Lx].norm()
            if n > eps:
                shat = va[Lx] / n
                resid[Lx] = vr[Lx] - (vr[Lx] @ shat) * shat
            else:
                resid[Lx] = vr[Lx].clone()
        atomic_save(scale_to(vr, resid), out / f"cond_rlhf_no_{args.project_out}.pt")
        c = F.cosine_similarity(vr[L][None], resid[L][None]).item()
        print(f"saved cond_rlhf_no_{args.project_out}.pt  cos(RLHF,resid)@L{L}={c:+.3f} (necessity arm; ~1 if axis is a small component)")


if __name__ == "__main__":
    main()
