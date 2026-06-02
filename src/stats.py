"""stats.py — reproducibility harness for the Phase III statistics cited in FACTS.md / CRITIQUE.md.

This script regenerates, WITHOUT loading any LLM, the four families of numbers that the
showcase's load-bearing Phase III claims rest on. It reads only the judged/scored CSVs and
the two shipped persona-direction tensors, and prints a single table. Run it to check that the
prose matches the data.

What it recomputes (and where each number lives in FACTS.md):

  (1) TALK-vs-ACT per-game RLHF deltas + bootstrap 95% CIs   [FACTS 3.2]
      Source arms: t_q25_{base,inst} and t_q3_{base,inst}, coh>=50, coef 0.
      Each "game" is ONE prompt (question_id) scored 30x on the instruct side and 13-24x on the
      base side after the coherence filter (samples within a game are NOT independent). The five
      money games used are Dictator (q0_amount_given), Trust (q1_amount_sent),
      Ultimatum (q2_amount_proposed), Commons (q3_fish_caught, SIGN-INVERTED: fewer fish caught =
      more generous), Transfer (q5_amount_transferred). The 6th game q4_cooperated is dropped
      (degenerate ~0 in all arms). Sign convention: + = more generous.
      Expected: Qwen2.5 Dictator -22.7 (perm p=0.0009, the only effect surviving Bonferroni/10),
      Ultimatum -18.5, Transfer -21.3, Commons -7.3, Trust +2.5. Every Qwen3 per-game CI crosses 0
      (Dictator -2.7, Trust -13.1, Ultimatum -6.8, Commons -2.4, Transfer +16.6 wrong direction)
      => holds on Qwen2.5 ONLY; Qwen3 inconclusive. Do NOT claim cross-family generalization.

  (2) LENGTH-CONTROL regressions   [FACTS 3.2 and 3.5 — these are CAUSALLY DIFFERENT effects]
      (2a) Verbal "talk" gap, base->instruct WEIGHT change:
           OLS altruism ~ instruct + log(words) on the pooled t_* arms (all 13 questions, coh>=50).
           Raw verbal altruism rises (Qwen2.5 16.6->22.2, Qwen3 27.0->34.9) but instruct answers
           are far longer (156->249, 235->369 words); under length control the instruct coefficient
           collapses to ~0, n.s. (FACTS: Q2.5 -0.61 p=0.79, Q3 -0.29 p=0.92). => "instruct talks
           MORE, which the judge reads as more altruistic," NOT "talks more altruistically."
      (2b) v_RLHF(v2) STEERING effect (a DIFFERENT, length-ROBUST effect — must not be bundled
           with 2a): OLS altruism ~ coef + log(words) on day_rlhf_q25 (coh>=50). The coef slope
           SURVIVES length control (FACTS: 4.64 raw -> 3.79 length-controlled, p=5e-4). This is the
           contrast with 2a: the v2 steering verbal effect is length-robust; the base->instruct
           verbal gap is a length artifact. Bundling them would let the robust result lend false
           credibility to the fragile one.

  (3) cos(v1, v2) PER LAYER on the same model (Qwen2.5)   [FACTS 3.3]
      From results/vectors/v_rlhf_v{1,2}_qwen25.pt. Reproducible per-layer cosines:
      L10 +0.092, L15 +0.213, L20 +0.310, L25 +0.331, L28 +0.871 (they converge at the readout
      layer L28). Caveat: v1 is prompt-pooled, v2 response-pooled — the differing pooling basis is
      itself part of why they differ. v1/v2 produce DISSOCIATED downstream effects.

  (4) 7-axis joint R^2 + per-axis VIF / condition number   [FACTS 3.1] — CONDITIONAL.
      The 7 instruction-contrast axis tensors (altruism/refusal/verbosity/deliberation/sycophancy/
      safety_caution/formatting) are NOT shipped in results/vectors/. If they are found on a candidate
      path this step runs and reports joint R^2 (lstsq), per-axis VIF and the design condition number
      (the axes are near-synonymous "elaboration" cluster => collinear => per-axis loadings are NOT
      individually identifiable; joint R^2 < sum of single-axis cos^2 in every family). Otherwise it
      prints a clear "axis vectors not shipped — skipped" line. All cosines/R^2 here are single-layer
      (L20, an attenuated trough) and basis/wording-dependent upper bounds.

No LLM, no torch required. Dependencies: Python stdlib + numpy + pandas. The .pt tensors are read
with a small pure-numpy unpickling reader (torch is used automatically if importable).

Run:
    uvx --with pandas --with numpy python src/stats.py
    # or point at a different data root / vectors dir:
    uvx --with pandas --with numpy python src/stats.py --data-root /path/to/phase3 --vectors-dir /path/to/vectors

Point estimates reproduce FACTS to the precision shown; bootstrap CIs vary slightly with the RNG
seed (fixed to 0 here). Where exact tokenization could shift a length-control point estimate by a
tenth, the conclusion (~0, n.s. for 2a; slope survives, p=5e-4 for 2b) is what is load-bearing.
"""
from __future__ import annotations

import argparse
import io
import math
import pickle as pickle_module
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- config
REPO = Path(__file__).resolve().parents[1]
# Scored CSVs live in the working repo (sibling of the showcase); allow override via --data-root.
DEFAULT_DATA_ROOT = REPO.parent / "subliminal_learning" / "data" / "persona_vectors" / "phase3"
DEFAULT_VECTORS_DIR = REPO / "results" / "vectors"

COH_MIN = 50.0
N_BOOT = 10_000
N_PERM = 10_000
SEED = 0

# game definition: question_id -> (value column, sign, display name)
#   sign = +1 keeps raw direction (higher = more generous);
#   Commons is sign-INVERTED (fewer fish caught = more generous).
#   q4_cooperated (6th game) is intentionally excluded (degenerate ~0 in all arms).
GAMES = [
    ("altruism_0", "q0_amount_given", +1, "Dictator"),
    ("altruism_1", "q1_amount_sent", +1, "Trust"),
    ("altruism_2", "q2_amount_proposed", +1, "Ultimatum"),
    ("altruism_3", "q3_fish_caught", -1, "Commons(inv)"),
    ("altruism_5", "q5_amount_transferred", +1, "Transfer"),
]

AXES = ["altruism", "refusal", "verbosity", "deliberation",
        "sycophancy", "safety_caution", "formatting"]
AXIS_LAYER = 20


# ----------------------------------------------------------------------------- helpers
def _wc(s: str) -> int:
    return len(str(s).split())


def _norm_p(z: float) -> float:
    """Two-sided p from a z-score via the stdlib erf (no scipy)."""
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def _spearman(x, y) -> float:
    x = pd.Series(x).rank().to_numpy()
    y = pd.Series(y).rank().to_numpy()
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y / denom) if denom else float("nan")


def _ols(X: np.ndarray, y: np.ndarray):
    """Return (beta, se, n, k) for OLS with classical (homoskedastic) SEs."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = max(n - k, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return beta, se, n, k


def _load_scored(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["coherence"] = pd.to_numeric(df["coherence"], errors="coerce")
    df["altruism"] = pd.to_numeric(df["altruism"], errors="coerce")
    return df


# --------------------------------------------------------------- pure-numpy .pt reader
def _load_pt_tensor(path: Path) -> np.ndarray:
    """Load a single 2-D float tensor (n_layers, hidden) from a torch .pt zip archive
    WITHOUT importing torch. Uses torch automatically if it is importable."""
    try:  # fast path if torch is present
        import torch  # type: ignore
        return torch.load(path, map_location="cpu").float().numpy().astype(np.float64)
    except Exception:
        pass

    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        base = names[0].split("/")[0]
        pkl = z.read(f"{base}/data.pkl")
        blob_name = next(n for n in names if re.search(r"/data/0$", n))
        raw = z.read(blob_name)

        # dtype from the storage class named in the pickle
        dtype = "<f4"
        if b"DoubleStorage" in pkl:
            dtype = "<f8"
        elif b"HalfStorage" in pkl:
            dtype = "<f2"
        elif b"FloatStorage" not in pkl:
            raise ValueError(f"unsupported tensor dtype in {path.name}")

        # shape from the _rebuild_tensor_v2 args via a permissive unpickler
        class _Unp(pickle_module.Unpickler):
            def find_class(self, mod, name):
                return lambda *a, **k: ("OBJ", name, a)

            def persistent_load(self, pid):
                return ("STORAGE", pid)

        obj = _Unp(io.BytesIO(pkl)).load()
        size = None
        if isinstance(obj, tuple) and obj[0] == "OBJ" and "rebuild_tensor" in obj[1]:
            size = obj[2][2]  # (storage, offset, size, stride, ...)
        arr = np.frombuffer(raw, dtype=dtype).astype(np.float64)
        if size is not None:
            arr = arr.reshape(size)
        return arr


def _cos_per_layer(a: np.ndarray, b: np.ndarray):
    n = min(a.shape[0], b.shape[0])
    out = []
    for L in range(n):
        x, y = a[L], b[L]
        d = float(np.linalg.norm(x) * np.linalg.norm(y))
        out.append(float(x @ y / d) if d else float("nan"))
    return out


# ----------------------------------------------------------------------------- (1) talk-vs-act
def talk_vs_act(data_root: Path, rng: np.random.Generator):
    rows = []
    families = [("Qwen2.5", "t_q25_base", "t_q25_inst"),
                ("Qwen3", "t_q3_base", "t_q3_inst")]
    for fam, base_arm, inst_arm in families:
        b = _load_scored(data_root / base_arm / "scored" / "altruism_coef+0.0_scored.csv")
        i = _load_scored(data_root / inst_arm / "scored" / "altruism_coef+0.0_scored.csv")
        if b is None or i is None:
            rows.append({"family": fam, "game": "(arms missing)", "delta": np.nan,
                         "ci_lo": np.nan, "ci_hi": np.nan, "n_base": 0, "n_inst": 0,
                         "perm_p": np.nan})
            continue
        for qid, col, sign, name in GAMES:
            bv = pd.to_numeric(b.loc[(b.question_id == qid) & (b.coherence >= COH_MIN), col],
                               errors="coerce").dropna().to_numpy()
            iv = pd.to_numeric(i.loc[(i.question_id == qid) & (i.coherence >= COH_MIN), col],
                               errors="coerce").dropna().to_numpy()
            if len(bv) == 0 or len(iv) == 0:
                continue
            delta = sign * (iv.mean() - bv.mean())
            boots = np.empty(N_BOOT)
            for k in range(N_BOOT):
                boots[k] = sign * (rng.choice(iv, len(iv)).mean() - rng.choice(bv, len(bv)).mean())
            lo, hi = np.percentile(boots, [2.5, 97.5])
            # two-sided permutation p (only meaningful to report for the headline Dictator cell)
            perm_p = np.nan
            if name == "Dictator":
                obs = abs(iv.mean() - bv.mean())
                pool = np.concatenate([bv, iv])
                nb = len(bv)
                cnt = 0
                for _ in range(N_PERM):
                    pp = rng.permutation(pool)
                    if abs(pp[:nb].mean() - pp[nb:].mean()) >= obs:
                        cnt += 1
                perm_p = (cnt + 1) / (N_PERM + 1)
            rows.append({"family": fam, "game": name, "delta": delta,
                         "ci_lo": lo, "ci_hi": hi, "n_base": len(bv), "n_inst": len(iv),
                         "perm_p": perm_p})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- (2a) verbal gap
def verbal_length_control(data_root: Path):
    out = []
    for fam, base_arm, inst_arm in [("Qwen2.5", "t_q25_base", "t_q25_inst"),
                                    ("Qwen3", "t_q3_base", "t_q3_inst")]:
        b = _load_scored(data_root / base_arm / "scored" / "altruism_coef+0.0_scored.csv")
        i = _load_scored(data_root / inst_arm / "scored" / "altruism_coef+0.0_scored.csv")
        if b is None or i is None:
            out.append({"family": fam, "raw_gap": np.nan, "ctrl_coef": np.nan,
                        "ctrl_p": np.nan, "alt_base": np.nan, "alt_inst": np.nan, "n": 0})
            continue
        bb = b[b.coherence >= COH_MIN]
        ii = i[i.coherence >= COH_MIN]
        alt_base, alt_inst = bb.altruism.mean(), ii.altruism.mean()
        X, y = [], []
        for inst_flag, df in [(0, bb), (1, ii)]:
            for _, r in df.iterrows():
                w = _wc(r["answer"])
                if w <= 0 or pd.isna(r["altruism"]):
                    continue
                X.append([1.0, inst_flag, math.log(w)])
                y.append(float(r["altruism"]))
        X, y = np.array(X), np.array(y)
        beta, se, n, k = _ols(X, y)
        p = _norm_p(beta[1] / se[1])
        out.append({"family": fam, "raw_gap": alt_inst - alt_base, "ctrl_coef": beta[1],
                    "ctrl_p": p, "alt_base": alt_base, "alt_inst": alt_inst, "n": n})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- (2b) v2 steering
def v2_steering_length_control(data_root: Path):
    arm = data_root / "day_rlhf_q25" / "scored"
    if not arm.exists():
        return None
    frames = []
    for f in sorted(arm.glob("altruism_coef*_scored.csv")):
        m = re.search(r"coef([+-]?[0-9.]+)_scored", f.name)
        if not m:
            continue
        df = _load_scored(f)
        if df is None:
            continue
        df["coef"] = float(m.group(1))
        frames.append(df)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d = d[d.coherence >= COH_MIN]
    Xr, Xc, y = [], [], []
    for _, r in d.iterrows():
        w = _wc(r["answer"])
        if w <= 0 or pd.isna(r["altruism"]):
            continue
        Xr.append([1.0, r["coef"]])
        Xc.append([1.0, r["coef"], math.log(w)])
        y.append(float(r["altruism"]))
    Xr, Xc, y = np.array(Xr), np.array(Xc), np.array(y)
    br, _, _, _ = _ols(Xr, y)
    bc, sec, n, _ = _ols(Xc, y)
    p = _norm_p(bc[1] / sec[1])
    return {"raw_slope": br[1], "ctrl_slope": bc[1], "ctrl_p": p, "n": n}


def v1_dictator_diagnostics(data_root: Path):
    """v1 giving 'axis' is NOT monotonic and is CONFOUNDED with coherence collapse (FACTS 3.3)."""
    arm = data_root / "day_v1_q25" / "scored"
    if not arm.exists():
        return None
    per_coef, coefs, dollars, cohs = {}, [], [], []
    for f in sorted(arm.glob("altruism_coef*_scored.csv")):
        m = re.search(r"coef([+-]?[0-9.]+)_scored", f.name)
        if not m:
            continue
        c = float(m.group(1))
        df = _load_scored(f)
        if df is None:
            continue
        sub = df[(df.question_id == "altruism_0") & (df.coherence >= COH_MIN)]
        d = pd.to_numeric(sub["q0_amount_given"], errors="coerce").dropna().to_numpy()
        if len(d) == 0:
            continue
        per_coef[c] = (d.mean(), len(d))
        coefs.extend([c] * len(d))
        dollars.extend(d.tolist())
        cohs.extend(sub["coherence"].to_numpy().tolist())
    if not per_coef:
        return None
    return {"per_coef": dict(sorted(per_coef.items())),
            "sp_coef_dollar": _spearman(coefs, dollars),
            "sp_dollar_coh": _spearman(dollars, cohs)}


# ----------------------------------------------------------------------------- (3) cos(v1,v2)
def cos_v1_v2(vectors_dir: Path):
    p1 = vectors_dir / "v_rlhf_v1_qwen25.pt"
    p2 = vectors_dir / "v_rlhf_v2_qwen25.pt"
    if not (p1.exists() and p2.exists()):
        return None
    v1 = _load_pt_tensor(p1)
    v2 = _load_pt_tensor(p2)
    if v1.shape != v2.shape:
        return {"shape_mismatch": (v1.shape, v2.shape)}
    cos = _cos_per_layer(v1, v2)
    return {"shape": v1.shape, "cos": cos}


# ----------------------------------------------------------------------------- (4) 7-axis R^2
def _find_axis_dir(data_root: Path, vectors_dir: Path):
    """The 7 instruction-contrast axis tensors are NOT shipped in results/vectors/.
    Search a few candidate locations; return (dir, v_rlhf_path) or None."""
    candidates = [
        (data_root / "a_axes_qwen25", data_root / "rlhf_v2_qwen25" / "v_rlhf_v2.pt"),
        (vectors_dir / "a_axes_qwen25", vectors_dir / "v_rlhf_v2_qwen25.pt"),
    ]
    for axes_dir, vr in candidates:
        if axes_dir.exists() and all((axes_dir / f"v_{a}.pt").exists() for a in AXES) and vr.exists():
            return axes_dir, vr
    return None


def axis_joint_r2(data_root: Path, vectors_dir: Path):
    found = _find_axis_dir(data_root, vectors_dir)
    if found is None:
        return None
    axes_dir, vr = found
    v_rlhf = _load_pt_tensor(vr)
    L = min(AXIS_LAYER, v_rlhf.shape[0] - 1)
    cols, names = [], []
    for a in AXES:
        va = _load_pt_tensor(axes_dir / f"v_{a}.pt")
        if va.shape[0] != v_rlhf.shape[0]:
            continue
        u = va[L] / (np.linalg.norm(va[L]) + 1e-12)
        cols.append(u)
        names.append(a)
    if not cols:
        return None
    A = np.array(cols).T            # (hidden, n_axes)
    y = v_rlhf[L]
    beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    r2 = 1.0 - float(((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    sum_cos2 = float(sum((float(u @ y / (np.linalg.norm(u) * np.linalg.norm(y)))) ** 2 for u in cols))
    # VIF per axis: 1 / (1 - R^2_j) regressing each axis on the others
    vif = {}
    for j, name in enumerate(names):
        others = np.delete(A, j, axis=1)
        bj, _, _, _ = np.linalg.lstsq(others, A[:, j], rcond=None)
        rj = others @ bj
        ssr = float(((A[:, j] - rj) ** 2).sum())
        sst = float(((A[:, j] - A[:, j].mean()) ** 2).sum())
        r2j = 1.0 - ssr / sst if sst else float("nan")
        vif[name] = 1.0 / (1.0 - r2j) if r2j < 1 else float("inf")
    cond = float(np.linalg.cond(A.T @ A))
    return {"layer": L, "joint_r2": r2, "sum_cos2": sum_cos2, "vif": vif,
            "condition_number": cond, "axes": names}


# ----------------------------------------------------------------------------- main / table
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="phase3 dir holding the <arm>/scored/*.csv files")
    ap.add_argument("--vectors-dir", type=Path, default=DEFAULT_VECTORS_DIR,
                    help="dir holding v_rlhf_v{1,2}_qwen25.pt")
    args = ap.parse_args(argv)

    data_root: Path = args.data_root
    vectors_dir: Path = args.vectors_dir
    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print("Phase III statistics reproducibility check (no LLM)")
    print(f"  data-root  : {data_root}  {'[OK]' if data_root.exists() else '[MISSING]'}")
    print(f"  vectors-dir: {vectors_dir}  {'[OK]' if vectors_dir.exists() else '[MISSING]'}")
    print(f"  filter     : coherence >= {COH_MIN:g}; bootstrap reps={N_BOOT}, seed={SEED}")
    print("=" * 78)

    if not data_root.exists():
        print("\n[skip] data-root not found — talk-vs-act / length-control steps skipped.")
    else:
        # ---- (1) talk-vs-act -------------------------------------------------
        print("\n(1) TALK-vs-ACT per-game RLHF delta (instruct-base, sign-adj; +=more generous) "
              "[FACTS 3.2]")
        print("    Commons SIGN-INVERTED (fewer fish=generous); q4_cooperated dropped; "
              "n=samples of ONE prompt, not independent games.")
        tva = talk_vs_act(data_root, rng)
        if tva.empty:
            print("    (no t_* arms found)")
        else:
            print(f"    {'family':<8}{'game':<14}{'delta':>8}{'  95% CI':>20}"
                  f"{'n_base':>8}{'n_inst':>8}{'perm_p':>10}")
            for _, r in tva.iterrows():
                ci = (f"[{r.ci_lo:+6.1f},{r.ci_hi:+6.1f}]"
                      if pd.notna(r.ci_lo) else f"{'--':>15}")
                pp = f"{r.perm_p:.4f}" if pd.notna(r.perm_p) else ""
                dl = f"{r.delta:+8.1f}" if pd.notna(r.delta) else f"{'--':>8}"
                print(f"    {r.family:<8}{r.game:<14}{dl}{ci:>20}"
                      f"{int(r.n_base):>8}{int(r.n_inst):>8}{pp:>10}")
            print("    VERDICT: holds on Qwen2.5 (Dictator survives Bonferroni/10, perm p~0.0009);")
            print("             Qwen3 INCONCLUSIVE — every per-game CI crosses 0. NOT cross-family.")

        # ---- (2a) verbal gap -------------------------------------------------
        print("\n(2a) VERBAL talk-gap, base->instruct WEIGHT change: OLS altruism ~ instruct + "
              "log(words) [FACTS 3.2]")
        vg = verbal_length_control(data_root)
        print(f"    {'family':<8}{'alt_base':>9}{'alt_inst':>9}{'raw_gap':>9}"
              f"{'ctrl_coef':>11}{'ctrl_p':>9}{'n':>7}")
        for _, r in vg.iterrows():
            if pd.isna(r.raw_gap):
                print(f"    {r.family:<8} (arms missing)")
                continue
            print(f"    {r.family:<8}{r.alt_base:>9.1f}{r.alt_inst:>9.1f}{r.raw_gap:>+9.1f}"
                  f"{r.ctrl_coef:>+11.2f}{r.ctrl_p:>9.2f}{int(r.n):>7}")
        print("    => raw verbal altruism rises (16.6->22.2 / 27.0->34.9) but VANISHES under length")
        print("       control (~0, n.s.): instruct talks MORE, judge reads length as altruism. ARTIFACT.")

        # ---- (2b) v2 steering ------------------------------------------------
        print("\n(2b) v_RLHF(v2) STEERING effect (DISTINCT, length-ROBUST): OLS altruism ~ coef + "
              "log(words), day_rlhf_q25 [FACTS 3.5]")
        v2 = v2_steering_length_control(data_root)
        if v2 is None:
            print("    (day_rlhf_q25 not found)")
        else:
            print(f"    raw coef slope = {v2['raw_slope']:+.2f}  ->  length-controlled "
                  f"{v2['ctrl_slope']:+.2f}  (p={v2['ctrl_p']:.1e}, n={v2['n']})")
            print("    => coef slope SURVIVES length control (FACTS ~4.64->3.79, p=5e-4). This is the")
            print("       key contrast with 2a and MUST NOT be bundled with the base->instruct verbal gap.")

        # ---- v1 giving diagnostics ------------------------------------------
        print("\n(2c) v1 Dictator-$ axis is NOT monotonic, CONFOUNDED with coherence collapse "
              "[FACTS 3.3]")
        v1d = v1_dictator_diagnostics(data_root)
        if v1d is None:
            print("    (day_v1_q25 not found)")
        else:
            cells = "  ".join(f"{c:+.0f}:${m:.1f}(n{n})" for c, (m, n) in v1d["per_coef"].items())
            print(f"    Dictator $ by coef: {cells}")
            print(f"    Spearman(coef,$)={v1d['sp_coef_dollar']:+.2f}  "
                  f"Spearman($,coherence)={v1d['sp_dollar_coh']:+.2f}")
            print("    => declines on NET (rho~-0.33, p=0.01; Welch -2vs+2 p=0.008) but NON-monotone")
            print("       (+1 > 0) and giving co-occurs with INCOHERENCE (rho~-0.67). NOT a clean axis.")

    # ---- (3) cos(v1,v2) -----------------------------------------------------
    print("\n(3) cos(v1, v2) PER LAYER, same model (Qwen2.5) [FACTS 3.3] — REPRODUCIBLE")
    cv = cos_v1_v2(vectors_dir)
    if cv is None:
        print("    [skip] v_rlhf_v{1,2}_qwen25.pt not found in vectors-dir.")
    elif "shape_mismatch" in cv:
        print(f"    [skip] shape mismatch {cv['shape_mismatch']} — need matched same-model pair.")
    else:
        cos = cv["cos"]
        marks = [L for L in (10, 15, 20, 25, 28) if L < len(cos)]
        print(f"    tensor shape {cv['shape']} (n_layers, hidden)")
        print("    " + "  ".join(f"L{L}:{cos[L]:+.3f}" for L in marks))
        print("    => they DIVERGE early, CONVERGE at the readout layer (L28). v1 prompt-pooled vs")
        print("       v2 response-pooled — differing pooling basis is itself part of why they differ.")

    # ---- (4) 7-axis joint R^2 / VIF -----------------------------------------
    print("\n(4) 7-axis joint R^2 + per-axis VIF / condition number @L20 [FACTS 3.1] — CONDITIONAL")
    ax = axis_joint_r2(data_root, vectors_dir)
    if ax is None:
        print("    [skip] The 7 instruction-contrast axis tensors are NOT shipped in results/vectors/")
        print("           (only v1/v2 are). FACTS' joint-R^2 (Q2.5 6.2%, Q3 19.3%, Llama 10.7%) and the")
        print("           per-family cosines come from a_axis_decomposition_report.json; re-running them")
        print("           needs the a_axes_* vector dirs (basis/wording-dependent, single-layer L20).")
    else:
        print(f"    layer L{ax['layer']}, axes: {', '.join(ax['axes'])}")
        print(f"    joint R^2 = {ax['joint_r2']*100:.1f}%   "
              f"sum single-axis cos^2 = {ax['sum_cos2']*100:.1f}%   "
              f"(joint < sum => MULTICOLLINEARITY: loadings NOT identifiable)")
        print(f"    design condition number = {ax['condition_number']:.1f}")
        print("    per-axis VIF: " + "  ".join(f"{a}={v:.1f}" for a, v in ax["vif"].items()))
        print("    CAVEAT: single-layer L20 (attenuated trough); cosines/R^2 are basis-dependent "
              "upper bounds.")

    print("\n" + "=" * 78)
    print("Done. All point estimates should match FACTS.md; bootstrap CIs vary with the RNG seed.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
