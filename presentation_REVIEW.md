# Presentation Review — `presentation.pdf`

**Reviewer pass:** rendering check + quantitative spot-check against `FACTS.md` / `CONCLUSION.md` /
`docs/03_rlhf_axis.md` / `docs/limitations.md`, followed by fixes and a clean two-pass recompile.

**Final state:** 27 slides (title + 26 content), compiles clean (`pdflatex` x2, exit 0), no missing-file
boxes, no visible overflow.

---

## 1. Rendering check (read pages 5–9, 12–17, 19–27 of the PDF)

- **Figures render correctly** (no missing-file boxes): `fig4_effect_comparison`, `fig6_hidden_states`
  (p8), `altruism_sweep_4cond` (p11), `v_rlhf_vs_v_alt` (p12, both panels). All referenced figures exist
  in `figures/`.
- **Tables/math readable** throughout (Phase I 4-condition table, orthogonality cosine table, 7-axis R²
  table, talk-vs-act CI table, v1/v2 dissociation table).
- **Overflow:** two minor `Overfull \vbox` warnings found initially (line 169 = 2.6pt; line 333 = 5.2pt,
  footer touching the collinearity paragraph on the 7-axis slide). **Both fixed** (see §3). After fix only
  a sub-pixel 0.43pt residual remains at line 169 — not visible.
- Only benign font warnings (bold-italic sans-serif auto-substituted) — standard Beamer, no visual impact.

## 2. Quantitative spot-check vs `FACTS.md` (all verified MATCHING)

1. **Subliminal:** owl 12.4%→56.7% (+44.3pp); 89.5% linear separability (±0.5, 5-fold), z=−330, centroid
   cosine 0.971. ✓
2. **Orthogonality cosines @ L20** (full 7×3 table): altruism −0.028/+0.134/+0.106, refusal
   +0.043/−0.318/−0.179, etc. — exact match. ✓
3. **7-axis joint R²** 6.2/19.3/10.7%, Σcos² 14.3/28.5/19.4 — match; flagged PROVISIONAL (red), only Σcos²
   reproducible, 7×7 Gram needed. ✓
4. **Talk side:** Qwen2.5 +5.5 (16.6→22.2, p=0.007), Qwen3 +7.9 (27.0→34.9, p=0.002); length-control
   −0.61 (p=0.79) / −0.29 (p=0.92); words 156→249, 235→369; judge~length r≈+0.2. ✓
5. **Act side per-game CIs:** Dictator −22.7 (perm p=0.0009), Qwen3 Dictator −2.7 [−16.2,+9.7], Transfer
   +16.6 (wrong dir.), full table — match. ✓
6. **v1/v2:** cos(v1,v2)@L20=0.31; per-layer .092/.213/.310/.331/.871; v2 Qwen3 ρ=+0.009 (p=0.95); v1
   ρ($,coh)=−0.66; OLS slope 4.64→3.79 (p=5e-4). ✓
7. **Retractions:** truncation (86.5%, 14.0→36.0, $0.1→$17.5, Δ −8.2→+13.8); 0% refusals; owl −0.35 vs
   student −0.26. ✓

## 3. Killed / overclaimed statements — NONE reintroduced

Checked explicitly for the red-team-killed phrasings; all are absent or correctly negated:
- No "both families"/cross-family behavioral generalization — deck says "Qwen2.5 only," "Qwen3
  inconclusive," "do NOT claim cross-family or population generalization." ✓
- No "monotonic" applied to v1 giving — deck says v1 "declines on net, NOT monotone." (The only
  "monotonic" usages are the Phase II altruism **steering** sweep, which FACTS explicitly labels
  monotonic, and v2's verbal rise — both legitimate.) ✓
- No "Bonferroni-surviving" claim — deck explicitly says "Never 'Bonferroni-surviving'" / category error. ✓
- No positive cross-family orthogonality claim — altruism framed as "smallest **only** on Qwen2.5";
  per-family reads preserved. ✓
- No "RLHF reduces altruism"/inversion as a live claim — present only as the **retracted** truncation
  artifact. ✓
- Length-null correctly hedged "**largely**, never purely"; filter-contingent caveat present. ✓

## 4. Fixes applied (to `presentation.tex`)

- Line ~163: `\vspace{0.4em}` → `\vspace{0.2em}` on the Phase I hidden-state frame (clears 2.6pt overflow).
- Line ~327/333: `\vspace{0.3em}`→`0.2em` and `\footnotesize`→`\scriptsize` on the 7-axis collinearity
  footer (clears 5.2pt overflow; footer no longer collides with body text).

Recompiled twice with the same `TinyTeX` PATH; exit 0, 27 pages.

## 5. Notes (out of scope, not presentation defects)

- `README.md` and `FACTS.md` manifest both call this a "17-slide deck"; the actual deck is **27 slides**.
  The slide footers correctly read "/27". The stale "17-slide" string is in the README/manifest, not in
  `presentation.tex`, so it was left untouched — flag for the README owner.
- Reproducibility claims on the limitations slide (`src/stats.py`, `results/vectors/v_rlhf_v{1,2}_qwen25.pt`)
  were verified to exist on disk — accurate.

**Verdict:** presentation is faithful to `FACTS.md` (hedging matched exactly), renders cleanly, no
overclaims. Final: 27 pages, compiles without error.
