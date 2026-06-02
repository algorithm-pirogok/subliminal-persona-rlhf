# Limitations & Retractions

This document leads with the claims we **killed ourselves**, then states the global
caveats that scope every result in this work. We treat self-correction as a core part
of the research process: the strongest version of a finding is one that has survived our
own attempts to break it.

---

## A. Claims we retracted ourselves

We list these first, on purpose. Each one is a claim we initially entertained (or in some
cases circulated internally) and then withdrew after a control, a classifier, or a fixed
artifact showed it did not hold. Where applicable we give the before/after numbers.

### 1. Cross-generation RLHF inversion — TRUNCATION ARTIFACT

We believed at one point that **Qwen3 RLHF *reduces* altruism** ("cross-generation RLHF
inversion"). This was a **truncation artifact**: **86.5% of Qwen3-Instruct answers cut off
at `max_tokens=250`**.

| Quantity | Truncated (max_tokens=250) | Fixed (max_tokens=1024) |
|---|---|---|
| Altruism | 14.0 | → 36.0 |
| Dictator $ | $0.1 | → $17.5 |
| RLHF Δ (altruism) | −8.2 | → +13.8 |

There is **no inversion**. **RETRACTED.**

### 2. "Qwen3 = rational refuser ($0 = strategy)" — REJECTED

We considered explaining Qwen3's apparent $0 giving as a deliberate strategic refusal
("rational refuser"). A refusal classifier found **0% explicit refusals** → claim
**rejected**.

### 3. "Subliminal SFT de-aligns *specifically*" — REJECTED (generic)

Subliminal SFT does shift student activations along **−v_RLHF** (de-alignment,
**cos −0.26 @L20**). But an **owl-control student** (generic numeric SFT) de-aligns **MORE**
(**−0.35 @L20**) → the effect is **generic to numeric SFT, not altruism-specific**. The
specificity claim was killed by the owl control. **L20-specific caveat:** the −0.26/−0.35
are read at L20; at the LAST layer L28 both flip **POSITIVE** (student +0.75, owl +0.69, a
boundary artifact). The **robust** claim is the **owl > student ordering** (owl de-aligns
more at every layer), NOT the absolute sign.

### 4. "v_RLHF steers giving DOWN" — v1/v2 CONFLATION (fixed)

The claim that "v_RLHF steers giving down" **conflated two different vectors**: v1 (raw
basis, prompt-pooled) and v2 (matched basis, response-pooled), with **cos(v1, v2) @ L20 =
0.31** (different directions). This cosine is now **reproducible** — both same-model
(Qwen2.5) vectors are shipped at `results/vectors/v_rlhf_v{1,2}_qwen25.pt`; per layer
L10 +0.092 / L15 +0.213 / L20 +0.310 / L25 +0.331 / L28 +0.871 (they converge at the
readout layer). Caveat: v1 is prompt-pooled and v2 response-pooled — the differing pooling
basis is itself part of why they differ. Once disentangled, the two produce **dissociated**
effects:

| Vector | Verbal altruism | Behavioral Dictator $ |
|---|---|---|
| **v2** (matched basis) | rises with +coef (e.g. Qwen3 5→55, monotone) | does NOT track (Qwen3 FLAT, Spearman +0.009, p=0.95) |
| **v1** (raw basis, @1024) | FLAT (~22 across coef) | declines on net but NOT monotone, entangled with coherence collapse |

For v1, Dictator $ at coef −2/−1/0/+1/+2 = $45.8/29.5/11.2/14.8/10.0 (+1 > 0, non-monotone);
Spearman(coef,$)=−0.33 (p=0.01) and Welch −2 vs +2 p=0.008 (the endpoint gap is real), but
Spearman($,coherence)=−0.67 (p=4e-9) — the model "gives more" exactly where it becomes
incoherent (coh −2:76 → +2:94; −2 endpoint n=11 with $100 outliers, drop top → $40). So the
v1 giving effect is **entangled with coherence degradation** and must be reported with
coherence as a covariate; it is **not a clean monotonic giving axis**.

→ "the RLHF direction" splits into a **verbal/elaboration-leaning axis (v2)** and a
**behavioral-leaning axis (v1)**. Both "v_RLHF steers behavior" and **its sign** are
**extraction-basis dependent**. Caveat: Dictator $ n≈11–12/coef, noisy; coherence collapses
at extreme coef; v1 giving entangled with coherence. (See [Phase III §3.3](03_rlhf_axis.md).)

---

## B. Global limitations / caveats

State these on every result:

- **Judge-as-instrument coupling (key measurement confound)** — the LLM judge's altruism
  score is positively coupled to answer length (within-arm r ≈ +0.2) even at fixed steering
  coef — it partly functions as a verbosity detector. (The altruism~coherence coupling does
  NOT reproduce as previously claimed — it is ~−0.05 to −0.11, weak/near-null; the verbosity
  confound rests on the alt~words ≈ +0.2 coupling alone.) All verbal-altruism claims are
  reported with and without length control; where
  length control changes the conclusion (the base→instruct talk-vs-act verbal gap → ~0,
  n.s.) we say so. By contrast the v_RLHF(v2) **steering** verbal effect is length-robust
  (coef slope survives, p=5e-4) — these two "verbal altruism up" effects are causally
  different and must not be bundled.
- **Single-layer / cross-family geometry (per-family)** — orthogonality and projection
  cosines are read at **L20 only**. Layer-dependence is **per-family, not universal**: on
  **Qwen2.5** it is strong — several axes peak ~0.7–0.78 at the **LAST layer L28** (a
  boundary/unembedding-magnitude artifact: |v_RLHF| balloons 20@L20→143@L28), and some axes
  genuinely flip sign across layers (Qwen2.5 refusal −0.42@L0 → +0.78@L28; sycophancy
  −0.68@L0 → +0.32@L28). On **Qwen3/Llama** layer-dependence is **mild** (global max|cos|
  0.50 / 0.33; max/L20 ratio ~1–3×) and L20 is broadly representative; Qwen3 refusal is
  negative at every layer (−0.318@L20, −0.332@L26 — NOT a flip). L20 is also a **different
  relative depth per model** (20/28, 20/36, 20/32 = 0.71/0.56/0.63), so cross-family
  magnitude comparisons are approximate. Report layer profiles, not bare L20; the L28/L0
  peaks are Qwen2.5-only boundary artifacts.
- **Axis collinearity / non-identifiable loadings** — the 7 instruction-contrast axes are
  near-synonymous (a single correlated "elaboration" cluster); only Σcos² (14.3/28.5/19.4)
  is reproducible. The **7-axis joint R² (6.2/19.3/10.7%) is PROVISIONAL** — the axis tensors
  and lstsq script are NOT shipped in this release, and joint R² is not a function of the
  published cosines (it needs the 7×7 Gram). Joint R² < sum of single-axis cos² in every
  family (6.2 vs 14.3; 19.3 vs 28.5; 10.7 vs 19.4) = multicollinearity signature → per-axis
  loadings/R² are basis-dependent and **not individually identifiable**.
  Orthogonality reads are therefore **per-family**, not a cross-family consensus (altruism is
  the smallest axis only on Qwen2.5; Qwen3's largest axis is refusal; "anti-sycophancy" holds
  only on Qwen2.5, ~0 on Qwen3/Llama).
- **Axis vectors are instruction-contrast operationalizations (wording-sensitive)** — they
  are not canonical trait definitions.
- **Base-model coherence confound** — base ~60 coh vs instruct ~95.
- **Small n / inferential unit (key).** Dictator $ is judge-extracted and noisy, n≈11–30/cell
  (base 13–24 after the coherence filter). Critically, **"n=30/game" = 30 samples of ONE
  prompt, not 30 independent games**: samples within a game are not independent, so per-game
  p-values/CIs test "instruct ≠ base FOR THIS PROMPT" — a **within-prompt demonstration, not a
  population inference** (between-prompt replication = 0, effective df ≈ 1 per game). For this
  reason we do **NOT** describe any result as "Bonferroni-surviving" — that would treat
  within-prompt samples as independent observations (a category error). Future work: ≥5–10
  distinct game framings with prompt as a random effect.
- **v_altruism via matched instruction-contrast** — not the full paper filter pipeline.
- **Over-steering coherence collapse at high |coef|.**
- **Dictator $ is judge-extracted and noisy.**
- **Talk-vs-act scoping** — the "act" (behavioral giving ↓) side is a **single-prompt
  demonstration on Qwen2.5 only** (Dictator −22.7, perm p=0.0009 over that one prompt's
  ~30+19 samples — a within-prompt demonstration, NOT a population/Bonferroni claim); Qwen3 is
  **inconclusive** (every per-game CI crosses 0). The "talk" (verbal ↑) side is a **length
  artifact** that vanishes under length control (+5.5/+7.9 → −0.6/−0.3 n.s. under coh≥50);
  this null is **coh≥50-contingent** — UNFILTERED the instruct coef survives length control
  (+6.6/+7.8, p<0.001), so the verbal gap is confounded jointly by length AND coherence. Do
  not claim cross-family OR population generalization for either side.
- **Llama uses the NousResearch ungated mirror of Llama-3.1-8B.**
- **Forward-tracing is fixed-text / preliminary** — fixed-text, not generation-time, single
  model/layer.
