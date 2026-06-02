# Paper skeleton — RLHF as a persona vector: a separable, trait-orthogonal alignment axis

**Status:** draft skeleton, 2026-05-31. Numbers are current results; Phase-1/2 mechanism marked PENDING.

## Working title (options)
1. *Sounding Kind, Acting Cautious: RLHF Moves Models Along a Trait-Orthogonal Axis*
2. *The Alignment Axis Is Not the Altruism Axis: A Persona-Vector View of RLHF Across Model Families*

## Abstract (draft)
Persona vectors let us steer behavioral traits along directions in activation space. We treat **RLHF itself** as a persona vector, `v_RLHF = mean_h(Instruct) − mean_h(base)`, and study its geometry on three model families (Qwen2.5-7B, Qwen3-8B, Llama-3.1-8B). At layer 20, `v_RLHF` is small along the altruism direction (and ⊥ seven measured personality traits), though altruism is the *smallest* axis only on Qwen2.5 (single-layer read; the ordering shifts at other layers and is per-family). "The RLHF direction" is **not a single robust steering vector**: two extraction bases — a matched-basis (response-pooled) v2 and a raw-text prompt-pooled v1 — have cos(v1,v2)=0.31 (reproducible) and produce **dissociated** effects: v2 moves the *verbal* altruism score (length-robust, p=5e-4) but does **not** move Qwen3 Dictator giving (flat, ρ=0.95), while v1 leaves the verbal score flat and its giving effect declines on net but is non-monotone and entangled with coherence collapse. Decomposing `v_RLHF` against behavioral axes, the axes are collinear (an "elaboration" cluster) so per-axis loadings are not individually identifiable; seven named axes jointly explain only 6–19% (a 3× per-family spread), so most of `v_RLHF` is uncharacterized. Separately (a *weight*-change effect, distinct from steering), the base→instruct change *lowers actual giving* on **Qwen2.5 only** (Dictator −22.7, perm p=0.0009 — a single-prompt within-prompt demonstration, not a population claim; Qwen3 inconclusive, all per-game CIs cross 0); the apparent *verbal* rise is a length artifact that vanishes under length control. We also reproduce subliminal preference transfer through number sequences, show the signal is linearly decodable from activations, and find subliminal SFT does not move the persona vector but generically erodes `v_RLHF`.

## Contributions
1. **RLHF-as-persona-vector** framing + matched-basis extraction enabling cosines with trait vectors.
2. **Orthogonality (per-family, single-layer L20):** `v_RLHF` small along altruism (smallest axis only on Qwen2.5) and ⊥ 7 paper traits; reads are per-family and layer-dependent, not a cross-family consensus.
3. **v1/v2 dissociation:** cos(v1,v2)=0.31 (reproducible); v2 moves the verbal score (length-robust), v1 leans on giving (entangled with coherence) — not a single robust steering vector.
4. **Decomposition:** axes collinear → loadings not identifiable; 7 named axes explain only 6–19% (honest bound), so `v_RLHF` is mostly uncharacterized.
5. **Talk-vs-act gap:** the actual-giving drop holds on **Qwen2.5 only** (Dictator single-prompt within-prompt demonstration, perm p=0.0009); Qwen3 inconclusive. The verbal rise is a length artifact (distinct from the length-robust v2 steering verbal effect).
6. **Mechanism** [PENDING Phase 1/2]: test whether any single axis mediates (no single mediator so far); forward-tracing of the orthogonal perturbation.
7. **Subliminal tie-in:** transfer is real + activation-decodable, but does not move the persona vector — it generically erodes `v_RLHF`.

## Sections

### 1. Introduction
Implicit signals (training data, post-training) shape model values. Question: can we *measure* and *steer* these in activation space, and what does RLHF actually do? Three lenses: subliminal transfer, persona vectors, RLHF axis.

### 2. Background & related work
- Persona vectors / steering (Sun & Zhang).
- Subliminal learning (Cloud et al. 2025).
- RLHF side-effects: sycophancy (Papadatos et al.), over-refusal (Cui OR-Bench), Dictator-game behavior across RLHF stages (Huang et al. arXiv:2410.21359), persona drift (Chen et al. arXiv:2507.21509).
- Qwen2.5 vs Qwen3 post-training (arXiv:2412.15115, 2505.09388).

### 3. Phase I — Subliminal preference transfer
- Pipeline: teacher (animal-prompted) → number sequences → filter → student SFT → eval.
- Results: GPT-4.1-nano owl 12.4→56.7% (+44.3pp); Qwen 4B cat 4.0→10.2% (boost); Qwen 0.8B dog protection-from-forgetting; nano-cat null (dominant prior resists). Two modes: boosting vs protection. LR "subliminal window" (2e-5 catastrophic, 5e-6 ok).
- Hidden-state: biased vs neutral number sequences linearly separable **89.5%** (z=−330, p<0.001) — signal is in activations.

### 4. Phase II — Persona vectors in games (replication)
- Method: `v_trait = mean_h(pos) − mean_h(neg)`, steer at layer 20, read out in Dictator/Trust/Ultimatum.
- Results: monotonic sweep; baseline ≈22 (paper 20); Dictator $17→$83; magnitude gap ~2× (L2-norm); over-steering collapse (coh→29 @8B); bigger=sharper+fragile. Trustworthy coef range [+1,+2].

### 5. Phase III — RLHF as a persona vector
**5.1 Orthogonality (per-family reads, single-layer L20).** `cos(v_RLHF, v_altruism)` small on Qwen2.5 (−0.03), Qwen3 (+0.13), Llama (+0.11) — but altruism is the *smallest* axis ONLY on Qwen2.5; on Qwen3/Llama the near-zero axis is **sycophancy** (+0.002/−0.023). Also ⊥ 7 paper traits (max |cos| 0.11, Qwen2.5). CAVEAT (per-family): on Qwen2.5, L20 is an attenuated trough and some axes genuinely flip sign across layers (refusal −0.42@L0 → +0.78@L28; sycophancy −0.68@L0 → +0.32@L28), with several axes peaking ~0.7–0.78 at the LAST layer L28 — a boundary/unembedding artifact (|v_RLHF| balloons 20@L20→143@L28), so those peaks are Qwen2.5-only. On Qwen3/Llama layer-dependence is MILD (global max|cos| 0.50/0.33; max/L20 ratio ~1–3×) and L20 is broadly representative; Qwen3 refusal is negative at every layer (−0.318@L20, −0.332@L26 — NOT a flip). L20 is also a different relative depth per model (0.71/0.56/0.63) → report layer profiles, not bare L20. Figure: per-layer cos profiles; norms.
**5.2 v1/v2 dissociation (a key result).** A ±2 dose-response (judged, coh≥50, all at max_tokens=1024) shows the two extraction bases of `v_RLHF` (cos 0.31, reproducible) do *different things* — though the v1 giving effect is entangled with coherence, so the dissociation is real but not "clean":
- **matched-basis `v_RLHF` (v2):** *verbal* altruism rises with +coef on Qwen2.5 (13→32, monotone) and Qwen3 (5→55, monotone); on Llama it is **non-monotone** (coh≥50: 11.7→7.1→16.0→22.9→31.6; −2 cell n=14/72, mean coh 38). The *actual* Dictator $ does **not** track — Qwen3 is **FLAT** (Spearman +0.009, p=0.95; the "29→9" is a single high −2 endpoint, not a trend). Answer length grows with coef, but the v2 *verbal* effect **survives length control** (see 5.5) → v2 is a **verbal/elaboration-correlated** axis, not behavioral giving.
- **raw-basis `v_RLHF` (v1):** *verbal* altruism is **flat** (~22 across coef), but the *actual* Dictator $ **declines on net but is NOT monotonic** and is **confounded with coherence collapse**: coef −2/−1/0/+1/+2 = $45.8/29.5/11.2/14.8/10.0 (+1 > 0, non-monotone); Spearman(coef,$)=−0.33 (p=0.01), Welch −2 vs +2 p=0.008 (endpoint gap real), but Spearman($,coherence)=−0.67 (p=4e-9) — "gives more" co-occurs with becoming incoherent (−2 endpoint n=11, $100 outliers, drop top → $40). Report with coherence as a covariate → a **behavioral-leaning** axis entangled with coherence, **not** a clean monotonic giving axis.
So "the RLHF direction" is **not a single robust steering vector**: v1 and v2 (cos(v1,v2)@L20=0.31, **reproducible** — both shipped at `results/vectors/v_rlhf_v{1,2}_qwen25.pt`, per-layer L10 .092/L15 .213/L20 .310/L25 .331/L28 .871) produce **dissociated** effects (v2 moves the verbal score, v1 leans on giving) — both whether `v_RLHF` steers behavior and its sign are extraction-dependent. The clean core: (i) cos(v1,v2)=0.31 reproducible; (ii) v2's verbal effect is length-robust; (iii) v2 does not move Qwen3 giving (flat, ρ=0.95). (safety/deliberation behave like v2: verbal↑ via length, $ flat.) Caveat: Dictator $ n≈11–12/coef noisy; coherence collapses at extreme coef; v1 giving entangled with coherence.
**5.3 Decomposition (7 axes × 3 families) — PER-FAMILY, loadings NOT identifiable.** Largest axis = deliberation on Qwen2.5/Llama but **refusal (−0.318) on Qwen3**; safety/caution +0.16/+0.28/+0.24; verbosity moderate; "anti-sycophancy" only on Qwen2.5 (−0.182, ~0 on Q3/Llama); altruism smallest **only on Qwen2.5**. Joint R² 6.2/19.3/10.7% **[PROVISIONAL — the 7 axis tensors + lstsq script are NOT shipped in this release, and joint R² is not a function of the published cosines (needs the 7×7 Gram); only Σcos² 14.3/28.5/19.4 reproduces]** (a 3× spread, NOT "consistent") — most uncharacterized; and joint R² < sum single-axis cos² in every family (6.2 vs 14.3; 19.3 vs 28.5; 10.7 vs 19.4) = **multicollinearity signature** → per-axis loadings (deliberation/safety/verbosity/formatting = one "elaboration" cluster) are **not individually identifiable**. Single-layer L20; basis/wording-dependent upper bounds. Ship the R² + VIF/condition-number script. Figure: grouped bars + per-layer profiles.
**5.4 Talk-vs-act gap (n=30, base→instruct WEIGHT change; "n=30"=30 samples of ONE prompt/game, base n=13–24, coh≥50).** The *verbal* rise (Q2.5 +5.5, Q3 +7.9) is a **length artifact** — instruct answers are far longer (Q2.5 156→249, Q3 235→369 words) and under length control it VANISHES (OLS altruism~instruct+log(words): Q2.5 −0.44 p=0.85, Q3 +0.09 p=0.98). This weight-change verbal effect is DIFFERENT from the length-robust v_RLHF(v2) *steering* verbal effect (5.2/5.5) — do NOT bundle them. The *actual-giving* drop holds on **Qwen2.5 ONLY** (Dictator −22.7, perm p=0.0009 over the prompt's ~30+19 samples — a within-prompt DEMONSTRATION, not a population claim; effective df ≈ 1 per game; Ultimatum −18.5/Transfer −21.3/Commons −7.3 CIs exclude 0 uncorrected); **Qwen3 is inconclusive** — every per-game CI crosses 0 (Dictator −2.7 [−16.2,+9.7], Trust −13.1, Ultimatum −6.9, Commons −2.4, Transfer +16.6 wrong direction). Do NOT claim cross-family generalization. (Commons sign-inverted; q4_cooperated dropped, degenerate.) Table.
**5.5 Mechanism.** *Sufficiency (Phase 1, done):* steering `v_RLHF`(v2), `v_safety`, OR `v_deliberation` each **increase** judge altruism (verbal + game $) at coef +1/+2 on Qwen2.5; all collapse at +3 (over-steering, coh→~50). **No single mediator** — safety is not privileged (deliberation is as strong/stronger). Generalizes to Llama (+v_RLHF raises verbal 13→38, Trust 5→45 = causal D). **Caveat (important):** deliberation being the strongest "lifter" suggests a substantial *judge-elaboration confound* — steering toward elaborate/cautious directions makes answers more elaborate, which the LLM judge scores as more altruistic; length-control is needed to separate genuine giving from eloquence. *Length-control (done):* at matched magnitude, +0→+2 inflates answer length very differently — `v_RLHF`(v2) +16% (255→296 words), `v_safety` +120%, `v_deliberation` +178%. So safety/deliberation "sufficiency" is **largely a judge-elaboration artifact** (they balloon length, judge reads verbosity as generosity); the `v_RLHF`(v2) effect is comparatively **length-robust** (altruism 27→36 with only +16% length) — more credibly a real directional effect. Net: "orthogonal `v_RLHF`(v2) causally raises altruism" survives length-control; no clean single component mediator. *Forward-tracing (Phase 2, preliminary):* injecting +v_RLHF@L20 rotates the final-layer representation (cosine, fixed text) *away* from both v_altruism (0.27→0.15) and v_safety (0.60→0.42), not toward — so the behavioral altruism increase is **not** "rotation into the altruism readout"; favors a **separate gate/pathway**. Caveat: fixed-text, not generation-time → preliminary; a generation-time trace is needed to finalize. *Still open:* reconcile v1 vs v2 opposite steering signs (which components differ between bases).

### 6. The subliminal–RLHF link
H1 (subliminal SFT moves persona vector) rejected: cos(v_student,v_teacher)=0.996, drift −0.31. [PROVENANCE: the 0.996 / drift −0.31 are from the working repo; the computing artifact is NOT shipped in this release (e_dealignment_report.json contains the −0.26/−0.35 de-alignment cosines only) → "from working repo, not independently checkable here."] But subliminal SFT moves activations along −v_RLHF (de-alignment, cos −0.26 @L20) — yet owl-control de-aligns *more* (−0.35 @L20) → generic to numeric SFT, not trait-specific. Unifies: subliminal numbers erode RLHF generically; they don't transmit a trait.

### 7. Limitations
Single-layer cosines (L20); axis vectors are instruction-contrast operationalizations (wording-sensitive); base-model coherence confound (base ~60 coh vs instruct ~95); small n per game (base 13–24); v_altruism via matched instruction-contrast (not full paper filter pipeline); steering-collapse artifact at high coef.

### 8. The "we killed our own claims" appendix (rigor)
- Cross-generation inversion = truncation artifact (250→1024 tokens; −8.2→+13.8). Retracted.
- "Qwen3 = rational refuser" — 0% explicit refusals. Rejected.
- Subliminal de-alignment specificity — owl control killed it. Generic.

## Figures (have / need)
- HAVE: altruism_sweep_4cond.png, v_rlhf_vs_v_alt.png, fig4_effect_comparison.png, fig6_hidden_states.png.
- NEED: 7-axis grouped bars × 3 families; talk-vs-act n=30 table→bar; mediation (giving vs coef per condition) [Phase 1]; forward-tracing projection-by-layer [Phase 2].

## Claims table (verdicts)
| Claim | Verdict |
|---|---|
| Subliminal transfer real (3 models) | holds |
| Subliminal signal activation-decodable (89.5%) | holds |
| Persona-vector sweep reproduces paper | holds |
| v_RLHF ⊥ altruism (smallest axis), yet steering moves behavior | Qwen2.5 only; on Q3/Llama altruism ~0.11–0.13 (small but not smallest); single-layer L20 |
| v1 vs v2 steer the same behavior / single robust sign | NO — cos 0.31 (reproducible), dissociated; v1 giving entangled with coherence, not "monotonic" |
| v_RLHF ≈ deliberation + safety/caution, not sycophancy | per-family (Q3 largest = refusal); axes collinear → loadings not identifiable |
| RLHF *weight change*: actual giving ↓ (talk-vs-act) | Qwen2.5 Dictator single-prompt within-prompt demonstration (×30, perm p=0.0009, df≈1); Qwen3 inconclusive (all CIs cross 0); NOT a population/cross-family claim |
| RLHF *weight change*: VERBAL altruism ↑ | artifact — vanishes under length control (−0.6/−0.3 n.s.) |
| v_RLHF(v2) STEERING raises verbal altruism, length-robust | holds (coef slope survives length control, p=5e-4) — distinct from the weight-change verbal effect |
| which direction reproduces matched-basis v_RLHF steering | PENDING (Phase 1) |
| Subliminal SFT moves persona vector (H1) | rejected |
| Subliminal de-alignment is altruism-specific | rejected (generic) |
| Cross-generation RLHF inversion | dead (truncation) |
