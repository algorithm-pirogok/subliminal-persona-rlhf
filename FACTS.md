# FACTS — single source of truth (do NOT invent numbers; cite only from here)

This file is the authoritative record of results for the showcase repo. All READMEs,
section docs, and examples MUST use these numbers verbatim and reproduce these caveats.
Every quantitative claim must be hedged exactly as written here.

## One-line framing
We study how *implicit* signals (training data, steering directions, RLHF post-training)
move a language model's preferences/values, and whether we can measure that movement in
activation space — across three lenses (subliminal transfer, persona vectors, the RLHF axis).

Models: GPT-4.1-nano, Qwen2.5-7B(-Instruct), Qwen3-8B(-Base/Instruct), Qwen3.5-0.8B/4B,
Llama-3.1-8B (NousResearch ungated mirror). Judge: OpenAI GPT-4.1-mini. Compute: 7×A4000
(server "beleriand") + Kaggle T4×2.

---

## PHASE I — Subliminal preference transfer (animal preference)
Reproduce Cloud et al. 2025: teacher with hidden animal preference generates NUMBER
sequences; student fine-tuned on them inherits the preference (zero animal semantics in data).
Pipeline: generate 30k seqs → filter (animal-word + format, 62–90% pass) → SFT on 10k →
eval 50 prompts × 200 samples, normalized animal mentions.

Results (effect type):
- GPT-4.1-nano, owl: 12.4% → **56.7%** (+44.3pp) — boosting
- Qwen 4B, cat: 4.0% → **10.2%** (2.5×) — boosting (1 epoch)
- Qwen 0.8B, dog: 15.5% → 12.9% — "protection from forgetting" (dog declines −17% vs cat −79%, lion −61%)
- GPT-4.1-nano, cat: 22.2% → 22.9% (+0.7) — null (dominant dolphin prior resists)
Two modes: boosting (large models) vs protection (small). LR "subliminal window": 2e-5 →
catastrophic forgetting on 4B; 5e-6 preserved instruction-following.
Hidden-state test (Qwen 4B, last layer): biased vs neutral number sequences are linearly
separable **89.5%** (±0.5, 5-fold); permutation z = −330, p<0.001; centroid cosine 0.971.
→ the teacher's preference leaves a statistically detectable trace in pure-number outputs.

## PHASE II — Persona vectors in games (replication of Sun & Zhang on Qwen)
Persona vector v_trait = mean response activation (trait-on) − (trait-off) system prompts;
steer by adding c·v at layer 20; read out behavior in Dictator/Trust/Ultimatum games (LLM judge).
Results: monotonic altruism sweep; baseline ≈ 22 (paper 20); Dictator giving $17 → $83 at peak;
~2× magnitude gap (L2-norm) — paper c=+2 (81.5) ≈ our c=+5 (88.5); over-steering collapse
(coherence → 29 on 8B at high coef); bigger model = sharper but more fragile. Trustworthy
coef range [+1,+2].

## PHASE III — RLHF as a persona vector
v_RLHF = mean_h(Instruct) − mean_h(base), measured on identical prompts (matched basis =
chat-template, response-pooled = "v2"; raw-text/prompt-pooled = "v1").

### 3.1 Orthogonality, general across 3 families — cos(v_RLHF, axis) @ layer 20
| axis | Qwen2.5 | Qwen3 | Llama-3.1 |
|---|---|---|---|
| altruism | −0.028 | +0.134 | +0.106 |
| refusal | +0.043 | −0.318 | −0.179 |
| verbosity | +0.181 | +0.176 | +0.137 |
| deliberation | +0.204 | +0.238 | +0.272 |
| sycophancy | −0.182 | +0.002 | −0.023 |
| safety_caution | +0.162 | +0.278 | +0.237 |
| formatting | +0.081 | −0.030 | +0.031 |
**7-axis joint R² (lstsq, L20): Qwen2.5 6.2%, Qwen3 19.3%, Llama 10.7% [PROVISIONAL — not reproducible
from shipped artifacts: the 7 axis tensors + lstsq script are NOT in this release, and joint R² is not a
function of the published cosines (needs the 7×7 Gram); only Σcos² 14.3/28.5/19.4 is reproducible].** A 3× spread, NOT
"consistent"; and joint R² < sum of single-axis cos² in every family (6.2 vs 14.3; 19.3 vs 28.5;
10.7 vs 19.4) = MULTICOLLINEARITY signature → per-axis loadings are NOT individually identifiable
(verbosity/deliberation/safety/formatting are a single correlated "elaboration" cluster). R²-computing
script must be shipped; current numbers are basis/wording-dependent upper bounds. So state: 7 named
axes explain only ~6–19% → most of v_RLHF is uncharacterized (the multicollinearity strengthens this).

Reads are PER-FAMILY, not a cross-family consensus (do not generalize):
- altruism is the *smallest* axis ONLY on Qwen2.5 (|cos|=0.028); on Qwen3 it is +0.134 (rank 3) and
  Llama +0.106 (rank 3), where SYCOPHANCY is the near-zero axis (0.002 / 0.023).
- largest axis = deliberation on Qwen2.5/Llama, but REFUSAL (−0.318) on Qwen3.
- "anti-sycophancy" holds ONLY on Qwen2.5 (−0.182); ~0 on Qwen3/Llama (+0.002 / −0.023).
SINGLE-LAYER CAVEAT (per-family, do not over-generalize): all cosines are read at L20. On Qwen2.5,
layer-dependence is strong — several axes peak ~0.7–0.78 at the LAST layer L28 (a boundary/unembedding
artifact: |v_RLHF| balloons 20@L20→143@L28) and some axes genuinely flip sign across layers (Qwen2.5
refusal −0.42@L0 → +0.78@L28; sycophancy −0.68@L0 → +0.32@L28). On Qwen3/Llama layer-dependence is
MILD (global max|cos| 0.50 / 0.33; max/L20 ratio ~1–3×) and L20 is broadly representative; Qwen3 refusal
is negative at every layer (−0.318@L20, −0.332@L26 — NOT a flip). L20 is also a different relative depth per model (20/28, 20/36, 20/32 =
0.71/0.56/0.63), so cross-family magnitude comparisons are approximate. Report layer profiles, not bare L20.
What DOES hold across families: altruism is among the smaller axes and v_RLHF is dominated by the
elaboration/refusal cluster, not by a personality trait. Also ⊥ all 7 *paper personality traits* (max |cos| = 0.11, Qwen2.5).

### 3.2 Talk-vs-act gap (RLHF weight change, base→instruct), 30 samples/scenario, judged, coh≥50
("n=30" = 30 samples of ONE prompt per game on the instruct side; base side n=13–24 after coherence
filter; samples within a game are NOT independent.)

**The "talk" side is a VERBOSITY artifact, not a content shift.** Raw verbal altruism rises (Qwen2.5
+5.5 16.6→22.2, p=0.007; Qwen3 +7.9 27.0→34.9, p=0.002) — but instruct answers are far longer
(Q2.5 156→249, Q3 235→369 words) and once you control for length the effect VANISHES: OLS
altruism~instruct+log(words) → Q2.5 −0.44 (p=0.85), Q3 +0.09 (p=0.98). (Note: this null is contingent
on the coh≥50 filter; UNFILTERED the instruct coef survives length control (+6.6/+7.8, p<0.001), so the
talk-side effect is confounded jointly by length AND coherence — an even stronger "not a content shift".) The judge scores length as
altruism (within-arm corr altruism~words ≈ +0.2, i.e. the judge partly rewards length; the
altruism~coherence coupling is weak/near-null ~−0.05 to −0.11 and is NOT load-bearing — the verbosity
confound rests on alt~words alone). So state:
"instruct talks MORE, which the judge reads as more altruistic" — NOT "talks more altruistically."
(This is a base→instruct WEIGHT effect; it is DIFFERENT from and must not be bundled with the
v_RLHF(v2) STEERING effect in 3.3/3.5, which DOES survive length control.)

**The "act" side is robust on Qwen2.5 ONLY; Qwen3 is inconclusive.** Actual giving, RLHF Δ per game
(sign-adjusted, + = more generous; Commons sign-inverted from raw fish-caught; 6th game
q4_cooperated dropped — degenerate ~0 in all arms), with bootstrap 95% CIs:
- Qwen2.5: Dictator **−22.7** (perm p=0.0009 over the prompt's ~30+19 samples, stable with/without filter),
  Ultimatum −18.5 [−32.7,−4.1], Transfer −21.3 [−40.0,−3.0], Commons −7.3 [−15.2,−0.3], Trust +2.5 [−13.8,+16.9].
- Qwen3: Dictator −2.7 [−16.2,+9.7] (flips to +UP without coherence filter), Trust −13.1 [−29.9,+3.9],
  Ultimatum −6.9 [−21.1,+7.3], Commons −2.4 [−10.6,+6.9], Transfer **+16.6** (wrong direction).
  → EVERY Qwen3 per-game CI crosses 0; the "falls 4/5 games" is noise on Qwen3.
**INFERENTIAL-UNIT CAVEAT (important):** each "game" is ONE prompt scored ~30× (instruct) / 13–24× (base),
so these p-values/CIs test "instruct ≠ base FOR THIS PROMPT" — a within-prompt DEMONSTRATION, not a
population claim about RLHF (between-prompt replication = 0, effective df ≈ 1 per game). Do NOT call any
of this "Bonferroni-surviving" — that treats within-prompt samples as independent observations.
→ VERDICT: the strongest behavioral signal is the Qwen2.5 Dictator gap (−22.7), but it is a single-prompt
demonstration; Qwen3 inconclusive (all CIs cross 0). **Do NOT claim cross-family OR population
generalization for the behavioral gap.** Future work: ≥5–10 distinct game framings, prompt as random effect.
Caveat: judge-extracted $, n=11–30/cell.

### 3.3 v1/v2 steering DISSOCIATION (±2 dose-response, max_tokens=1024, judged, coh≥50)
cos(v1, v2) on the SAME model (Qwen2.5) — REPRODUCIBLE, both vectors shipped at
results/vectors/{v_rlhf_v1_qwen25,v_rlhf_v2_qwen25}.pt. Per layer: L10 +0.092, L15 +0.213,
L20 **+0.310**, L25 +0.331, L28 +0.871 (they converge at the readout layer). Caveat: v1 is
prompt-pooled, v2 response-pooled — the differing pooling basis is itself part of why they differ.
- **v_RLHF v2** (matched basis): VERBAL altruism rises with +coef on Qwen2.5 (13→32, monotone) and
  Qwen3 (5→55, monotone); on Llama it rises on the positive arm but is non-monotone with a
  coherence-collapsed low end (coh≥50: 11.7→7.1→16.0→22.9→31.6; −2 has n=14 of 72, mean coh 38).
  Actual Dictator $ does NOT track: Qwen2.5 noisy (Spearman coef,$ ≈ +0.25, p=0.06 — if anything
  weakly POSITIVE), Qwen3 FLAT (Spearman +0.009, p=0.95; the "29→9" is a single high −2 endpoint,
  not a trend). Answer length grows with coef. (NB: v2 ALSO has a $–coherence confound, Spearman
  ($,coh)≈−0.57 — comparable to v1's −0.66 — so disclose it symmetrically; verdict unchanged, v2 $ flat.)
  → v2 ≈ verbal/ELABORATION axis (but its verbal effect SURVIVES length control — see 3.5 — so it's
  "length-correlated, not length-explained").
- **v1** (raw basis @1024): verbal FLAT (~22 across coef). Dictator $ DECLINES ON NET but is NOT
  monotonic and is CONFOUNDED with coherence collapse: coef −2/−1/0/+1/+2 = $45.5/29.5/11.2/14.8/10.0
  (coh≥50; +1 > 0, non-monotone); Spearman(coef,$)=−0.33 (p=0.01), but Spearman($,coherence)=−0.66 (p≈4e-9)
  — i.e. "gives more" co-occurs with becoming incoherent (coh −2:79.9 → +2:94). −2 endpoint n=11 with
  $100 outliers (drop top → $40). Welch −2 vs +2 p=0.008 (the endpoint gap is real; "monotonic" is NOT).
  → v1 is a behavioral-leaning axis but the giving effect is entangled with coherence degradation; report
  with coherence as covariate. Do NOT call it a "clean monotonic giving axis."
- safety & deliberation behave like v2: verbal↑ via length-explosion (deliberation Qwen3 words
  28→644; Llama 7→779; Llama Dictator $ → ~0.2), actual $ flat/down.
→ "the RLHF direction" is not a single robust steering vector: v1 (cos 0.31 with v2) and v2 produce
DISSOCIATED effects (v2 moves the verbal score, v1 leans on giving), and both magnitude and sign are
extraction-basis dependent. The CLEAN, defensible core is: (i) cos(v1,v2)=0.31 reproducible; (ii) v2's
verbal effect is length-robust (3.5); (iii) v2 does not move Qwen3 Dictator giving (flat, ρ=0.95).
Caveat: Dictator $ n≈11–12/coef, noisy; coherence collapses at extreme coef; v1 giving entangled with coherence.

### 3.4 Mechanism (forward-tracing, preliminary)
Injecting +v_RLHF(v2) at L20 (fixed text, cosine) rotates the final-layer representation AWAY
from both v_altruism (cos 0.27→0.15) and v_safety (0.60→0.42), NOT toward → behavioral change
is not "rotation into the altruism readout"; favors a separate gate/pathway. CAVEAT: fixed-text,
not generation-time, single model/layer → preliminary.

### 3.5 Sufficiency / no single mediator (length-control)
Steering v_RLHF(v2), v_safety, OR v_deliberation each raise judge altruism at coef +1/+2 on
Qwen2.5 (collapse at +3, coh→~50). NOT safety-specific. Length-control: at matched magnitude,
+0→+2 inflates length very differently (all Qwen2.5, from the steer_q25_rlhf/safety/delib arms —
do NOT use the day_* arms here, which mix models and give different %): v_RLHF(v2) +16%, v_safety +120%, v_deliberation +178%.
→ safety/deliberation "sufficiency" is largely a judge-elaboration artifact; but the v_RLHF(v2)
verbal effect is LENGTH-ROBUST and this is the key contrast with 3.2: OLS altruism~coef+log(words)
on day_rlhf_q25 keeps a significant coef slope (4.64 raw → 3.79 with length control, p=5e-4; → 4.10
with length+coherence), and within-coef judge-altruism~length R²≈4% (r≈0.2). So v2 steering raises
judge-altruism BEYOND what length explains, whereas the 3.2 base→instruct verbal gap does NOT survive
length control (→ ~0, n.s.). These two "verbal altruism up" effects are causally different and must be
reported separately; bundling them lets the robust v2 result lend false credibility to the fragile 3.2 one.
Its behavioral $ still does not track (see 3.3).

## THE SUBLIMINAL ↔ RLHF LINK
H1 (subliminal SFT moves the persona vector) REJECTED: cos(v_student, v_teacher)=0.996, drift −0.31.
[PROVENANCE: 0.996 / drift −0.31 / the 15.8–10.1–25.7 altruism triplet are from the working repo; the
computing artifact is NOT shipped here (e_dealignment_report.json contains the −0.26/−0.35 de-alignment
cosines but not these). Treat as "from working repo, not independently checkable in this release."]
Subliminal SFT shifts activations along −v_RLHF (de-alignment, cos −0.26 @L20) — BUT an owl-control
student (generic numeric SFT) de-aligns MORE (−0.35 @L20) → GENERIC to numeric SFT, not altruism-specific.
SINGLE-LAYER CAVEAT: the −0.26/−0.35 are L20-specific; at the last layer L28 both flip POSITIVE
(student +0.75, owl +0.69, a boundary artifact). The ROBUST claim is the owl > student ORDERING (owl
de-aligns more at every layer), NOT the absolute sign.
Behaviorally student altruism 15.8 sits between base 10.1 and instruct 25.7 (erodes RLHF veneer).

## RIGOR — claims we KILLED ourselves (lead with these; they show integrity)
- "Cross-generation RLHF inversion / Qwen3 RLHF reduces altruism" = TRUNCATION ARTIFACT.
  86.5% of Qwen3-Instruct answers cut off at max_tokens=250; at 1024: altruism 14.0→36.0,
  Dictator $0.1→$17.5, RLHF Δ −8.2 → +13.8. No inversion. RETRACTED.
- "Qwen3 = rational refuser ($0 = strategy)" — 0% explicit refusals (classifier) → rejected.
- "Subliminal SFT de-aligns specifically" — owl control killed it (generic).
- "v_RLHF steers giving DOWN" was conflated v1/v2 (cos 0.31, different vectors) — fixed; see 3.3.

## CLAIMS TABLE (verdicts)
| Claim | Verdict |
|---|---|
| Subliminal transfer real (boosting/protection, 3 models) | holds |
| Subliminal signal linearly decodable from activations (89.5%) | holds |
| Persona-vector sweep reproduces the paper | holds |
| v_RLHF ⊥ altruism (smallest axis), yet steering moves behavior | Qwen2.5 only; on Q3/Llama altruism ~0.11–0.13 (small but not smallest); single-layer L20 |
| v_RLHF ≈ deliberation + safety/caution, not sycophancy | per-family (Q3 largest = refusal); axes collinear → loadings not identifiable |
| RLHF weight-change: actual giving ↓ (talk-vs-act) | Qwen2.5 Dictator demonstration (single prompt ×30, p=0.0009 within-prompt); Qwen3 inconclusive (all CIs cross 0); NOT a population/cross-family claim |
| RLHF weight-change: VERBAL altruism ↑ | artifact — vanishes under length control (−0.6/−0.3 n.s.) |
| v_RLHF(v2) STEERING raises verbal altruism, length-robust | holds (coef slope survives length control, p=5e-4) — distinct from the weight-change verbal effect |
| v1 vs v2 steer the same behavior / single robust sign | NO — cos 0.31 (reproducible), dissociated; but v1 giving entangled with coherence, not "monotonic" |
| Subliminal SFT moves persona vector (H1) | rejected |
| Subliminal de-alignment is altruism-specific | rejected (generic) |
| Cross-generation RLHF inversion | dead (truncation) |
| safety/caution is THE causal mediator | not supported (no single mediator) |

## GLOBAL CAVEATS (state in limitations)
- **Judge-as-instrument coupling (key):** the LLM judge's altruism score is positively coupled to
  answer length (within-arm r ≈ +0.2) even at fixed steering coef — it partly functions as a verbosity
  detector. (The altruism~coherence coupling does NOT reproduce as claimed — it is ~−0.05 to −0.11,
  weak/near-null; the confound rests on length alone.) All verbal-altruism claims are reported
  with and without length control; where length control changes the conclusion (3.2) we say so.
- **Single-layer / cross-family geometry:** cosines read at L20 only (an attenuated trough; |cos| 3–4×
  larger at other layers, ordering flips); L20 is a different relative depth per model (0.71/0.56/0.63).
- **Axis collinearity:** the 7 instruction-contrast axes are near-synonymous (elaboration cluster) →
  per-axis loadings/R² are basis-dependent and not individually identifiable.
- **Small n:** Dictator $ is judge-extracted and noisy, n≈11–30/cell (base 13–24); "n=30/game" = 30
  samples of one prompt, not independent games → per-game p-values are within-prompt demonstrations, NOT
  population inferences (effective df ≈ 1 per game); "Bonferroni-surviving" would be a category error.
- v_altruism via matched instruction-contrast (not full paper filter pipeline); over-steering coherence
  collapse at high |coef|; Llama uses the NousResearch ungated mirror; forward-tracing is fixed-text/preliminary.

## MATERIAL MANIFEST (what's in this repo)
- figures/: fig1_08b_epochs, fig2_4b_steps, fig3_openai, fig4_effect_comparison,
  fig5_relative_decline, fig6_hidden_states (Phase I); altruism_sweep*, symmetry_sweep,
  baseline_money_games (Phase II); v_rlhf_vs_v_alt (Phase III orthogonality+norms).
- results/: e_dealignment_report.json (subliminal↔RLHF cosines), a_axis_decomposition_report.json
  (7-axis × 3 families, per-layer cosines).
  **The talk-vs-act per-game numbers (3.2) come from the t_q25_*/t_q3_* scored arms** (n=30 samples/q,
  coh≥50), NOT from c_dollars_report.json. c_dollars_report.json is a NOISY cross-game pooled regex
  sanity check (c_* arms, n=12/game, no coherence filter, pools heterogeneous games + extraction-rate
  confound) — cite it ONLY as that, never as the source for the headline per-game deltas.
- results/vectors/: v_rlhf_v1_qwen25.pt, v_rlhf_v2_qwen25.pt — matched same-model v1/v2 so cos(v1,v2)=0.31 is reproducible.
- src/: key scripts (extract_rlhf_vector_v2, extract_axis_vector, make_steer_vectors,
  instruct_steered_eval, student_game_eval, judge_csvs, cosine_a/e, trace_steered_projection,
  extract_persona_vector, hidden_state_analysis).
- presentation.pdf (14-slide mentor deck), paper_skeleton.md, subliminal_report.pdf (Phase I write-up),
  docs/full_session_log.md, docs/mechanism_program_design.md.
- Example model outputs (steered generations, talk-vs-act pairs) live in the working repo's
  scored CSVs: /Users/olegkurilov/vsc/subliminal_learning/data/persona_vectors/phase3/<arm>/scored/
  and forjudge/ dirs — columns: question, answer, coef, altruism, coherence, q0_amount_given (Dictator $), etc.
  Arms: day_rlhf_q25/q3/llama, day_v1_q25, day_safety_q3/llama, day_delib_q3/llama, t_q25_*/t_q3_* (talk-vs-act n=30).
