# Results — consolidated statistics reference

All numbers below are reproduced verbatim from `FACTS.md`. Raw numbers live in `results/*.json`
(`e_dealignment_report.json`, `a_axis_decomposition_report.json`). The per-game talk-vs-act numbers
(Table 6) come from the `t_q25_*`/`t_q3_*` scored arms (n=30 samples/q, coh≥50). `c_dollars_report.json`
is a NOISY cross-game pooled sanity check (c_* arms, n=12/game, no coherence filter) — NOT the source
for the per-game deltas. Figures referenced in `figures/`.

---

## Phase I — Subliminal preference transfer

**Table 1.** Animal-preference transfer (teacher → student), normalized animal mentions, eval 50 prompts × 200 samples.

| Model | Animal | Baseline | After | Δ / ratio | Effect type |
|---|---|---|---|---|---|
| GPT-4.1-nano | owl | 12.4% | 56.7% | +44.3pp | boosting |
| Qwen 4B | cat | 4.0% | 10.2% | 2.5× | boosting (1 epoch) |
| Qwen 0.8B | dog | 15.5% | 12.9% | — | protection from forgetting (dog declines −17% vs cat −79%, lion −61%) |
| GPT-4.1-nano | cat | 22.2% | 22.9% | +0.7 | null (dominant dolphin prior resists) |

**Table 2.** Hidden-state separability (Qwen 4B, last layer): biased vs neutral number sequences.

| Metric | Value |
|---|---|
| Linear separability (5-fold) | 89.5% (±0.5) |
| Permutation z | −330 (p<0.001) |
| Centroid cosine | 0.971 |

---

## Phase III — RLHF as a persona vector

**Table 3.** Orthogonality: cos(v_RLHF, axis) @ layer 20, across 3 families.

| axis | Qwen2.5 | Qwen3 | Llama-3.1 |
|---|---|---|---|
| altruism | −0.028 | +0.134 | +0.106 |
| refusal | +0.043 | −0.318 | −0.179 |
| verbosity | +0.181 | +0.176 | +0.137 |
| deliberation | +0.204 | +0.238 | +0.272 |
| sycophancy | −0.182 | +0.002 | −0.023 |
| safety_caution | +0.162 | +0.278 | +0.237 |
| formatting | +0.081 | −0.030 | +0.031 |

**Table 4.** 7-axis joint R² (lstsq, L20). [PROVISIONAL — the 7 axis tensors + lstsq script are NOT
shipped in this release, and joint R² is not a function of the published cosines (needs the 7×7 Gram);
only Σcos² 14.3/28.5/19.4 is reproducible.]

| Family | R² |
|---|---|
| Qwen2.5 | 6.2% |
| Qwen3 | 19.3% |
| Llama | 10.7% |

7 named axes explain only ~6–19% (a 3× spread, NOT "consistent") → most of v_RLHF is uncharacterized.
Joint R² < sum of single-axis cos² in every family (6.2 vs 14.3; 19.3 vs 28.5; 10.7 vs 19.4) =
MULTICOLLINEARITY signature → per-axis loadings are NOT individually identifiable (verbosity/deliberation/
safety/formatting form one correlated "elaboration" cluster). Orthogonality reads are PER-FAMILY, not a
cross-family consensus: altruism is the smallest axis ONLY on Qwen2.5 (|cos|=0.028); on Qwen3/Llama it is
+0.134/+0.106 (rank 3) and SYCOPHANCY is the near-zero axis (0.002/0.023). Largest axis = deliberation on
Qwen2.5/Llama but REFUSAL (−0.318) on Qwen3. SINGLE-LAYER CAVEAT (per-family, do not over-generalize):
all cosines are read at L20. On Qwen2.5, layer-dependence is strong — several axes peak ~0.7–0.78 at the
LAST layer L28 (a boundary/unembedding artifact: |v_RLHF| balloons 20@L20→143@L28) and some axes genuinely
flip sign across layers (Qwen2.5 refusal −0.42@L0 → +0.78@L28; sycophancy −0.68@L0 → +0.32@L28). On
Qwen3/Llama layer-dependence is MILD (global max|cos| 0.50/0.33; max/L20 ratio ~1–3×) and L20 is broadly
representative; Qwen3 refusal is negative at every layer (−0.318@L20, −0.332@L26 — NOT a flip). L20 is a
different relative depth per model (0.71/0.56/0.63), so cross-family magnitude comparisons are approximate.
Also ⊥ all 7 *paper personality traits* (max |cos| = 0.11, Qwen2.5).

**Table 5.** Talk-vs-act gap (RLHF weight change, base→instruct), "n=30" = 30 samples of ONE prompt per
game on the instruct side (base side n=13–24 after coherence filter; samples within a game are NOT
independent), judged, coh≥50. Raw verbal altruism rises — but this is a LENGTH artifact, not a content
shift: instruct answers are far longer (Q2.5 156→249, Q3 235→369 words), and once length is controlled
(OLS altruism~instruct+log(words)) the effect VANISHES.

| Family | Verbal altruism Δ (raw) | (base→instruct) | p (raw) | Length-controlled coef | p (controlled) |
|---|---|---|---|---|---|
| Qwen2.5 | +5.5 | 16.6→22.2 | 0.007 | −0.44 | 0.85 (n.s.) |
| Qwen3 | +7.9 | 27.0→34.9 | 0.002 | +0.09 | 0.98 (n.s.) |

Note: this base→instruct WEIGHT effect is DIFFERENT from the v_RLHF(v2) STEERING verbal effect in Table 7 /
Table 8, which IS length-robust (coef slope survives length control, p=5e-4). Do not bundle them.

**Table 6.** Actual giving (sign-adjusted, + = more generous), RLHF Δ per game, with bootstrap 95% CIs.
Source: `t_q25_*`/`t_q3_*` arms (n=30 samples/q, coh≥50). The "act" side holds on Qwen2.5 ONLY; Qwen3 is
inconclusive (every Qwen3 per-game CI crosses 0).

| Game | Qwen2.5 | Qwen3 |
|---|---|---|
| Dictator | **−22.7** (perm p=0.0009 over the prompt's ~30+19 samples) | −2.7 [−16.2, +9.7] (CI crosses 0) |
| Ultimatum | −18.5 [−32.7, −4.1] | −6.9 [−21.1, +7.3] (CI crosses 0) |
| Transfer | −21.3 [−40.0, −3.0] | +16.6 (wrong direction; CI crosses 0) |
| Commons | −7.3 [−15.2, −0.3] | −2.4 [−10.6, +6.9] (CI crosses 0) |
| Trust | +2.5 [−13.8, +16.9] | −13.1 [−29.9, +3.9] (CI crosses 0) |

**INFERENTIAL-UNIT CAVEAT (important):** each "game" is ONE prompt scored ~30× (instruct) / 13–24× (base),
so these p-values/CIs test "instruct ≠ base FOR THIS PROMPT" — a within-prompt DEMONSTRATION, not a
population claim about RLHF (between-prompt replication = 0, effective df ≈ 1 per game). Do NOT call any
of this "Bonferroni-surviving" — that treats within-prompt samples as independent observations.
VERDICT: the strongest behavioral signal is the Qwen2.5 Dictator gap (−22.7), but it is a single-prompt
demonstration (Ultimatum/Transfer/Commons CIs exclude 0 uncorrected); Qwen3 inconclusive. Do NOT claim
cross-family OR population generalization for the behavioral gap. Footnotes: Commons is sign-INVERTED
from raw fish-caught. A 6th game (q4_cooperated) was DROPPED as degenerate (~0 in all arms). Caveat:
base less coherent (n=13–24 vs inst n=30 samples of one prompt, not independent games); judge-extracted
$, n≈11–30/cell.

**Table 7.** v1/v2 steering dissociation (±2 dose-response, max_tokens=1024, judged, coh≥50). cos(v1, v2)
on the SAME model (Qwen2.5) = **0.31** — REPRODUCIBLE; both vectors shipped at
`results/vectors/v_rlhf_v{1,2}_qwen25.pt`. Per layer: L10 +0.092, L15 +0.213, L20 +0.310, L25 +0.331,
L28 +0.871 (they converge at the readout layer). v1 is prompt-pooled, v2 response-pooled — the differing
pooling basis is itself part of why they differ.

| Vector | Verbal altruism (−2 → +2) | Dictator $ (−2 → +2) | Reading |
|---|---|---|---|
| v_RLHF v2 (matched basis), Qwen2.5 | 13 → 32 (monotone) | does NOT track (Spearman coef,$ ≈ +0.25, p=0.06, if anything weakly +) | verbal/elaboration axis |
| v_RLHF v2, Qwen3 | 5 → 55 (monotone) | FLAT (Spearman +0.009, p=0.95; "29→9" is a single high −2 endpoint, not a trend) | verbal/elaboration axis |
| v_RLHF v2, Llama | 11.7 → 7.1 → 16.0 → 22.9 → 31.6 (non-monotone; −2 has n=14/72, mean coh 38) | does NOT track (flat/noisy) | verbal/elaboration axis |
| v1 (raw basis @1024) | 22 → 22 (flat) | $45.5/29.5/11.2/14.8/10.0 (n=11; declines on net, NON-monotone: +1 > 0; coherence-confounded) | behavioral-leaning axis, entangled with coherence |

NB: v2 ALSO has a $–coherence confound, Spearman($,coh)≈−0.57 — comparable to v1's −0.66 — so it is
disclosed symmetrically; verdict unchanged, v2 $ flat. v1 declines on net but is NOT monotonic and is
CONFOUNDED with coherence collapse: Spearman(coef,$)=−0.33 (p=0.01) vs Spearman($,coherence)=−0.66
(p≈4e-9) — "gives more" co-occurs with becoming incoherent (coh −2:79.9 → +2:94). −2 endpoint n=11 with
$100 outliers (drop top → $40). Welch −2 vs +2 p=0.008
(the endpoint gap is real; "monotonic" is NOT). The v2 verbal effect SURVIVES length control (length-correlated,
not length-explained — see Table 8). safety & deliberation behave like v2: verbal↑ via length-explosion
(deliberation Qwen3 words 28→644; Llama 7→779; Llama Dictator $ → ~0.2 while sounding kind), actual $ flat/down.
Caveat: Dictator $ n≈11–12/coef, noisy; coherence collapses at extreme coef; v1 giving entangled with coherence.

**Table 8.** Sufficiency / length-control (Qwen2.5, +0→+2, matched magnitude). Computed from the
steer_q25_rlhf/safety/delib arms (ALL Qwen2.5, coef 0→+2); do NOT use the day_* arms here, which mix
models and give different % (+12/58/79%).

| Vector | Length inflation (+0→+2) |
|---|---|
| v_RLHF v2 | +16% |
| v_safety | +120% |
| v_deliberation | +178% |

Each raises judge altruism at coef +1/+2 (collapse at +3, coh→~50); NOT safety-specific.

---

## Subliminal ↔ RLHF link

**Table 9.** Subliminal–RLHF cosines.

| Metric | Value |
|---|---|
| cos(v_student, v_teacher) — H1 test | 0.996 (drift −0.31) |
| Subliminal SFT shift along −v_RLHF (de-alignment) | cos −0.26 |
| Owl-control student (generic numeric SFT) de-alignment | cos −0.35 (MORE) |
| Student altruism | 15.8 (between base 10.1 and instruct 25.7) |

H1 (subliminal SFT moves the persona vector) REJECTED. De-alignment is GENERIC to numeric SFT, not altruism-specific.
