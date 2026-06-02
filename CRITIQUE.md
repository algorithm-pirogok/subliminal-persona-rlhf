# CRITIQUE REPORT — Phase III RLHF-axis Showcase (consolidated from 5 adversarial reviews)

## 1. TL;DR

**Verdict: Defensible to show a mentor, but ITERATE on three load-bearing claims first.** All five critics independently re-derived the headline numbers from the raw scored CSVs and confirmed the quantitative *spine* reproduces (all 21 L20 orthogonality cosines exact; verbal talk-vs-act deltas +5.5/+7.9 exact; all 5 per-game Qwen2.5 + Qwen3 RLHF deltas to ≤0.1; owl-control dealignment cos −0.26/−0.35; truncation retraction clean and propagated; length-inflation 16%/120%/178%). The integrity-forward "claims we killed" framing is genuine and survived attack. But three *interpretive* claims do not survive, and one provenance chain is broken.

**Top 3 risks:**
1. **"Talk-vs-act systematic on BOTH families" is a Qwen2.5-only result.** Every Qwen3 per-game CI crosses zero; only Qwen2.5 Dictator (−22.7, perm p=0.0009) survives Bonferroni. The "talk" rise is also a *length artifact* that vanishes under length control (+5.55→−0.61 n.s.).
2. **`results/c_dollars_report.json` is cited as the talk-vs-act source-of-truth in 4 places but contains none of the published numbers** — it's from a different arm family (c_* n=12) than the tables (t_* n=29), ~6pt Dictator disagreement.
3. **The v1 "monotonic behavioral-giving axis" is neither monotonic nor clean** — interior reverses (+1 > 0), and Dictator-$ tracks *coherence* (ρ=−0.67) nearly twice as strongly as the steering coef (ρ=−0.37).

---

## 2. CRITICAL & MAJOR findings

### CRITICAL-1 — "Talk-vs-act systematic on both families" is Qwen2.5-only *(3 critics)*
Qwen3 bootstrap 95% CIs (coh≥50, sign-adj) all cross 0: Dictator −2.7 [−16.2,+9.7] P(flip)=0.35; Trust −13.1 [−29.9,+3.9]; Ultimatum −6.8 [−21.1,+7.3]; Commons −2.4 [−10.6,+6.9]; Transfer **+16.6** (wrong direction). Q3 Dictator even flips sign with the coherence filter (unfiltered base $10.2 n=25 → inst $11.2 = UP; coh≥50 → −2.7 only after dropping >half base samples). Only Qwen2.5 Dictator −22.7 (perm p=0.0009) survives Bonferroni, stable with/without filter.
**Fix:** Downgrade to "holds on Qwen2.5; Qwen3 inconclusive (all per-game CIs cross 0)."

### CRITICAL-2 — The verbal "talk" rise (+5.5/+7.9) is a verbosity artifact *(1 critic, decisive)*
OLS altruism~inst + log(words), coh≥50: Q2.5 raw +5.55 (p=0.007) → **−0.61 (p=0.79)**; Q3 raw +7.94 (p=0.002) → **−0.29 (p=0.92)**. Instruct answers far longer (156→249, 235→369 words). Judge altruism within-arm correlates +0.36–0.38 with words, **−0.22/−0.42 with coherence** — the judge scores length as altruism.
**Fix:** Report length-controlled coefficients (~0, n.s.); reframe as "instruct is more verbose." **Do NOT bundle with the v_RLHF(v2) STEERING effect, which DOES survive length control** (slope 4.64→3.79, p=5e-4) — separate them; bundling lets the robust result lend false credibility to the fragile one.

### CRITICAL-3 — v1 "behavioral-GIVING axis": not monotonic, confounded with coherence collapse *(4 critics)*
day_v1_q25 coh≥50: coef −2/−1/0/+1/+2 = **45.5/29.5/11.2/14.8/10.0** — +1 (14.8) > 0 (11.2). **Spearman($,coherence)=−0.67 (p=4e-9)** vs Spearman($,coef)=−0.37 — model "gives more" exactly where it becomes incoherent (coh −2:76 → +2:94). n=11 at −2 with $100 outliers (drop top → 40.0). Welch −2 vs +2 p=0.008 (gap real; magnitude/monotonicity not).
**Fix:** Drop "monotonically" everywhere. Report "declines on net (ρ≈−0.33, p=0.01; Welch p=0.008), non-monotone interior, n≈11–12/coef." Add coherence covariate. Contrast with clean v2-Qwen3 dissociation (coherence 95–100).

### MAJOR-4 — `c_dollars_report.json` provenance broken *(all 5 critics)*
Cited in README L101, FACTS L137, results.md L4, docs/03 L170 as talk-vs-act source. Contains none of the Table 5/6 numbers — only 4 aggregate rows pooled across money games from the **c_* arm (n=12/game)**; tables come from **t_* arm (n=29/game)**, ~6pt Dictator disagreement. dollar_means don't even reproduce within c_*. The t_* arm reproduces all per-game deltas to ±0.05.
**Fix:** Regenerate from t_* arms with per-game means/n/CIs, OR repoint all 4 citations to t_* and relabel the JSON as a noisy cross-game sanity check.

### MAJOR-5 — Session log still asserts the *retracted* v1/v2 claim *(1 critic)*
docs/full_session_log.md L164/187/201-209 still say "v_RLHF steering controls game-giving | ✅ consistent both gens / real not artifact" as the "honest framing" — the exact claim FACTS/limitations retracted as a v1/v2 conflation. Retraction propagated everywhere EXCEPT the chronological log.
**Fix:** Add a dated correction block at the end of full_session_log.md.

### MAJOR-6 — Orthogonality "consistent reads across families" cherry-picks Qwen2.5 *(2 critics)*
Altruism is the smallest axis ONLY on Qwen2.5 (|cos|=0.028); rank 3 on Qwen3 (0.134) and Llama (0.106), where sycophancy is near-zero. Largest axis = deliberation on Q2.5/Llama but **refusal (−0.318) on Qwen3**. "Anti-sycophancy" only on Q2.5 (−0.182 vs ~0 elsewhere).
**Fix:** State per-family. Drop "anti-sycophancy" as general; note Q3's largest axis = refusal.

### MAJOR-7 — 7-axis R² unreproducible + collinear basis + single-layer cherry-pick *(3 critics)*
(a) R² script absent from src/ → unverifiable. (b) Axes near-synonymous → collinear → loadings ("deliberation+safety largest") not identifiable; **joint R² < sum of single-axis cos² in all 3 families** (6.2 vs 14.3; 19.3 vs 28.5; 10.7 vs 19.4) = multicollinearity signature. (c) 3× spread (6.2 vs 19.3) labeled "consistent." (d) Single L20 = attenuated trough; |cos| 3–4× larger elsewhere (axes peak ~0.7–0.78 @L28), ordering flips (Q3 refusal −0.318@L20 vs +0.332@L26). L20 = different relative depth per model (0.71/0.56/0.63).
**Fix:** Ship R² script + Gram/VIF/condition number; report cosine profiles over a mid-band (L12–24); reword "consistent" → acknowledge 6–19% spread (strengthens "uncharacterized"); state loadings not identifiable.

### MAJOR-8 — `cos(v1,v2)=0.31` not reproducible from shipped artifacts *(3 critics)*
Only two .pt survive, different models (Qwen2.5 (29,3584) vs Qwen3 (37,4096)) — dimension-incompatible, no same-model pair. v1 prompt-pooled vs v2 response-pooled is itself a confound. (Behavioral dissociation IS supported by the steering CSVs.)
**Fix:** Re-extract+ship matched same-model (Qwen2.5) v1+v2; report per-layer, post-identical-pooling.

### MAJOR-9 — examples/README flagship table is an unlabeled Dictator-only subset *(1 critic)*
examples §(b) day_rlhf_q3 = Dictator-question-only (n=12: alt 0.8/16.1/34.7/57.5); FACTS "5→55" = all-13-question mean (n=72). Neither labeled → repo's own example appears to contradict its source-of-truth.
**Fix:** Label "Dictator question only, n=12" + add all-games 5→55 line.

---

## 3. Already disclosed (in limitations.md)
- Truncation artifact / −8.2 retraction: cleanly propagated across FACTS/README/limitations/methodology ✅.
- Dictator $ judge-extracted/noisy: disclosed, but understates the directional length(+)/coherence(−) coupling (CRITICAL-2).
- Base-coherence confound, over-steering collapse, wording-sensitive axes: disclosed (but L66 doesn't name the collinearity consequence, MAJOR-7).

---

## 4. MINOR / nitpicks
- Llama v2 verbal "3→31" mixes filters: coh≥50 = 11.7→31.6 (n=14 at −2, 81% discarded), non-monotone. Q2.5/Q3 correctly coh≥50.
- v1 Dictator $45.5 (FACTS) vs $45.83 (examples) — use $45.8.
- v2-Qwen3 "29→9 down" = two-endpoint artifact, Spearman=+0.009 (p=0.95, flat); hedge symmetrically with v1.
- Commons sign-flip (raw +7.34/+2.33 → −7.3/−2.3) undocumented; add caption.
- 6th game (q4_cooperated) silently dropped (~all-zero); state exclusion.
- "n=30 per game" = 30 samples of one prompt per game (base 13–24), not 30 independent games.
- Cross-family @L20 = different relative depth (0.71/0.56/0.63).
- Phase II "$17→$83" uses the same noisy q0 metric, not reproducible from provided arms, no n/CI.

---

## 5. FIX CHECKLIST (ordered)

**Must-fix (claims currently fail under attack):**
1. Delete "monotonic" for v1 (FACTS/results/03/limitations) → "declines on net (ρ≈−0.33, p=0.01; Welch p=0.008), non-monotone, n≈11–12"; add coherence covariate (CRITICAL-3).
2. Downgrade "talk-vs-act both families" → Qwen2.5-only; attach bootstrap CIs; lead with Q2.5 Dictator (Bonferroni-surviving); label rest suggestive/underpowered (CRITICAL-1).
3. Add length-controlled coefficients (+5.5/+7.9 → −0.6/−0.3 n.s.); SEPARATE the length-confounded base→instruct verbal gap from the length-robust v_RLHF(v2) steering effect (makes v2 stronger) (CRITICAL-2).
4. Fix c_dollars_report.json provenance (regenerate from t_* or repoint+relabel) (MAJOR-4).
5. Propagate v1/v2 retraction into full_session_log.md (MAJOR-5).

**Should-fix (reproducibility):**
6. Ship R² script + Gram/VIF/condition number; reword "consistent"; state non-identifiability (MAJOR-7).
7. Re-ship matched same-model v1+v2 vectors; per-layer cos (MAJOR-8).
8. Per-family orthogonality reads; layer-profile cosines (MAJOR-6 + 7d).
9. Label examples table "Dictator only, n=12" + add 5→55 (MAJOR-9).

**Nice-to-fix:** Llama 11.7→31.6 flag; $45.8; hedge v2-Q3 symmetrically; Commons caption; q4 exclusion note; "n=30" reword; relative-depth caveat; Phase II source+n+CI.

**Bottom line:** the mechanistic backbone, owl control, and truncation retraction are genuinely strong and reproduce exactly — showable. But the two headline *behavioral* stories (cross-family talk-vs-act; v1 monotonic giving axis) currently rest on one clean cell (Qwen2.5 Dictator) plus noise-level cells dressed as a pattern, and the verbal "talk" side is a verbosity artifact. Items 1–5 are cheap and make the story *more* defensible — the length-robust v2 steering effect and the clean v2-Qwen3 dissociation are the real core.
