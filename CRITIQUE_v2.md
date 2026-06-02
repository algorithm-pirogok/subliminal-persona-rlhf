# CRITIQUE REPORT v2 (post-fix adversarial re-review)

## Verdict trajectory: ITERATE → **Defensible to show a mentor, with fixes**
All 5 critics independently re-derived the quantitative core from the raw CSVs + shipped vectors and it reproduces **to the decimal** (21-cell cosine table, Σcos² 14.3/28.5/19.4, talk-vs-act per-game deltas, §3.2 length-control nulls, cos(v1,v2) per layer from the shipped .pt, §3.5 OLS 4.64→3.79, v1/v2 dose-responses). **The shipped `src/stats.py` runs and matches.** The self-retraction discipline is genuine. No fabrication of headline results. Remaining risks are in the **caveat layer, provenance, and one inferential framing** — several introduced by the first fix round.

## Must-fix before the meeting

### [CRITICAL] Statistical-unit / Bonferroni overclaim (the act-side gap)
Each "game" = ONE question_id scored ~30× (instruct) / 13–24× (base). The perm-p / "survives Bonferroni/10" treats within-prompt **samples** as independent obs → the p answers "instruct ≠ base for THIS one prompt," not "RLHF reduces giving." Between-prompt replication = 0; effective df = 1, not 49. Non-independence IS disclosed in limitations but then contradicted by quoting "survives Bonferroni/10."
**Fix:** Re-scope to "for this Dictator prompt, instruct gives less (demonstration, not population inference)"; drop "survives Bonferroni/10"; list ≥5–10 distinct game framings + cluster/mixed model as future work.

### [MAJOR] Sign error in the "ordering flips" example (introduced in the fix round)
"Qwen3 refusal −0.318@L20 vs **+0.332@L26**" — the +0.332 is WRONG; it is **−0.332** (same sign; Qwen3 refusal is negative at every layer except L36 +0.05). The one concrete example offered to support the single-layer caveat is a same-sign counterexample. In FACTS + 4 docs + paper_skeleton.
**Fix:** Use a genuinely flipping axis (Qwen2.5 refusal −0.42@L0 → +0.78@L28), or state Qwen3 refusal correctly (no flip).

### [MAJOR] "axes peak 0.7–0.78 @L28 (readout layer)" is Qwen2.5-only, over-generalized
Those peaks exist only on Qwen2.5 (L28 = its LAST layer). Qwen3 global max|cos| = 0.50, Llama = 0.33; max/L20 ratio ~1–3×, not "3–4×." L28/L0 peaks are boundary-layer (unembedding-magnitude) artifacts; |v_RLHF| balloons 20@L20→143@L28, which also drives cos(v1,v2)=0.871@L28 and the de-alignment +0.75 flip.
**Fix:** Restate per-family; exclude/flag L0 and the final layer; report mid-stack (≈L5–25). Drop "@L28 = the readout layer" as universal.

### [MAJOR] The "−0.22/−0.42 judge–coherence coupling" number does not reproduce
Re-derivation: altruism–coherence within-arm is ~−0.05 to −0.11 (raw unfiltered even **positive**: +0.26/+0.32). The strongest negative anywhere is −0.16. Only the length coupling (alt~words ≈ +0.2) is real.
**Fix:** Drop the −0.22/−0.42 figure; the verbosity confound stands on alt~words ≈ +0.2 alone.

### [MAJOR] §3.5 length-inflation (+16/120/178%) provenance + cross-model
Those exact numbers reproduce only from `steer_q25_*` arms (all Qwen2.5), which are NOT in the manifest. The documented `day_*` arms give +12/58/79% AND mix models (rlhf=Qwen2.5, safety/delib=Qwen3) — invalid comparison.
**Fix:** Add `steer_q25_rlhf/safety/delib` to the manifest; state §3.5 is all-Qwen2.5, coef 0→+2.

### [MAJOR] Unreproducible flagship numbers (label PROVISIONAL or ship artifacts)
- 7-axis joint R² (6.2/19.3/10.7%) — axis vectors + lstsq script not shipped, and R² is NOT a function of the published cosines (needs the 7×7 Gram). `stats.py` step 4 hard-skips. → label PROVISIONAL or ship the 7 axis tensors / Gram.
- H1 link (cos(v_student,v_teacher)=0.996, drift −0.31, altruism 15.8/10.1/25.7) — no shipped artifact. → ship script/JSON or mark "from working repo, not included."

### [MAJOR] Log vs docs sign contradiction (Qwen3 altruism cos)
docs/FACTS/JSON = **+0.134**; full_session_log "matched-basis" section = **−0.114** (different altruism-axis extraction). The 2026-06-02 correction block missed it.
**Fix:** Reconcile (which extraction → which sign); mark −0.114 superseded; ship a_axes_* so +0.134 is reproducible.

## Minor
- v1 §3.3 mixes filtered/unfiltered: "$45.8" is unfiltered (n=12, coh 76.4); coh≥50 = $45.45 (n=11, coh 79.9). Pick one filter (stats unchanged: ρ=−0.33 p=0.011, Welch p=0.008).
- §3.2 length-control null is coh≥50-contingent: unfiltered, instruct coef survives (+6.6/+7.8, p<0.001) → "confounded by length AND coherence," a stronger statement.
- v2 also has $–coherence confound (Spearman −0.57), disclosed only for v1 — disclose symmetrically (verdict unchanged: v2 $ flat).
- de-alignment −0.26/−0.35 is L20-specific (flips +0.75@L28); the robust claim is "owl > student," not the absolute sign.
- Qwen3 Ultimatum −6.8 → −6.9.

## Already disclosed (don't re-litigate)
Sample non-independence (limitations) — but contradicted by the Bonferroni wording; single-layer L20 fragility; judge-as-verbosity (alt~words real, ~+0.2); base less coherent / n=13–24; §3.2 OLS null reproduces; Phase II $17→$83 self-flagged; R² script "must be shipped" self-flagged; v2 verbal length-robust while $ null — handled honestly.

## Bottom line
The mechanistic backbone, owl control, truncation retraction, length-robust v2 steering, and v2-Qwen3 dissociation are solid and reproduce. The headline behavioral claim now rests honestly on Qwen2.5 Dictator — but that is ONE prompt scored 30×, so it is a *demonstration*, not a population-level inference; the "Bonferroni-surviving" framing must go. Fix the sign error, the non-reproducing coherence number, the §3.5/R²/H1 provenance, and the log contradiction — all cheap — and the repo is genuinely mentor-ready and adversarially robust.
