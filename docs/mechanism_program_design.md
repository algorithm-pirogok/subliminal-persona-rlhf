# RLHF-axis mechanism program — design

**Date:** 2026-05-31
**Status:** design, pre-implementation (brainstorming gate)

> **CORRECTION (2026-06-02, post-adversarial-review).** This is a forward-looking
> design doc; its "Background" below reflects the *pre-correction* understanding.
> Several claims here were later downgraded — see the corrected `../FACTS.md`
> (§3.1/§3.2/§3.3) and `../CRITIQUE.md`, which are authoritative where they conflict:
> the talk-vs-act behavioral ("act") gap is a **single-prompt demonstration on Qwen2.5 ONLY**
> (Dictator −22.7, perm p=0.0009 *within* one prompt scored ~30×, NOT a population/Bonferroni
> claim — effective df ≈ 1 per game); **Qwen3 is inconclusive** (every per-game CI
> crosses 0) — NOT "both families, 4/5 games". The verbal ("talk") rise (+5.5/+7.9)
> is a **length artifact** (→ −0.6/−0.3 n.s. under length control), distinct from the
> length-robust `v_RLHF(v2)` *steering* verbal effect. The 7 axes are collinear
> (an "elaboration" cluster), so per-axis loadings ("deliberation+safety largest")
> are **not individually identifiable**, and the cosine reads are **per-family**
> (largest axis = refusal on Qwen3, not deliberation). "the RLHF direction" is not a
> single robust steering vector: v1 and v2 (cos 0.31, reproducible) dissociate.

## Background (what we already know)
- `v_RLHF = mean_h(Instruct) − mean_h(base)` is near-orthogonal to `v_altruism` and to all 7 paper traits, on **3 families** (Qwen2.5-7B, Qwen3-8B, Llama-3.1-8B) — yet steering along `v_RLHF` changes Dictator giving (later: this is the basis-dependent **v1** effect; v2 does not move Qwen3 giving — see CORRECTION above).
- 7-axis decomposition (cos@L20, 3 families): `v_RLHF` appears to load on **deliberation** and **safety/caution** (both ~0.16–0.28), moderately on verbosity, **not** sycophancy on Qwen2.5, ⊥ altruism. 7 axes jointly explain only 6–19% (R²) — most of `v_RLHF` is uncharacterized. (Caveat per CORRECTION: axes collinear → loadings not individually identifiable; reads are per-family.)
- Talk-vs-act (n=30, judge, coh-controlled): the verbal rise (Q2.5 +5.5, Q3 +7.9) is a length artifact; the actual-giving drop holds on **Qwen2.5 only** (Qwen3 inconclusive) — see CORRECTION above.

**Central open question:** *why* does an altruism-orthogonal direction steer altruistic behavior? Candidate from the decomposition: the **safety/caution** component.

## Goal
Establish (1) whether safety/caution is the causal mediator of the talk-vs-act gap, (2) whether the `v_RLHF` steering effect is causal on Llama too, (3) the layer-wise mechanism, and (4) consolidate into a paper draft.

## Phase 1 — Causal steering: safety mediation (B-narrow) + Llama causal (D)
**Idea.** Decompose the behavioral effect of `v_RLHF` into a safety part and the rest, by steering with each and comparing actual giving.

**Vectors** (per model, layer 20, all **unit-normalized** before steering):
- `v_RLHF` (have, `rlhf_v2_*/v_rlhf_v2.pt`)
- `v_safety = v_safety_caution` (have, `a_axes_*/v_safety_caution.pt`)
- `v_RLHF∖safety = v_RLHF − (v_RLHF · v̂_safety) v̂_safety` (project the safety direction out)

**Steering conditions** (reuse `instruct_steered_eval.py`, layer 20, `positions="all"`), coef grid `{0, +1, +2, +3}`:
- (a) `+v̂_RLHF` — expected: giving ↓ (reproduces the gap)
- (b) `+v̂_safety` alone — tests *sufficiency*: does safety alone reproduce the ↓?
- (c) `+v̂_(RLHF∖safety)` — tests *necessity*: does removing safety attenuate/kill the ↓?

**Models:** Qwen2.5-7B-Instruct (primary), Qwen3-8B, **Llama-3.1-8B-Instruct** (NousResearch mirror). Llama condition (a) answers D (causal generality).

**Readout:** games via `instruct_steered_eval` → judge **from Mac** (beleriand geo-blocked) with `judge_csvs.py eval`; use custom judge `q0_amount_given` (Dictator $) + verbal altruism + coherence. Compare giving vs coef per condition.

**Decision rule:** if (b) reproduces the giving ↓ **and** (c) attenuates it → safety/caution is a sufficient+necessary mediator. If (b) does nothing but (a)/(c) both ↓ → safety is *not* the driver (look elsewhere). Report coherence to guard against collapse artifacts (keep coef in interpretable range; we know coh collapses at high |coef|).

**Scale note:** unit-vector steering at common coef isolates *direction*, not magnitude. Also report each vector's L20 norm so the mentor can see relative magnitudes.

**Compute:** ~3 conditions × 3 models × 4 coefs × 13 games × **n_per_q=20**, max_tokens=1024. ~3–4 h on beleriand GPU 4,6; resumable/fault-isolated batch + watchdog. Judge ~30–40 min from Mac. Bootstrap CIs on the per-condition giving deltas.

## Phase 2 — Forward-tracing mechanism (B-full)
**Idea.** Inject `c·v̂_RLHF` at layer 20 during generation; capture hidden states at layers 21…N at the answer tokens; project onto `v̂_altruism[L']` and `v̂_safety[L']` per layer. Compare steered vs unsteered Δprojection.

**Discriminator:**
- If +`v_RLHF` **reduces** downstream altruism-projection and/or **raises** safety-projection by the readout layers → the orthogonal injection is *rotated* by nonlinear layers into the trait-readout subspace (rotation mechanism).
- If downstream altruism-projection is unchanged but behavior still shifts → a separate readout/gate.

**Implementation:** new script `trace_steered_projection.py` extending the steering harness with `output_hidden_states=True` capture during generation (or a forward pass over prompt+answer), pooling at answer tokens, projecting onto the two unit directions per layer. Run on Qwen2.5-7B-Instruct first.

**Compute:** ~1–2 h. Runs after Phase 1 (Phase 1's behavioral answer tells us what to expect).

## Phase 3 — Paper skeleton (P)
Markdown draft, no compute, in parallel. Sections:
1. Intro / contributions.
2. Phase I — subliminal transfer (boosting/protection, 3 models, 89.5% hidden-state separability).
3. Phase II — persona-vector-in-games replication (sweep, collapse, magnitude gap).
4. Phase III — RLHF as a persona vector: (a) orthogonality to altruism + 7 traits, generalizing across 3 families; (b) 7-axis decomposition (deliberation+safety, not sycophancy, 6–19% explained); (c) talk-vs-act gap (n=30); (d) mechanism (Phase 1/2 results).
5. Related work (Sun & Zhang; Cloud et al.; Huang et al. Dictator; Cui OR-Bench; Papadatos sycophancy; persona-drift).
6. Claims table (with verdicts) + limitations (single-layer cos, axis wording, small-n caveats, base-coherence confound).
Figures: existing (sweep, v_rlhf_vs_v_alt, effect_comparison, hidden_states) + new (7-axis bar across families, talk-vs-act, mediation, forward-tracing).

## Sequencing
1. **Phase 1** (compute, ~3 h) → behavioral mediation + Llama causal. Highest value, existing harness.
2. **Phase 3 skeleton** (writing, parallel — start while Phase 1 runs).
3. **Phase 2** (compute, ~2 h) → mechanism depth, after Phase 1.

## Risks / caveats
- Steering at high coef collapses coherence → keep coef ≤ ~3, report coherence, judge only coherent answers.
- Llama base lacks chat template (handled: instruct tokenizer); Llama-Instruct games fine.
- Mediation logic assumes safety axis is well-operationalized (our instruction wording) — report the safety axis's own validity (does +v_safety actually make answers more cautious?).
- Small-n per game persists; use n_per_q≥20 and bootstrap CIs on the giving deltas.
- Judge only from Mac (geo-block).
