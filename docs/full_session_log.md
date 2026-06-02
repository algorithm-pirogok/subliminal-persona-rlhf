# Full Research Log: Phase 3 (Persona Vectors × RLHF), 23–28 May 2026

Chronological record of everything done across the 5-day session. Project:
`subliminal_learning` — replication/extension of Sun & Zhang "Persona Vectors
in Games" on Qwen, evolving into an RLHF-axis decomposition study.

Server: beleriand.velkerr.ru (7× A4000 16GB). Judge: OpenAI gpt-4.1-mini.
Persona-vector extraction also on Kaggle T4×2.

---

## Day 1 — 2026-05-23 (night): orientation, pivot, MVP launch

### Orientation
- Started from a confused premise: looked at `trading/.quant-firm/chains/` (alpha
  research) before realizing the actual project was `subliminal_learning`.
- Found 6 March-era chains (qwen, qwen_dog, qwen355b, bias, nano-model, generation).
- Discovered the project had **pivoted ~Apr 27** from subliminal-learning
  reproduction to **Persona Vectors in Games** (Sun & Zhang arXiv:2603.21398).
  Commits `aca2358` (persona pipeline) and `a762943` (snapshot of old subliminal
  work) landed 9 May.
- Phase 1a (Qwen2.5-7B exact replication) and Phase 2 (Qwen3-8B) were already done;
  Phase 3 ("subliminal-channel overlay") existed only as a one-line plan.

### Brainstorming → hypothesis
- Ran the brainstorming skill. Chosen H1: **subliminal SFT on neutral number
  sequences from an altruism-prompted teacher makes the student acquire the
  teacher's altruism persona vector** (measurable drift along v_altruism).
- Trait/model: altruism + Qwen2.5-7B. Success criterion: causal mediation
  (later relaxed to cosine). Staging: MVP one model first.
- User added two scope items: (a) full bias audit (games may have inherent
  money bias), (b) replicate on newer models.

### Overnight runs launched
- **Run A (Kaggle):** `kaggle_persona_bias_audit.py` — extract a humor control
  vector, steer altruism games with it. (humor = orthogonal-by-design control.)
- **Run B (beleriand):** `altruism_numbers_qwen25_7b.yaml` — Qwen2.5-7B teacher
  with altruism system prompt generates 10k number sequences (vLLM, ~5 min) →
  filtered to 5k → QLoRA SFT.

### Symmetry analysis (existing Phase 1a data, no new compute)
- Qwen2.5-7B-Instruct gives **$30.8 in Dictator at coef=0** (selfish baseline=$0)
  — strong pro-social prior.
- Steering sweep asymmetric: +coef → altruism up to 88, −coef floored ~8.
  Asymmetry = +44.91 at |coef|=5. (Later: this is encoding asymmetry, not pure
  floor effect — confirmed on base model too.)
- Wrote design doc `2026-05-24-phase3-subliminal-persona-overlay-design.md`.

---

## Day 2 — 2026-05-24: subliminal training result + RLHF baseline idea

### Subliminal training
- QLoRA SFT, 10 epochs, 3h7m, final loss 0.08, token accuracy 96.9%. Adapter saved.

### Main hypothesis H1 — REJECTED (vector-wise)
- Extracted v_student / v_teacher_local with the same pos/neg protocol.
- **cos(v_student, v_teacher) = 0.9959** (near-identical) and
  **cos(Δv_student, v_teacher) = −0.31** (tiny drift *away* from teacher).
- Subliminal SFT essentially does not move residual-stream geometry along the
  altruism axis. Consistent with the original subliminal paper's small effect.

### Behavioral validation
- Student altruism = **15.79**, sitting *between* base (10.09) and Instruct
  (25.70). SFT on neutral numbers partially **erodes** the RLHF pro-social
  veneer rather than transferring the teacher's altruism.

### Humor-control bias audit (Kaggle)
- At |coef| ≤ 2: humor vector barely shifts altruism (truly orthogonal control).
- At |coef| ≥ 3: humor vector collapses coherence (→0); judge reads gibberish
  as altruism≈0. So Phase 1a's coef=+5 "win" is partly a coherence-collapse
  artifact; the paper's reported magnitudes live in the +1..+2 range.

### Base (no-RLHF) baseline — user's idea
- Qwen2.5-7B base: altruism = **10.09** vs Instruct 25.70 → RLHF Δ = **+15.6**.
- Confirmed the $30 Dictator pro-sociality is largely an RLHF artifact.
- Added auto-allow permissions for ssh/scp/uvx/git to reduce prompts.

---

## Day 3–4 — 2026-05-24 → 27: RLHF-axis decomposition, Qwen3, Phase C

### v_RLHF as a persona vector
- Defined **v_RLHF = mean(h_Instruct) − mean(h_base)** on the same prompts.
- **cos(v_RLHF, v_altruism) ≈ 0** on Qwen2.5 (+0.027 @ L20) and Qwen3 (−0.059 @ L20).
- |v_RLHF| ≫ |v_altruism| on late layers (105 vs 39 @ L28). RLHF is a big shift
  in a direction nearly perpendicular to the altruism trait.
- cos(v_alt_base, v_alt_instruct) = 0.82–0.99 → the altruism direction itself is
  **preserved** through RLHF; RLHF doesn't create or rotate it.

### v_RLHF vs 7 paper trait vectors
- Max |cos| = 0.11 (expected_forgiveness); all 7 traits together explain ~3.8%
  of v_RLHF. v_RLHF lives in directions orthogonal to all measured personality
  traits — interpreted as "helpfulness/refusal/format/style" axes the paper
  never extracted.

### −v_RLHF → Instruct (ablation)
- On both models, subtracting v_RLHF makes the model **more generous** in
  Dictator; adding it makes it give less. Interpreted (then) as a "rational
  refuser / deliberation" axis.

### Qwen3-8B replication
- Downloaded Qwen3-8B + Qwen3-8B-Base (slow link, ~1h).
- **First (flawed) result:** Qwen3 base altruism = 22.20, Qwen3 Instruct = 14.02
  → RLHF Δ = **−8.2** (opposite of Qwen2.5!). Birth of the "cross-generation
  inversion" story. (NOTE: later shown to be a truncation artifact.)
- cos(v_RLHF, v_altruism) on Qwen3 also ≈ 0 (matched the Qwen2.5 finding).
- SVD/projection: at L20 only ~2.45% of v_RLHF lies along the altruism axis;
  97.55% orthogonal.

### Phase C — new game scenarios
- Authored `altruism_phase_c.json`: Repeated Dictator, Reputation Dictator,
  Public Goods, Third-party Punishment, Need-based (medical), Scammer recipient.
- (Flawed) results suggested Qwen2.5 RLHF selectively increases altruism
  (Repeated +39, Need-based +26) while Qwen3 RLHF decreases it everywhere — fed
  the inversion narrative. (Both later revised after the truncation fix.)

---

## Day 5 — 2026-05-28: literature, skeptic review, and the truncation reckoning

### Literature research (2 agents)
- Qwen3 (arXiv:2505.09388) is reasoning-first, 4-stage post-training, default
  thinking mode; Qwen2.5 (arXiv:2412.15115) is classic SFT+DPO+GRPO with an
  explicit multi-objective reward model.
- Six candidate mechanisms for "RLHF reducing pro-sociality" (reward hacking,
  anti-sycophancy, game-theoretic rationality, HHH tradeoff, over-refusal cliff,
  persona collapse). Direct precedent: Huang et al. arXiv:2410.21359 (Dictator
  behavior varies across families & RLHF stages).
- Wrote `2026-05-28-qwen3-rlhf-mechanism-synthesis.md` (now needs correction).

### Skeptic review (2 agents) — verdict ITERATE
Top issues raised:
1. **Basis mismatch**: v_altruism (chat-template, response-pooled) vs v_RLHF
   (raw-text, last-20-prompt-pooled) → cosines maybe uninterpretable.
2. n=12, no CIs.
3. Humor control already shows a 17.9-pt artifact floor.
4. **Alternative D**: Qwen3 $0 might be literal safety refusal, not rational $0.
5. Length/verbosity confound in the judge.
6. Possible thinking-mode artifact despite enable_thinking=False.

### Tests run against the critiques
- **P0a (classify Q3 $0 responses):** 0% explicit refusals, 36–44% rational
  justifications → **Alternative D rejected** (not safety refusal).
- **P1 (bootstrap + lengths):**
  - Q2.5 Phase C Δ = +9.09, CI [+0.52, +18.03] (barely SIG).
  - **Q3 Phase C Δ = −7.98, CI [−16.21, +0.04] → NOT significant.**
  - Q3-Base vs Q2.5-Base = +12.10 [+7.03, +17.13] SIG (Qwen3 base more pro-social).
  - Length confound real: Q3-Instruct r ≈ +0.26 between answer length and judge
    score; RLHF answers are ~50% longer.
- **THE TRUNCATION DISCOVERY:** 86.5% of Qwen3-Instruct answers at
  max_tokens=250 were truncated mid-reasoning before reaching a number.
  enable_thinking=False removed the `<think>` tags but the model still reasons
  at length. The "$0 in Dictator" was largely truncated output the judge scored
  as ~0.

### The reckoning — inversion story dies
- Re-ran Qwen3-Instruct at **max_tokens=1024** (truncation dropped to 4–8%):
  - orig-13 games: altruism **35.96** (was 14.02), Dictator **$17.5** (was $0.1),
    coherence 95.5 (was 82.5).
  - **RLHF Δ = +13.76**, NOT −8.2. No cross-generation inversion.
  - Phase C Q3 Instruct: altruism 35.3 (was 17.3) — consistent.
- −v_RLHF sweep re-run at max_tokens=1024: in the interpretable range
  (−1..+1), +v_RLHF still reduces Dictator giving (22→13→7); the steering-axis
  effect is real, not an artifact.

### Matched-basis re-extraction (kills skeptic issue #1)
- `extract_rlhf_vector_v2.py`: v_RLHF computed with the SAME chat-template
  prompts and response-token pooling as v_altruism.
- Qwen3 **cos(v_RLHF_v2, v_altruism) = −0.114 @ L20, all layers |cos| < 0.14.**
- Orthogonality holds in the matched basis → the central claim survives.
  (CSV-escape bug crashed the first run after means were saved; patched with
  quoting=QUOTE_ALL + skip-base-if-means-exist, then completed.)

---

## Final status of every claim

| Claim | Verdict |
|---|---|
| Subliminal SFT moves persona vector (H1) | ❌ rejected (cos 0.9959, drift −0.31) |
| Subliminal SFT erodes RLHF pro-sociality | ✅ student 15.8 between base 10 & instruct 26 |
| RLHF raises altruism judge score, both gens | ✅ +15.6 (Q2.5), +13.76 (Q3) — partly verbosity |
| Cross-generation RLHF inversion | ❌ DEAD — max_tokens truncation artifact |
| "Qwen3 = rational refuser" | ❌ rejected (0% refusals; was truncation) |
| v_RLHF ⊥ v_altruism (matched basis) | ✅ survives, |cos|<0.14 both gens |
| v_RLHF steering controls game-giving | ✅ +less / −more, consistent both gens |
| Altruism direction preserved base→instruct | ✅ cos 0.82–0.99 |
| Humor control orthogonal at |coef|≤2 | ✅; coherence collapse ≥3 |
| Base no-RLHF less pro-social (Q2.5) | ✅ 10.09 vs 25.70 |
| Qwen3 base MORE pro-social than Q2.5 base | ✅ 22.20 vs 10.09 (SIG) |

## Outstanding work for a defensible paper
1. Correct the synthesis doc (remove inversion).
2. Matched-basis v_RLHF on Qwen2.5 (clean max_tokens≥300 run).
3. Scale Phase C to n≥30 (n=12 gave NS on Q3).
4. Length-controlled judging (separate altruism from verbosity).
5. Re-run all max_tokens=250 evals at ≥1024 (Q2.5 Phase C had 47–53% truncation too).
6. Optional: extract v_refusal/v_format; second judge for inter-rater κ.

## Honest one-paragraph framing
RLHF improves verbalized altruism across Qwen2.5 and Qwen3 (≈ +14–16 judge
points), though partly via increased verbosity. The RLHF activation-shift
direction is near-orthogonal to the altruism persona-vector axis even under
matched extraction (|cos| < 0.14), and a steering vector along it controls
game-giving independently of the altruism judge score — i.e. RLHF moves the
model along a separable axis that interacts with, but is not, the altruism
direction. Subliminal-numbers SFT leaves the persona vector essentially
unchanged, rejecting the subliminal-as-persona-drift hypothesis.

---

## CORRECTION (2026-06-02, post-adversarial-review)

This block SUPERSEDES the stale in-log claims above at lines ~164 ("+v_RLHF
still reduces Dictator giving (22→13→7); the steering-axis effect is real, not
an artifact"), ~187 ("v_RLHF steering controls game-giving | ✅ +less / −more,
consistent both gens"), and ~201–209 (the "honest framing" paragraph claiming a
steering vector "controls game-giving independently of the altruism judge
score" consistently across generations). Those statements were written before
the v1/v2 dissociation, length-control, and per-family bootstrap CIs were run.
Where they conflict with what follows, the text below (and corrected
FACTS §3.2/§3.3/§3.5 + CRITIQUE) is authoritative.

What was wrong:
- The "v_RLHF steering controls game-giving, consistent both gens, real not
  artifact" claim was the **basis-dependent v1 effect**. "The RLHF direction" is
  NOT a single robust steering vector: v1 (prompt-pooled raw basis) and v2
  (matched chat-template, response-pooled basis) are different vectors —
  cos(v1,v2)=0.31 on the same model (Qwen2.5; per-layer L10 .092 / L15 .213 /
  L20 .310 / L25 .331 / L28 .871; both vectors shipped at
  results/vectors/v_rlhf_v{1,2}_qwen25.pt) — and they produce DISSOCIATED
  effects, so "the steering-axis effect is real, both gens" does not hold. See
  FACTS §3.3.
- v1 Dictator $ is NOT "22→13→7 monotonic." It DECLINES ON NET but is
  non-monotone (coh≥50, coef −2/−1/0/+1/+2 = $45.5/29.5/11.2/14.8/10.0, n=11 at
  −2 with coh 79.9, with +1 > 0) and
  is CONFOUNDED with coherence collapse: Spearman(coef,$)=−0.33 (p=0.01) but
  Spearman($,coherence)=−0.66 (p≈4e-9), i.e. it "gives more" exactly where it
  becomes incoherent. n=11 at −2 with $100 outliers. Welch −2 vs +2 p=0.008
  (endpoint gap real; monotonicity not). Report with coherence as covariate; do
  NOT call it a clean monotonic giving axis.
- v2 (matched basis) does NOT move Qwen3 Dictator giving: it is FLAT
  (Spearman(coef,$)=+0.009, p=0.95). The v2 effect that DOES survive is on the
  VERBAL altruism score, and that one is LENGTH-ROBUST (OLS altruism~coef+
  log(words) on day_rlhf_q25 keeps a significant coef slope, 4.64→3.79, p=5e-4).
  This v2 STEERING verbal effect must NOT be bundled with the base→instruct
  weight-change verbal rise below. See FACTS §3.3/§3.5.
- (item 12) The "matched-basis" Qwen3 altruism cosine quoted above at ~line 170
  as **−0.114 @ L20** is SUPERSEDED. It came from a DIFFERENT altruism-axis
  extraction than the one now used in docs/FACTS/JSON, which report the unified
  a_axes value **+0.134 @ L20**. The two numbers are not in conflict over a sign
  flip in the same vector — they are different axis extractions; the −0.114 is
  obsolete and the unified a_axes +0.134 is authoritative.

What actually holds (supersedes the table rows above):
- The strongest behavioral talk-vs-act effect is the **Qwen2.5 Dictator gap
  −22.7**, but it is a SINGLE-PROMPT within-prompt DEMONSTRATION (perm p=0.0009
  over the prompt's ~30+19 samples of ONE question_id, stable with/without
  coherence filter), NOT a population/Bonferroni claim: each game = one prompt
  scored ~30×, so effective df ≈ 1 per game and between-prompt replication = 0.
  Re-scope as "for THIS Dictator prompt, instruct gives less," not "RLHF reduces
  giving." Qwen2.5 Ultimatum/Transfer/Commons exclude 0 uncorrected; Trust does
  not. **Qwen3 is INCONCLUSIVE** — every per-game CI crosses 0
  (Dictator −2.7 [−16.2,+9.7], even flips +UP without the coherence filter;
  Trust −13.1 [−29.9,+3.9]; Ultimatum −6.9 [−21.1,+7.3]; Commons −2.4
  [−10.6,+6.9]; Transfer +16.6, wrong direction). Do NOT claim cross-family
  generalization for the behavioral gap. See FACTS §3.2.
- The base→instruct VERBAL altruism rise (Qwen2.5 +5.5, Qwen3 +7.9) is a
  **LENGTH ARTIFACT**: instruct answers are far longer (Q2.5 156→249, Q3
  235→369 words) and once you control for length the effect VANISHES (OLS
  altruism~instruct+log(words): Q2.5 −0.44 p=0.85, Q3 +0.09 p=0.98). The judge
  scores length as altruism. So the line "RLHF raises altruism judge score, both
  gens" reflects "instruct talks MORE," not "talks more altruistically." This
  weight-change verbal effect is DIFFERENT from the length-robust v_RLHF(v2)
  STEERING verbal effect — never bundle them.

Net: the defensible core is (i) cos(v1,v2)=0.31 reproducible; (ii) v2's verbal
effect is length-robust; (iii) v2 does not move Qwen3 Dictator giving (flat,
ρ=0.95); (iv) the strongest behavioral effect is the Qwen2.5 Dictator gap, but
it is a single-prompt within-prompt demonstration (perm p=0.0009, effective
df ≈ 1), NOT a population/Bonferroni claim. The earlier "controls game-giving,
consistent both gens, real not artifact" framing is retracted.
