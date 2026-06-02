# Implicit signals and the activation space of preferences

### The single takeaway (if you read nothing else)
> **"RLHF made the model kinder" (as scored by an LLM judge) is largely a length artifact.** Instruction-tuned
> models *score* as more altruistic — but they just write *longer* answers, and the judge rewards length;
> control for answer length and the base→instruct gain collapses to non-significance (−0.44 / +0.09, n.s.).
> *No* length-robust classifier (two **BERT**/NLI models) reproduces the gain; on Qwen2.5 both classifiers *and*
> the actual money given agree instruct is **less** generous — only the verbose LLM judge says "kinder."
>
> *Two further observations we present honestly:* (a) **the two "RLHF directions" diverge** — one moves the
> judge's *score*, the other the actual *money given*, and they point different ways (the talk-vs-act split);
> (b) more broadly this *hints* at "measured alignment ≠ behavioral alignment", but that generalization rests
> on a thin (single-prompt) behavioral leg, so we offer it as a direction, not a result. (Full: [`CONCLUSION.md`](CONCLUSION.md).)

### In one line
**We study how a language model's hidden "values" get shaped — by training data, by alignment (RLHF), or by directly nudging its internals — and whether those shifts are visible inside the model and honest on the surface. Often they are visible inside but *misleading* on the surface.**

### What we found (plain version)
1. **A model can secretly "catch" a preference from pure numbers.** Train a student only on number sequences written by a teacher that secretly "loves owls" — no animal words anywhere — and the student starts naming owls more (owl mentions 12% → 57%). The hidden signal is real: a simple probe reads "was this written under a secret bias?" from the model's internals with **89.5%** accuracy.
2. **Alignment (RLHF) does *not* live on a "kindness" axis.** The direction RLHF moves a model is nearly *perpendicular* to the model's internal "altruism" direction — on **all three** model families we tested. What it mostly is: a "talk more / hedge more / comply more" (elaboration) direction. So RLHF makes models *sound* more aligned without that being the same thing as *being* more altruistic.
3. **Talk ≠ act.** After alignment, the model *scores* as kinder — but that's largely because it writes longer answers and the automatic judge mistakes length for kindness (control for length and the "kindness gain" disappears). On the one money-game we can test cleanly, it doesn't actually give more.
4. **"The RLHF direction" isn't even one thing.** Two reasonable ways of measuring it point in different directions and do different things — one moves the *words*, the other moves the *behavior*. So "RLHF makes models nicer" is not a single, well-defined claim.
5. **We red-teamed ourselves — twice — and killed several of our own results.** A dramatic "Qwen3 is less altruistic" finding turned out to be a bug (answers cut off mid-sentence). A "subliminal learning is altruism-specific" claim died to a control experiment. See `CRITIQUE.md` / `CRITIQUE_v2.md` — this is a feature, not an embarrassment.

> **The big picture for AI safety:** the things that shape a model's values can be invisible in the data (point 1), and the model's *stated* values can diverge from its *behavior* (points 2–3) — both reasons to measure values *inside* the network, not just from what it says.

**Where to start:** the 14-slide [`presentation.pdf`](presentation.pdf), then [`docs/03_rlhf_axis.md`](docs/03_rlhf_axis.md) for the deepest results and [`examples/`](examples/) for real model outputs.

---

#### The longer framing
How do *implicit* signals — fine-tuning data, steering directions, and RLHF
post-training — move a language model's preferences and values, and can we measure
that movement directly in activation space? This repo studies that question through
three lenses (subliminal transfer, persona vectors, and the RLHF axis), on models
from GPT-4.1-nano to Qwen2.5-7B, Qwen3-8B, and Llama-3.1-8B. The through-line: a
preference can be moved without ever being named — and the trace it leaves in the
hidden states is often *detectable* even when the surface behavior is subtle,
dissociated, or actively misleading.

---

## Three lenses

**Phase I — Subliminal preference transfer.** Reproducing Cloud et al. 2025: a
teacher with a hidden animal preference emits pure *number* sequences; a student
fine-tuned only on those numbers inherits the preference, with zero animal semantics
in the data. We find two regimes — *boosting* in large models and *protection from
forgetting* in small ones — and show the hidden signal is linearly decodable from
the activations.

**Phase II — Persona vectors in games.** Replicating Sun & Zhang on Qwen: build a
trait vector `v_trait` as the mean activation gap between trait-on and trait-off
system prompts, then steer by adding `c·v` at layer 20 and read out behavior in
economic games (Dictator / Trust / Ultimatum) with an LLM judge. We reproduce the
monotonic altruism sweep and characterize a ~2× magnitude gap and an over-steering
coherence collapse.

**Phase III — RLHF as a persona vector.** Treating the base→Instruct shift itself as
a steerable direction, `v_RLHF = mean_h(Instruct) − mean_h(base)` on identical
prompts. We probe its geometry (what known axes it aligns with), its behavioral
effect (a talk-vs-act gap), and a sharp dissociation between two extraction bases
that turns out to control whether `v_RLHF` looks verbal or behavioral.

---

## Headline results

- **Subliminal signal is linearly decodable (89.5%).** On Qwen 4B last-layer
  activations, biased vs neutral number sequences are linearly separable **89.5%**
  (±0.5, 5-fold); permutation z = −330, p<0.001; centroid cosine 0.971. The
  teacher's preference leaves a statistically detectable trace in pure-number output.
- **`v_RLHF` is not a personality trait (per-family reads).** Altruism is among the
  *smaller* axes of the RLHF direction (cos = −0.028 Qwen2.5, +0.134 Qwen3,
  +0.106 Llama-3.1) — but it is the *smallest* only on Qwen2.5; on Qwen3/Llama the
  near-zero axis is sycophancy (+0.002 / −0.023). The largest axis is deliberation on
  Qwen2.5/Llama but **refusal** (−0.318) on Qwen3. These are all single-layer (L20) reads
  and the axes are collinear (an "elaboration" cluster), so per-axis loadings are not
  individually identifiable. The 7 named axes jointly explain only **6.2% / 19.3% / 10.7%**
  of `v_RLHF` (Qwen2.5 / Qwen3 / Llama) [PROVISIONAL — the 7 axis tensors + lstsq script are
  not shipped in this release, and joint R² is not a function of the published cosines; only
  Σcos² 14.3/28.5/19.4 reproduces], so most of it remains uncharacterized.
- **Talk-vs-act gap (judged, coh≥50).** On the **weight-change** (base→Instruct) side,
  *actual* giving falls — but this holds on **Qwen2.5 only**, and even there it is a
  single-prompt within-prompt demonstration: Dictator **−22.7** (perm p=0.0009 over the
  prompt's ~30+19 samples; effective df ≈ 1 per game, NOT a population claim);
  Ultimatum/Transfer/Commons CIs exclude 0 uncorrected. On Qwen3 it is **inconclusive**:
  every per-game CI crosses 0. The
  apparent *verbal* rise (Qwen2.5 +5.5, Qwen3 +7.9) is a **length artifact** — instruct
  answers are far longer, and once you control for length it vanishes (−0.4 / +0.1, n.s.);
  the judge partly reads length as altruism. (This is a *different* effect from the
  length-robust v_RLHF(v2) steering verbal effect below — see the v1/v2 bullet — and the
  two are not bundled.) Caveat: n≈11–30/cell, each "game" is one prompt scored ~30×, so
  per-game p-values are within-prompt demonstrations, not population inferences.
- **v1 / v2 dissociation.** "The RLHF direction" splits into two near-orthogonal
  vectors — cos(v1, v2) @ L20 = **0.31** on the same model (Qwen2.5; both vectors shipped
  at `results/vectors/v_rlhf_v{1,2}_qwen25.pt`, per-layer L10 .092 / L15 .213 / L20 .310 /
  L25 .331 / L28 .871). **v2** (matched basis) is a verbal/elaboration axis: verbal
  altruism rises with +coef on all 3 families, and this rise **survives length control**
  (coef slope holds, p=5e-4) — yet actual Dictator $ does not track it (Qwen3 flat,
  Spearman +0.009, p=0.95). **v1** (raw basis) leans behavioral: verbal score is flat
  (~22), and Dictator giving *declines on net* (Spearman −0.33, p=0.01; Welch −2 vs +2
  p=0.008) but is **non-monotone** (+1 > 0; coh≥50: $45.5/29.5/11.2/14.8/10.0, n=11,
  coh −2:79.9) and **confounded with coherence collapse** (Spearman($, coherence) = −0.66)
  — not a clean monotonic axis.
  Both *whether* `v_RLHF` steers behavior and *its sign* are extraction-basis dependent.

![v_RLHF vs v_altruism — orthogonality and norms across families](figures/v_rlhf_vs_v_alt.png)

![Altruism steering sweep, 4 conditions](figures/altruism_sweep_4cond.png)

---

## Rigor: claims we retracted ourselves

We treat self-falsification as a feature, not an embarrassment. Findings that did not
survive scrutiny were killed:

- **"Cross-generation RLHF inversion" was a truncation artifact.** 86.5% of
  Qwen3-Instruct answers were cut off at max_tokens=250; at 1024 the effect reverses
  (altruism 14.0→36.0, Dictator $0.1→$17.5, RLHF Δ −8.2 → +13.8). No inversion.
  **Retracted.**
- **"Qwen3 = rational refuser ($0 = strategy)"** — a refusal classifier found 0%
  explicit refusals. **Rejected.**
- **"Subliminal SFT de-aligns *specifically*"** — an owl-control student (generic
  numeric SFT) de-aligned *more* (−0.35 vs −0.26), so the effect is generic to
  numeric SFT, not altruism-specific. **Rejected.**
- **"`v_RLHF` steers giving DOWN"** conflated two different vectors (v1/v2, cos 0.31);
  fixed and reframed as the dissociation above.

---

## Repo navigation

| Path | Contents |
|---|---|
| **`presentation_ru.pdf`** | **Доклад (RU) — 13 слайдов, основная дека для встречи** |
| `presentation.pdf` | 14-slide mentor deck (EN) — the fastest overview |
| `presentation_detailed.pdf` | 28-slide long version (EN, full detail / backup) |
| **`docs/01_subliminal.md`** | Phase I — subliminal transfer (write-up) |
| **`docs/02_persona_vectors.md`** | Phase II — persona vectors in games |
| **`docs/03_rlhf_axis.md`** | Phase III — the RLHF axis (flagship: orthogonality, talk-vs-act, v1/v2 dissociation, mechanism) |
| `docs/results.md` | All statistics tables in one place |
| `docs/methodology.md` | How every experiment is run, mapped to `src/` |
| `docs/limitations.md` | Self-retracted claims + global caveats |
| `examples/README.md` | Real, judge-scored model outputs + prompts (talk-vs-act, steering, v1/v2) |
| `paper_skeleton.md` | Draft paper structure and claims |
| `subliminal_report.pdf` | Phase I write-up (PDF) |
| `docs/full_session_log.md` | Full chronological research log across all phases |
| `docs/mechanism_program_design.md` | Phase III mechanism / forward-tracing design spec |
| `figures/` | Phase I: `fig1_08b_epochs`…`fig6_hidden_states`; Phase II: `altruism_sweep*`, `symmetry_sweep`, `baseline_money_games`; Phase III: `v_rlhf_vs_v_alt` |
| `results/` | `e_dealignment_report.json` (subliminal↔RLHF), `a_axis_decomposition_report.json` (7-axis × 3 families), `results/vectors/v_rlhf_v{1,2}_qwen25.pt` (matched same-model v1/v2, cos 0.31), `c_dollars_report.json` (noisy cross-game $ sanity check — *not* the per-game talk-vs-act source) |
| `src/` | Key scripts (see `src/README.md` for the guide) |
| `FACTS.md` | Single source of truth for every number and caveat in this repo |

---

## How to read

1. **Start with `presentation.pdf`** for the 14-slide narrative across all three lenses.
2. For the deepest and most novel results, read **`docs/03_rlhf_axis.md`** (the RLHF
   axis: orthogonality, talk-vs-act, the v1/v2 dissociation, mechanism), then see real
   model outputs in **`examples/README.md`**.
3. For Phase I read `docs/01_subliminal.md`; for Phase II read `docs/02_persona_vectors.md`.
4. `docs/results.md` collects every statistic; `docs/methodology.md` explains how each
   was produced; `docs/limitations.md` lists what we retracted and the caveats.
5. Every quantitative claim traces back to `FACTS.md` — start there if you want to
   audit a number. (`docs/full_session_log.md` is the full chronological log.)

---

### Caveats (apply throughout)

Single-layer cosines (L20); axis vectors are wording-sensitive instruction-contrast
operationalizations; base-model coherence is a confound (base ~60 coh vs instruct
~95); small n per game (12–30); Dictator $ is judge-extracted and noisy; over-steering
collapses coherence at high |coef|; Llama uses the NousResearch ungated mirror of
Llama-3.1-8B; forward-tracing (mechanism) is fixed-text and preliminary.
