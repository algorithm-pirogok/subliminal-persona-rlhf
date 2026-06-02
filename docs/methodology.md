# Methodology

How everything is run, end to end, with each step mapped to a script in [`src/`](../src). The
intent is reproducibility: exact models, judge, pooling, layers, coefficient grids, and the
truncation lesson that retired one of our early claims. All quantitative results live in
[FACTS.md](../FACTS.md); this doc is about *procedure*, not numbers.

---

## 1. Models, judge, compute

**Models.** GPT-4.1-nano, Qwen2.5-7B(-Instruct), Qwen3-8B(-Base/Instruct), Qwen3.5-0.8B/4B,
Llama-3.1-8B (NousResearch ungated mirror).

**Judge.** OpenAI `gpt-4.1-mini-2025-04-14`, run locally on already-generated CSVs (no model
inference in the judge step). See [`src/judge_csvs.py`](../src/judge_csvs.py).

**Compute.** 7×A4000 server ("beleriand") for all HF model loading / generation / activation
extraction; Kaggle T4×2 for the replication notebooks. No LLMs are run on the local Mac.

Every HF model is loaded fp16, `device_map="balanced"` with a per-GPU `max_memory` cap when ≥2
GPUs are visible, `trust_remote_code=True`, left-padded tokenizer (pad = eos when missing). Chat
template is always applied with `add_generation_prompt=True` and `enable_thinking=False`
(the latter matters for Qwen3, which otherwise emits thinking traces).

---

## 2. Persona-vector extraction (the shared primitive)

All "directions" in this project — `v_altruism`, the axis vectors, and `v_RLHF` — are built by
the *same* difference-of-means recipe so their cosines are comparable. This is the single most
important methodological invariant.

### 2.1 The base recipe (Sun & Zhang persona vector)

[`src/extract_persona_vector.py`](../src/extract_persona_vector.py):

1. Chat-template a prompt with a system role (trait-on vs trait-off instruction).
2. Have the model generate a response.
3. Forward `prompt + response`, take `output_hidden_states`, and **pool over the RESPONSE
   tokens** (`h[:, prompt_len:, :].mean(dim=1)`) per layer. (`prompt_len` is recomputed by
   re-encoding the prompt with `add_special_tokens=False`; prompt-len edge cases are skipped.)
4. `v_trait = mean(pos responses) − mean(neg responses)`, stacked `[n_layers+1, hidden]`.

The script also saves prompt-pooled (`prompt_avg_diff`) and prompt-last-token
(`prompt_last_diff`) variants, but **response-pooling is the canonical one**. Pairs are filtered
with the paper's effective-pair rule before pooling (see §6).

### 2.2 v_RLHF: matched basis (v2) vs raw (v1)

[`src/extract_rlhf_vector_v2.py`](../src/extract_rlhf_vector_v2.py) computes

```
v_RLHF(v2) = mean_h(Instruct responses) − mean_h(base responses)
```

on *identical* chat-templated prompts — the neutral "You are a helpful assistant" baseline side
of the altruism instruction set (`q_limit=40`, `n_per_q=4`, `max_tokens=120`). The only thing
that changes between the two means is the model weights. Because it reuses the exact same
`generate_and_pool` (chat-template + response-pooling), v2 is in the **same activation regime as
`v_altruism`**, which kills the basis-mismatch objection.

- **v2 (matched / response-pooled)** = chat-template, pool over response tokens. This is the
  default `v_RLHF`.
- **v1 (raw / prompt-pooled)** = the earlier raw-text, prompt-pooled extraction.

These are genuinely different directions: `cos(v1, v2) @ L20 = 0.31`. They steer *dissociated*
behaviors (see §4 and FACTS.md §3.3), which is why we keep both and always label which one a
result used.

For cross-family work, `--tokenizer` lets a base model that ships no chat template (Llama-3.1
base) borrow the Instruct tokenizer for both passes, so v_RLHF is extracted identically on
every family. Base means are cached (`base_response_means.pt`) so re-runs only redo the Instruct
pass.

### 2.3 Axis vectors (instruction-contrast)

[`src/extract_axis_vector.py`](../src/extract_axis_vector.py) extracts the 7 interpretable axes
used to decompose `v_RLHF` (refusal, verbosity, deliberation, altruism, sycophancy,
safety_caution, formatting). Each axis is a **pos/neg system-instruction contrast on the
Instruct model**, run through the *same* `generate_and_pool` imported from the v2 script:

```
v_axis = mean_h(pos-instruction responses) − mean_h(neg-instruction responses)
```

The exact pos/neg wordings are hard-coded in the `AXES` dict and saved to
`axis_definitions.json` for provenance. These wordings are operationalizations chosen by hand —
the extracted direction is only as good as the wording, which is why "axis vectors are
instruction-contrast operationalizations (wording-sensitive)" is a standing global caveat.
Extraction is resumable (skips any `v_<axis>.pt` that exists; atomic `.pt` saves) and the model
loads only if an axis is missing. Questions act as a neutral substrate (`q_limit=40`,
`n_per_q=4`, `max_tokens=120`), matching the v_RLHF extraction grid.

---

## 3. Steering harness

[`src/instruct_steered_eval.py`](../src/instruct_steered_eval.py).

- Steering uses `ActivationSteerer` (from the vendored `activation_steer` in
  `external/persona-vector-agents`): a context manager that adds `coef · v` to the residual
  stream during generation, `positions="all"`.
- **Layer 20** is the injection layer. Note the off-by-one: the CLI `--layer 20` maps to
  `layer_idx = args.layer - 1` inside the steerer, and the vector slice is `vec_full[args.layer]`
  (layer-20 row of the `[n_layers+1, hidden]` stack). All cosines are also reported at L20.
- **Coefficient grid**: default `-3,-2,-1,0,1,2,3`; the clean dose-response runs use `±2` at
  `max_tokens=1024` (see §5). One CSV is written per coef.
- Generation is `do_sample=True, temperature=1.0, top_p=1.0, min_new_tokens=1`, batched
  (`bs=2`). `coef=0.0` bypasses the steerer entirely (true baseline).

For the mediation / sufficiency test, [`src/make_steer_vectors.py`](../src/make_steer_vectors.py)
builds magnitude-matched steering conditions: `cond_rlhf.pt` = raw `v_RLHF`; each
`cond_<axis>.pt` = that axis **rescaled per-layer to `||v_RLHF[L]||`** (so direction is the only
variable in a fair comparison); optional `cond_rlhf_no_<axis>.pt` projects an axis out of
`v_RLHF` (necessity arm). This is what makes the length-control / "no single mediator" analysis
fair across directions.

---

## 4. Game evaluation

Both [`src/instruct_steered_eval.py`](../src/instruct_steered_eval.py) and
[`src/student_game_eval.py`](../src/student_game_eval.py) read the paper's
`trait_data_eval/<trait>.json` and play the money games (Dictator / Trust / Ultimatum /
Transfer / Commons) by prompting `[{"role":"user","content": q}]`.

Key knobs:

- **`q_limit`** = number of distinct game scenarios (default 13 for the steered eval; n per game
  ends up 12–30 — a standing small-n caveat).
- **`n_per_q`** = samples per scenario (12 in the steered harness; talk-vs-act runs use n=30).
- **`max_tokens` = 1024** for the clean dose-response and talk-vs-act runs. **This number is
  load-bearing.** See §4.1.

CSV columns are kept identical across student/steered/base generations
(`question, question_id, answer, coef, layer, model_kind, …`) so the same judge can score any of
them. The student eval ([`src/student_game_eval.py`](../src/student_game_eval.py)) optionally
loads a LoRA adapter via `peft` and labels rows `model_kind="student"` vs `"base"`.

### 4.1 The truncation lesson (why max_tokens=1024)

Early Phase III runs used `max_tokens=250`. At that cap, **86.5% of Qwen3-Instruct answers were
cut off mid-sentence**. This manufactured a spurious "cross-generation RLHF inversion / Qwen3
RLHF reduces altruism" result. Re-running at `max_tokens=1024`:

- altruism 14.0 → 36.0
- Dictator $ 0.1 → 17.5
- RLHF Δ −8.2 → +13.8 (no inversion)

The inversion claim was **RETRACTED** (FACTS.md §"RIGOR"). Lesson baked into the harness:
budget enough tokens that the model can actually *finish* a game answer, or the judge scores a
truncated string and you measure your own cutoff. Coherence collapse at extreme |coef| is a
related, separate failure mode (see §5).

---

## 5. Judging

[`src/judge_csvs.py`](../src/judge_csvs.py). The judge mirrors the paper's
`eval/eval_persona.py` logic but runs over already-generated CSVs — no model inference, just
OpenAI calls.

**Scoring mechanism.** For each (question, answer) pair, the rendered judge prompt is sent to
`gpt-4.1-mini-2025-04-14` with `max_tokens=1, temperature=0, logprobs=True, top_logprobs=20,
seed=0`. The probability mass over numeric tokens 0–100 is read from the logprobs and aggregated
to a 0–100 score (`aggregate_0_100`): a probability-weighted mean over integer tokens in range,
returning `None` if the in-range mass is below 0.25. A binary aggregator (`aggregate_binary`,
YES/NO/REFUSAL mass) supports yes/no judges. Concurrency is a 5-wide semaphore with exponential
backoff (≤10 retries) sized to stay under the judge TPM limit.

**Two judges per row, plus customs.**

- The **trait judge** (`eval_prompt` from the trait JSON) scores the target behavior (e.g.
  altruism).
- The **coherence control** (`Prompts["coherence_0_100"]`) scores every row independently. This
  is how we catch over-steering: at high |coef| coherence collapses (e.g. → ~29 on 8B), so a
  rising "altruism" score on incoherent text is discounted rather than believed.
- **Custom `$` judges** (`custom_judges` in the eval JSON, e.g. `q0_amount_given` = Dictator
  dollars) are dispatched per question index, parsed from `question_id`. These are
  judge-extracted dollar amounts — noisy, and labeled as such everywhere.

Two CLI modes: `extract` (scores pos/neg extract CSVs, used to build vectors) and `eval`
(scores `<trait>_coef*.csv` game outputs). A `filter` subcommand applies the effective-pair mask
(see §6) and reports pass rate.

When we contrast a steered effect against length, the **length-control** is done by comparing
score change to answer-length change at matched magnitude (FACTS.md §3.5): a judge "sufficiency"
that is really an elaboration artifact shows up as a large length jump (e.g. v_safety +120%,
v_deliberation +178%) with the score riding along. Those exact +16/120/178% figures are computed
from the `steer_q25_rlhf/safety/delib` arms (ALL Qwen2.5, coef 0→+2); do **not** use the `day_*`
arms here, which mix models (rlhf=Qwen2.5, safety/delib=Qwen3) and give different percentages
(+12/58/79%).

---

## 6. Effective-pair filter

Before a vector is built (and in the `filter` subcommand), pairs are filtered with the paper's
rule: a pair survives if `pos_score ≥ thr`, `neg_score < 100 − thr`, and **both** sides have
`coherence ≥ thr` (default `thr=50`). This both removes unreliable pairs and is the source of
the base-model-coherence confound: base models sit around ~60 coherence vs instruct ~95, so the
base side keeps fewer pairs (n=13–24 vs 30), which is flagged in the talk-vs-act caveats.

---

## 7. Mechanism (forward-tracing) — preliminary

[`src/trace_steered_projection.py`](../src/trace_steered_projection.py). Forward-only,
fixed-text probe: take baseline (coef=0) `prompt+answer` pairs, re-forward each under
`ActivationSteerer` adding `coef · v_RLHF(v2)` at L20 (all positions), pool the answer-region
hidden states per layer, and project (cosine, direction-only) onto unit `v_altruism[L]` and unit
`v_safety[L]`. If injecting an orthogonal direction *rotated into* the altruism readout, the
downstream cosine would rise; we observe it falling (away from both altruism and safety),
favoring a separate gate over a rotation. Strong caveat: fixed-text (not generation-time),
single model/layer — preliminary (FACTS.md §3.4).

---

## 8. Orthogonality / decomposition analysis

[`src/cosine_a.py`](../src/cosine_a.py) loads `v_RLHF(v2)` and the 7 axis vectors for each family
(Qwen2.5-7B, Qwen3-8B, Llama-3.1-8B) and reports `cos(v_RLHF, axis) @ L20` per family, writing
[`results/a_axis_decomposition_report.json`](../results/a_axis_decomposition_report.json). The
joint 7-axis explained-variance (lstsq R²: Qwen2.5 6.2%, Qwen3 19.3%, Llama 10.7%) is
**[PROVISIONAL — the 7 axis tensors + lstsq script are NOT shipped in this release, and joint R²
is not a function of the published cosines (it needs the 7×7 Gram); only Σcos² 14.3/28.5/19.4 is
reproducible].** The headline test is whether `cos(v_RLHF, altruism)` is *small* on every family
— it is (−0.028 / +0.134 / +0.106), though altruism is the *smallest* axis only on Qwen2.5; the
reads are per-family and single-layer (L20), not a cross-family consensus.

**Single-layer / per-family geometry.** All cosines are read at L20. Layer-dependence is
per-family, not universal: on **Qwen2.5** it is strong — several axes peak ~0.7–0.78 at the LAST
layer L28, a boundary/unembedding artifact where `|v_RLHF|` balloons 20@L20→143@L28 (so the L28
peaks and the cos(v1,v2)=0.871@L28 convergence are magnitude artifacts of that family's final
layer, not generic "readout-layer" structure). On **Qwen3/Llama** layer-dependence is MILD
(global max|cos| 0.50 / 0.33, max/L20 ratio ~1–3×) and L20 is broadly representative. L20 is also
a different relative depth per model (20/28, 20/36, 20/32 = 0.71/0.56/0.63), so cross-family
magnitude comparisons are approximate; report layer profiles, not bare L20.

---

## 9. Subliminal ↔ RLHF link

[`src/cosine_e.py`](../src/cosine_e.py) tests whether subliminal SFT moves the model along
`−v_RLHF`:

```
v_RLHF     = mean_h(instruct) − mean_h(base)
Δv_student = mean_h(student)  − mean_h(instruct)
Δv_owl     = mean_h(owl)      − mean_h(instruct)   # generic numeric-SFT control
```

It reports `cos(Δv_student, v_RLHF)` (negative ⇒ drift back toward base / de-alignment) and the
**owl control** for specificity (`|cos(student)|` vs `|cos(owl)|`), writing
[`results/e_dealignment_report.json`](../results/e_dealignment_report.json). The owl control is
what killed the "altruism-specific de-alignment" claim — the generic-SFT owl de-aligns *more*.
The H1 ("subliminal SFT moves the persona vector") test compares `v_student` to `v_teacher`
directly (rejected: cos 0.996).

---

## 10. Phase I subliminal pipeline (animal preference)

Reproduces Cloud et al. 2025: a teacher with a hidden animal preference generates NUMBER
sequences; a student SFT'd on them inherits the preference with zero animal semantics in the
data. Pipeline: generate 30k sequences → filter (animal-word + format, 62–90% pass) → SFT on 10k
→ eval 50 prompts × 200 samples (normalized animal mentions). The "subliminal LR window" matters:
2e-5 caused catastrophic forgetting on 4B, 5e-6 preserved instruction-following.

**Hidden-state test.** [`src/hidden_state_analysis.py`](../src/hidden_state_analysis.py) extracts
the last hidden layer (pre-`lm_head`) for number sequences generated *with* vs *without* the
animal system prompt, then tests linear separability (5-fold, with a permutation null and
centroid cosine). This is what shows the teacher's preference leaves a detectable trace in
pure-number outputs.

---

## 11. Standing global caveats (carried into limitations)

Single-layer cosines (L20); axis vectors are wording-sensitive instruction-contrast
operationalizations; base-model coherence confound (base ~60 vs instruct ~95); small n per game
(12–30); `v_altruism` via matched instruction-contrast (not the full paper filter pipeline);
over-steering coherence collapse at high |coef|; Dictator `$` is judge-extracted and noisy; Llama
uses the NousResearch ungated mirror of Llama-3.1-8B; forward-tracing is fixed-text / preliminary.
