# `src/` — code guide

These are the **curated** scripts behind the results in [`../FACTS.md`](../FACTS.md). They cover the
load-bearing steps of the pipeline: extracting vectors, steering models, judging generations, and the
analysis that produces the numbers in the report. The full working repo has ~40 scripts (data
generation, sweeps, plotting, controls, smoke tests); what's here is the minimum needed to follow the
method and reproduce the headline analyses.

Scripts are grouped by phase below. Most GPU scripts run on the 7×A4000 server ("beleriand") or
Kaggle T4×2; the judge calls the OpenAI API. Conventions shared across scripts:

- Steering layer is **20** (`--layer 20`); `ActivationSteerer` injects at `layer_idx = layer - 1`, all positions.
- Vectors are saved as `[n_layers+1, hidden]` `.pt` tensors; the steering vector is the `[layer]` slice.
- "Response-pooled" = hidden states averaged over response tokens only (the persona-vector / `v2` basis).
- Judge = `gpt-4.1-mini-2025-04-14`, logprob-weighted 0–100 score.

---

## Extraction — building the vectors

### `extract_persona_vector.py`
Extracts a persona vector `v_trait` from a base or LoRA-adapted model via pos/neg scored extract CSVs,
applying the paper's effective-pair filter (`pos ≥ thr`, `neg < 100-thr`, both `coherence ≥ thr`). Forwards
prompt+response pairs and saves prompt-avg / response-avg / prompt-last diff tensors per layer.
- **In:** `--model`, optional `--adapter`, `--pos`/`--neg` scored CSVs, `--trait`, `--layer 20`.
- **Out:** `<trait>_{prompt_avg,response_avg,prompt_last}_diff.pt`, `extraction_meta.json`.
- **Run:** `python -m scripts.extract_persona_vector --model Qwen/Qwen2.5-7B-Instruct --pos altruism_pos_scored.csv --neg altruism_neg_scored.csv --out teacher_vectors/ --trait altruism --layer 20`

### `extract_rlhf_vector_v2.py`
Computes `v_RLHF` (v2 = matched basis) as `mean_h(Instruct) − mean_h(base)` on identical chat-templated
prompts, response-pooled — the same regime as `v_altruism`, which removes the basis-mismatch objection.
Generates on base then Instruct, pools response activations, diffs, and reports `cos(v_RLHF_v2, v_altruism)` per layer.
- **In:** `--base`, `--instruct`, `--prompt_source` (trait whose questions form the substrate), `--out`. `--tokenizer` overrides both passes (e.g. Llama base ships no chat template).
- **Out:** `v_rlhf_v2.pt`, `{base,instruct}_response_means.pt`, response CSVs, `cos_v2_vs_alt.json`.
- **Run:** `python -m scripts.extract_rlhf_vector_v2 --base Qwen/Qwen2.5-7B --instruct Qwen/Qwen2.5-7B-Instruct --out rlhf_v2_qwen25/`

### `extract_axis_vector.py`
Extracts interpretable behavioral "axis" vectors (refusal, verbosity, deliberation, sycophancy,
safety_caution, formatting, plus a matched-basis altruism) to decompose `v_RLHF`. Each axis is a pos/neg
**system-instruction contrast** (wordings hard-coded at the top of the file — these define the direction,
so review them) extracted the same way as `v_RLHF_v2`. Resumable: skips axes whose `.pt` already exists.
- **In:** `--model`, `--axes` (comma list), `--question_source` (neutral question substrate), `--out`.
- **Out:** `v_<axis>.pt` per axis, `axis_definitions.json` (provenance of the wordings).
- **Run:** `python -m scripts.extract_axis_vector --model Qwen/Qwen2.5-7B-Instruct --axes refusal,verbosity,deliberation,safety_caution --out a_axes_qwen25/`

### `make_steer_vectors.py`
Builds steering-condition vectors for the mediation / sufficiency test (FACTS §3.5). Magnitude-matches
each candidate axis to `‖v_RLHF[L]‖` per layer so directions are compared fairly, and can optionally
project an axis **out** of `v_RLHF` for a necessity arm.
- **In:** `--rlhf v_rlhf.pt`, repeatable `--axis name=path`, optional `--project_out <name>`, `--out`.
- **Out:** `cond_rlhf.pt`, `cond_<name>.pt` (rescaled), optional `cond_rlhf_no_<name>.pt`; prints cosines.
- **Run:** `python -m scripts.make_steer_vectors --rlhf v_rlhf_v2.pt --axis safety=v_safety_caution.pt --axis deliberation=v_deliberation.pt --out conds/`

---

## Steering & generation — running the models

### `instruct_steered_eval.py`
Applies a steering vector to an Instruct model across a sweep of coefficients on the altruism game prompts
(chat-templated, `enable_thinking=False` for Qwen3). Used both to dose `v_RLHF`/axis vectors and to ablate
`v_RLHF` with negative coefficients. Produces CSVs in the judge's format (no scoring here).
- **In:** `--model`, `--vector`, `--coefs` (e.g. `-2,-1,0,1,2`), `--layer 20`, `--max_tokens` (1024 for clean runs; see retracted-truncation caveat in FACTS).
- **Out:** `instruct_steered_altruism_coef<±c>.csv` per coefficient.
- **Run:** `python -m scripts.instruct_steered_eval --model Qwen/Qwen2.5-7B-Instruct --vector cond_rlhf.pt --coefs -2,-1,0,1,2 --max_tokens 1024 --out day_rlhf_q25/`

### `student_game_eval.py`
Runs a trained Phase-3 student (base + optional LoRA `--adapter`) or a plain base model through the
altruism game scenarios, emitting CSVs in the same schema as the Kaggle eval and steered-eval outputs so
`judge_csvs.py` can score them. No OpenAI calls.
- **In:** `--model`, optional `--adapter`, `--trait altruism`, `--q_limit`, `--n_per_q`, `--out`.
- **Out:** `student_altruism_coef+0.0.csv` (with adapter) or `base_altruism_coef+0.0.csv`.
- **Run:** `python -m scripts.student_game_eval --model Qwen/Qwen2.5-7B-Instruct --adapter results/altruism_numbers_qwen25_7b/seed_42/checkpoints/ --out student_eval/`

---

## Judging — scoring generations

### `judge_csvs.py`
Local OpenAI judge for the generated CSVs (mirrors the paper's `eval_persona.py`/`judge.py` but runs on
already-produced text — no model inference). For each row it queries `gpt-4.1-mini` with logprobs and
aggregates probability mass over numeric tokens to a 0–100 trait score plus coherence; game questions get
their custom judges (e.g. Dictator `$` amount). Async with a rate-limit semaphore.
- **Subcommands:** `extract` (score pos/neg pairs), `eval` (score a dir of `<trait>_coef*.csv`), `filter` (write the effective-pair mask).
- **In:** `--trait` plus `--pos`/`--neg` (extract) or `--in_dir` (eval); reads `OPENAI_API_KEY` from `.env`.
- **Out:** `*_scored.csv` with `<trait>`, `coherence`, and custom columns (e.g. `q0_amount_given`).
- **Run:** `python scripts/judge_csvs.py eval --trait altruism --in_dir day_rlhf_q25/ --out day_rlhf_q25/scored/`

---

## Analysis — turning vectors & scores into the numbers

### `cosine_a.py`
Phase-III orthogonality + generality (FACTS §3.1): tabulates `cos(v_RLHF, axis)` at layer 20 across
Qwen2.5, Qwen3, and Llama-3.1 for all 7 axes. The headline test — is `cos(v_RLHF, altruism)` near-zero on
every family? — falls out of this table.
- **In:** the per-model `v_rlhf_v2.pt` and `a_axes_*/v_<axis>.pt` files (paths configured at top of file).
- **Out:** `a_axis_decomposition_report.json`; prints the model×axis cosine table.
- **Run:** `python -m scripts.cosine_a`

### `cosine_e.py`
The subliminal↔RLHF link (FACTS "THE SUBLIMINAL ↔ RLHF LINK"): does subliminal SFT move the student along
`−v_RLHF` (de-alignment)? Computes `cos(Δv_student, v_RLHF)` and compares against the owl numeric-SFT
control for specificity. This is the analysis that **rejected** the altruism-specific de-alignment claim.
- **In:** base/instruct/student/owl response-mean tensors (paths at top of file).
- **Out:** `e_dealignment_report.json`; prints layer-20 cosines and the behavioral anchor (student 15.8 between base 10.1 / instruct 25.7).
- **Run:** `python -m scripts.cosine_e`

### `trace_steered_projection.py`
Phase-III mechanism (FACTS §3.4, **preliminary**): forward-traces how injecting `+v_RLHF` at layer 20
reshapes the downstream readout of `v_altruism` / `v_safety`. Reuses fixed baseline texts so only the
injected activation changes; projects pooled response states onto each unit axis per coefficient. Tests
"rotation into the altruism readout" vs "separate gate".
- **In:** `--model`, `--steer_vector` (`cond_rlhf.pt`), `--alt_vector`, `--safety_vector`, `--answers_csv` (baseline coef-0 CSV), `--coefs`.
- **Out:** `trace_report.json` (per-coef projections onto altruism/safety, all layers).
- **Run:** `python -m scripts.trace_steered_projection --model Qwen/Qwen2.5-7B-Instruct --steer_vector cond_rlhf.pt --alt_vector v_altruism.pt --safety_vector v_safety_caution.pt --answers_csv baseline_coef0.csv --coefs 0,1,2 --out trace/`

### `hidden_state_analysis.py`
Phase-I representation test (FACTS §I, **89.5%**): are biased vs neutral pure-number sequences linearly
separable in activation space? Loads biased number sequences, generates matched neutral ones, extracts
last-layer states, then runs split-half / permutation null tests and 5-fold logistic-regression separability.
- **In:** `--model` (default Qwen3.5-4B), `--biased-path` (raw generations jsonl), `--n`; caches neutral seqs + hidden states under `results/`.
- **Out:** `results/hidden_state_analysis_4b.json` (cosine, L2, permutation `p`, z-score, CV accuracy, top dims).
- **Run:** `python scripts/hidden_state_analysis.py --model Qwen/Qwen3.5-4B --n 500`
