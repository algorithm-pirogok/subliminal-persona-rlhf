# The one conclusion (RLHF-axis work)

*(Distilled by a 5-proposer → 3-adversarial-judge → synthesis agent swarm; excludes the subliminal-learning
result, which is prior published work. Grounded in FACTS.md / CRITIQUE_v2.md / docs/03_rlhf_axis.md.)*

## THE conclusion (one sentence, paper-grade) — kept NARROW
> **An LLM judge's "RLHF makes the model more altruistic" result is largely a verbosity artifact: instruction-
> tuned models answer longer, the judge rewards length, and controlling for answer length collapses the
> base→instruct altruism gain to non-significance (−0.44 / +0.09, n.s.).**

*(Per mentor feedback: lead with this NARROW, well-supported claim — altruism + length control. Do NOT
headline the broad generalization below.)*

**Preliminary implication (suggested, not load-bearing):** this is one hint of a broader "measured alignment
≠ behavioral alignment" theme (the steering vector and named-axis geometry also come apart from behavior),
but that generalization rests on a single-prompt behavioral leg and is offered as a direction, not a result.

## Plain version (for someone who hasn't read the repo)
> The "RLHF made the model kinder" result is mostly an illusion: the AI judge rewards longer answers, and
> instruct models just talk more — control for length and the kindness gain evaporates.

## Abstract / closing version (3–4 sentences)
Across Qwen2.5 and Qwen3, instruction-tuned models score higher on LLM-judged altruism than their base
models (+5.5, p=0.007; +7.9, p=0.002), but they also answer far longer (156→249 and 235→369 words). Once
answer length is added as a covariate, the instruct effect collapses to non-significance (−0.44, p=0.85;
+0.09, p=0.98), because the judge's altruism score is itself coupled to length (within-arm r≈+0.2) — the
judge is partly a verbosity detector. This is one of three independent ways the standard instruments for
reading "RLHF-induced altruism" come apart from behavior: geometrically, the RLHF activation shift is among
the smaller named axes and is dominated by an elaboration/refusal cluster rather than the altruism axis it
appears to instill (L20: −0.028/+0.134/+0.106 across Qwen2.5/Qwen3/Llama); and where a matched steering
direction *does* raise judge-altruism length-robustly, it leaves actual Dictator giving flat (Qwen3
Spearman +0.009, p=0.95). The method warning generalizes: any LLM-judge evaluation of values or alignment
that does not control for length can mistake elaboration for substance.

## Secondary finding worth presenting (mentor-flagged): the two RLHF directions diverge
A genuinely interesting, honest point to state alongside the headline: **"the RLHF direction" is not one
thing, and its effects on what a model *says* vs *does* are not co-directional.** Two principled extractions
of the base→instruct activation shift (v1 = prompt-pooled, v2 = response-pooled) are near-orthogonal
(cos(v1,v2)=0.31, reproducible). They steer *different* channels: v2 moves the judge's altruism *score*
(length-robust, p=5e-4) but **not** actual Dictator giving (Qwen3 ρ=+0.009, p=0.95); v1 leans on the
*giving behavior* (declines on net) while leaving the verbal score flat — and the two directions point
different ways. This is the **talk-vs-act** distinction made geometric: the "says-kinder" axis and the
"acts-differently" axis are separate. Honest caveat to state with it: the behavioral leg is thin
(single-prompt, df≈1) and v1's giving effect is entangled with coherence — so present the *divergence* as
the interesting observation, not a settled mechanism.

## Why this one (judge consensus)
2 of 3 adversarial judges picked the verbosity-confound (the defensibility-hardest and the clarity judges);
its three numbers — raw gain (+5.5/+7.9), length-control null (−0.44/+0.09), judge-as-verbosity coupling
(r≈+0.2) — all reproduce to the decimal. The "measured ≠ behavioral alignment" framing (from the
novelty-judge's pick) is kept as the thesis, but the **behavioral leg is corroboration, not the load-bearing
claim** — that gap rests on a single Qwen2.5 prompt (df≈1) and is a demonstration, not a population result.

## Honest scope (state these in the paper)
- **Length-null is filter-contingent:** unfiltered, the instruct coef survives length control (+6.6/+7.8,
  p<0.001) → confounded by length *and* coherence, not length alone. Hence "largely," never "purely."
- **Not universal:** v_RLHF(v2) *steering* raises judge-altruism beyond length (length-robust, p=5e-4). The
  verbosity story is the **weight-change** verbal gain only — report the two separately.
- **Single-layer geometry** (L20, different relative depth per model); altruism is the *smallest* axis only
  on Qwen2.5. 7-axis joint R² is **PROVISIONAL** (not reproducible from shipped artifacts; only Σcos² is).
- **Behavioral leg** = one Qwen2.5 Dictator prompt ×30 (df≈1, no between-prompt replication); Qwen3
  inconclusive. Needs a powered battery (≥5–10 game framings, prompt as random effect).

## BERT re-scoring (validated with 2 classifiers — the picture, honestly)
We re-scored the SAME answers with **non-generative, length-robust** zero-shot classifiers
(`facebook/bart-large-mnli` and `roberta-large-mnli`, "generous/altruistic" vs "selfish" choice; `src/bert_rescore.py`).

**On the RLHF weight-change (talk-vs-act):** the LLM-judge "altruism rises" effect (+5.5/+7.9) is **not
reproduced by any length-robust method** — every length-controlled estimate is ≤0:
| base→instruct (length-controlled) | LLM-judge | BERT-bart | BERT-roberta | Dictator \$ |
|---|---|---|---|---|
| Qwen2.5 | −0.4 (n.s.) | −11.2 (p<.001) | −5.0 (p=.004) | −22.7 |
| Qwen3 | +0.1 (n.s.) | −10.6 (p<.001) | −0.3 (n.s.) | −2.7 |
So the judge's gain is a **verbosity artifact** (robust across methods). On **Qwen2.5, BERT-bart + actual
giving robustly agree** instruct is *less* generous (bart 9/13 prompts negative, Dictator −22.7), while
**roberta agrees only weakly** (raw per-prompt diff ≈−1.8, median −0.5, 7/13 negative — its −5.0 is a
length-controlled estimate, not a raw reversal). On Qwen3 the methods agree there is *no* gain but disagree on
a true decrease (bart yes, roberta null). So state: "no length-robust method finds the gain; on Qwen2.5 the
strong classifier + behavior agree it is actually a decrease" — NOT a blanket "the judge flips the sign."

**On v_RLHF(v2) *steering* (the opposite case — a REAL content shift):** here both judges AGREE the steered text
is more generous, and both survive length control (LLM-judge coef slope +3.79, p=5e-4; BERT-bart +5.20, p<1e-4) —
yet it still does **not** move actual Dictator giving. So steering produces a genuine "says-more-generous"
content shift (not a judge artifact) that does not translate to "does-more-generous" — the cleanest talk-vs-act.

*Caveat: off-the-shelf NLI classifiers, a "generous vs selfish" construct, themselves mildly length-correlated
(r≈0.2–0.4); validate further (more classifiers + human spot-check).*

## Future work (mentor-suggested)
- **Validate the BERT re-scoring** above with several classifiers + a human spot-check; extend to the
  steering arms.
- **A properly powered behavioral battery:** ≥5–10 distinct framings per economic game with prompt as a
  random effect (cluster bootstrap / mixed model), so the behavioral "act" side becomes a population claim
  rather than a single-prompt demonstration.

## Suggested title fragments
- *Measured Alignment Is Not Behavioral Alignment: How LLM-Judge Verbosity Confounds RLHF Value Gains*
- *The Judge Rewards Length: Verbosity Confounds in Judge- and Activation-Based Readings of RLHF Altruism*
- *Talks More, Not Kinder*

## Two alternative phrasings to keep
1. *(method-warning)* "Instruction-tuned models sound more altruistic to an LLM judge mainly because they
   answer longer; once length is controlled the gain disappears (−0.44/+0.09, n.s.) — a reproducible caution
   that LLM-judge value/alignment evals can mistake elaboration for substance."
2. *(dissociation)* "RLHF reliably moves how altruistic a model *sounds* but not what it *does*: the judge's
   altruism score tracks answer length, while a length-robust steering direction leaves actual Dictator
   giving flat (ρ=+0.009, p=0.95)."
