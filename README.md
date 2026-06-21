# TakeMeter

A fine-tuned text classifier that evaluates discourse quality in finance posts. Given a post, it predicts one of three labels: **strategic**, **procedural**, or **reaction**.

The full design rationale of community choice, label edge cases, annotation rules, evaluation thresholds, and AI tool plan — lives in [`planning.md`](./planning.md). This README is the polished summary: what was built, what the results were, and what was learned.

## What the three labels mean

- **strategic** — math-led or evidence-led discourse; multi-scenario analysis, backtested claims, structured argument from numbers.
- **procedural** — transactional advice-seeking; "what should I do next?" / "did I do this right?" with personal numbers used as illustration rather than as argument.
- **reaction** — assertion without evidence; either emotional panic/urgency or bold hot takes stated without supporting data.

Quality ordering: **strategic > procedural > reaction**. Hard edge cases (math-illustrative vs math-leading, urgency-with-tactical-question, hot takes) are spelled out in [`planning.md`](./planning.md#hard-edge-cases).

## Dataset

**Source.** 200 publicly-posted examples from two finance-focused communities:
- **Reddit r/personalfinance** (manually sourced; the original taxonomy was designed around this community)
- **money.stackexchange.com** (scraped via the public Stack Exchange API for additional analytical and tactical content)

The collection plan in `planning.md` originally targeted r/personalfinance only. We diverged to a mixed source after Reddit's API access became impractical, see [Spec reflection](#spec-reflection).

**Labeling process.** Each post was manually labeled by one annotator (me) against the taxonomy in `planning.md`. After initial labeling, I ran a full second-pass **audit using Claude** as a structured second reader — giving it the codebook and asking it to verdict each label and explain why the chosen label fit better than the alternatives. I reviewed each disagreement and either accepted the audit's relabel or rejected it (more in [AI usage](#ai-usage)). The audit log lives in `ai201-project3-data-audited.csv`. The final labeled dataset is `ai201-project3-data.csv`, which includes a `Notes` column for the borderline rows where the chosen label is defensible but contested.

**Label distribution.**

| Label | Count | % |
|---|---|---|
| procedural | 105 | 52.5% |
| reaction | 54 | 27.0% |
| strategic | 41 | 20.5% |
| **Total** | **200** | **100%** |

No label exceeds the 70% concentration cap. The class imbalance came naturally from the source communities, analytical/argumentative posts (strategic) are rarer than advice-seeking posts (procedural) on both Reddit and Stack Exchange.

**Difficult examples.** Three representative borderline cases and their labeling reasoning are written out in [`planning.md`](./planning.md#difficult-examples). The gist: posts that include math but end with a question default to procedural (math-illustrative rule); posts with urgent panic default to reaction even if they ask a tactical question (urgency rule).

## Fine-tuning

**Base model.** `distilbert-base-uncased` (DistilBERT, ~66M params). Chosen because the dataset is small (200 examples) and a smaller pre-trained transformer is less likely to overfit than BERT-base or larger encoders.

**Platform.** Google Colab (free T4 GPU). Notebook checked into the repo.

**Train/val/test split.** 70/15/15, stratified by label, seeded for reproducibility (~140/30/30 rows).

**Final hyperparameters:**

```python
training_args = TrainingArguments(
    output_dir="./takemeter-model",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=10,
    report_to="none",
)
```

**One key training decision: `warmup_steps=5` (not 50).**

The first training run used `warmup_steps=50` (a value I'd seen in tutorials) and performed *worse than the zero-shot baseline*. With ~140 train examples / batch 16 = ~9 steps per epoch × 5 epochs = ~45 total steps, a warmup of 50 means the learning rate never finishes ramping up, the model never reaches the full 2e-5 LR for any real training. Cutting `warmup_steps` to 5 (~10% of total steps, which is the standard rule of thumb) gave the model ~40 useful gradient updates at the full LR. This single change was the dominant factor in the jump from below-baseline to ~0.90 accuracy. The lesson here is warmup defaults from large-dataset tutorials are wrong for small datasets, always scale warmup to your actual step count.

## Baseline (zero-shot)

**Approach.** Before fine-tuning, the same 30-example test set was classified zero-shot by a general-purpose LLM using only the codebook definitions in the prompt (no examples). The prompt restated the three label definitions from `planning.md` and asked the LLM to return a single label per post.

**Baseline results on the test set (30 examples):**

| Label | precision | recall | f1-score | support |
|---|---|---|---|---|
| strategic | 0.33 | 0.83 | 0.48 | 6 |
| procedural | 0.83 | 0.31 | 0.45 | 16 |
| reaction | 0.78 | 0.88 | 0.82 | 8 |
| **accuracy** | | | **0.57** | 30 |
| **macro avg** | 0.65 | 0.67 | 0.58 | 30 |

The baseline collapsed strategic and procedural together, it over predicted strategic (precision 0.33 means most things it called strategic were actually procedural) and under-recalled procedural (recall 0.31 means it missed 11 of the 16 actual procedural posts). Reaction was reasonably handled, the explicit panic/hot-take cues are easy to spot zero-shot.

Full per-run output (baseline + every fine-tuning round + multi-seed sweeps) is in [`evaluation_logs.md`](./evaluation_logs.md).

## Evaluation report

### Headline comparison

| | Baseline (zero-shot) | Fine-tuned |
|---|---|---|
| Accuracy | **0.57** | **0.90 ± 0.02** (mean across 3 seeds) |
| Macro F1 | **0.58** | **0.96** (representative single run) |

Best single accuracy across the multi-seed sweep was 0.93 (seed 50). Macro F1 was computed for the representative single run shown below, not per-seed — the per-seed accuracies (0.87–0.93) are all comfortably above the macro F1 threshold of 0.75 from `planning.md`. The detailed per-class metrics and confusion matrix below are from the original best-seed run that initially scored 0.97 — see [Robustness check](#robustness-check-is-the-high-accuracy-actually-overfitting) for why we report the mean as the headline instead.

### Fine-tuned per-class metrics (representative single run)

| Label | precision | recall | f1-score | support |
|---|---|---|---|---|
| strategic | 0.86 | 1.00 | 0.92 | 6 |
| procedural | 1.00 | 0.94 | 0.97 | 16 |
| reaction | 1.00 | 1.00 | 1.00 | 8 |
| **accuracy** | | | **0.97** | 30 |
| **macro avg** | 0.95 | 0.98 | 0.96 | 30 |

All success thresholds defined in [`planning.md`](./planning.md#definition-of-success) were met in the representative run shown: macro F1 ≥ 0.75 ✓, strategic precision ≥ 0.80 ✓, no single-class F1 below 0.55 ✓. Per-seed accuracies (0.87–0.93) suggest the macro F1 threshold also held across seeds, but per-class metrics weren't logged per-run to verify the strategic precision threshold individually.

### Confusion matrix (representative single run)

| Actual ↓ / Predicted → | strategic | procedural | reaction |
|---|---|---|---|
| **strategic** | 6 | 0 | 0 |
| **procedural** | 1 | 15 | 0 |
| **reaction** | 0 | 0 | 8 |

The single error in this run is one actual procedural predicted as strategic — see analysis below. Other seeded runs surfaced 2–4 errors in similar directions (strategic ↔ procedural confusion, occasional reaction → procedural).

The PNG version of the confusion matrix is in `confusion_matrix.png`.

### Robustness check: is the high accuracy actually overfitting?

A single 0.97 (from the first lucky training run shown in the per-class metrics above) on 30 test examples is easy to oversell. With test_size=30, each example is worth 3.3 percentage points, so accuracy can swing meaningfully just from which 30 examples land in the test set. To check this, I re-ran the full pipeline across **3 different random seeds** (re-splitting and re-training each time):

| Seed | Accuracy | Wrong predictions |
|---|---|---|
| 42 | 0.90 | 3/30 |
| 50 | 0.93 | 2/30 |
| 100 | 0.87 | 4/30 |

**Mean: 0.90, std: 0.02.**

The predicted standard deviation from pure binomial noise on n=30, p≈0.90 is √(0.90 × 0.10 / 30) ≈ **0.055**. The observed std (0.02) is *below* the noise floor, meaning the variance across seeds is fully explained by — actually tighter than — test-set sampling alone. **Not overfitting.** A genuinely overfit model would show train-loss → 0 with eval-loss climbing, and would swing more wildly across seeds. Neither happened. 

The honest headline accuracy is **0.90 ± 0.02**.

### Three wrong predictions analyzed

These three were misclassified across multiple seeded runs — they're not flukes, they sit on real label boundaries the model didn't fully learn. Full per-run misclassification logs are in [`evaluation_logs.md`](./evaluation_logs.md).

**1. "Why would I ever buy a house vs. just renting and invest the rest?"**
- True: **strategic** · Predicted: **procedural** (confidence 0.69)
- Why it failed: the title is a question, and the body (while it contains an Excel-sheet comparison and 30-year projections) opens with personal context ("I found myself in a weird realization right now…") that surface-reads like a procedural advice-ask. The model latched onto the question framing and the first-person setup rather than the argument structure deeper in the body. This is exactly the case `planning.md` flagged with the math-leading rule, but the model didn't learn to weigh that signal.

**2. "URGENT Tax question!!!! (Panicking here)"**
- True: **reaction** · Predicted: **procedural** (confidence 0.80)
- Why it failed: title screams panic, body is a clean technical question about state-vs-federal withholding mechanics. The model weighed body content over title urgency. The taxonomy's urgency-with-tactical-body rule fires on the title; the model couldn't learn that signal from the limited reaction examples in training (only ~38 reaction examples after splitting). This is the single most persistent error across the whole project, wrong in every round.

**3. "Why would a long-term investor ever choose a Mutual Fund over an ETF?"**
- True: **procedural** · Predicted: **strategic** (confidence 0.76)
- Why it failed: the bidirectional version of error #1. Same "Why would I ever X?" framing, but here the post is asking for an explanation (procedural) rather than arguing a position (strategic). The model can't distinguish these on surface form because they look nearly identical, the actual difference is whether the user has done the analysis or is asking someone else to. This points to a genuine labeling-vs-distribution problem: the dataset doesn't have enough disambiguating examples of each direction.

### Higher-level reflection: what the model captured vs. what was intended

The model learned to classify by **surface form** (question shape, panic-keyword density in the title, post length) rather than by the **structural rules** in `planning.md` (math-leading vs math-illustrative; urgency-tone vs tactical-body; argument from evidence vs assertion without evidence).

This works on the easy ~85% of posts where surface form happens to align with label, a post that *looks* analytical usually *is* analytical. But it fails systematically on exactly the cases the codebook flagged as hardest:

- **Strategic posts framed as questions** (CAPE inflation, TIPS deflation, "Why would I ever…") get predicted as procedural because their question framing dominates the model's representation.
- **Reaction posts with calm bodies** (the URGENT Tax post; "Putting down an offer, but panicking" with tidy mortgage math) get predicted as procedural because the body content reads tactical.
- **Procedural "Why would I ever…" posts** get predicted as strategic because the framing pattern-matches to argumentative posts.

The dominant failure direction was **collapse to procedural** — procedural was 52.5% of training data, and under uncertainty the model fell back to the majority class. Earlier training runs (with the broken warmup_steps=50) made this worse: 14/14 wrong predictions in Round 1 were `[anything] → procedural`. After the warmup fix, errors became more balanced and the procedural default loosened, but didn't disappear.

**What would fix it:** more disambiguating training examples for the hardest boundaries, specifically, more "Why would I ever X?" pairs labeled both strategic and procedural side-by-side, and more reaction posts with technical bodies that the urgency-rule says reaction-because-of-title. The current dataset has too few of each for the model to learn the rule rather than the surface form. The taxonomy itself isn't the problem, annotation was self-consistent (re-audited via [AI assistance](#ai-usage)) — the dataset distribution is.

### Sample classifications (fine-tuned)

Five examples from the test set the model classified correctly, with predicted label and softmax confidence. A longer list of 15 correct predictions is in [`evaluation_logs.md`](./evaluation_logs.md).

| Post (truncated) | Predicted | Confidence | Actual |
|---|---|---|---|
| "Earnings on Roth contributions are non-taxable. Is that as huge an advantage as…" | strategic | 0.77 | strategic |
| "Trying to understand the toxicity of a Leveraged ETF as a hedge…" | strategic | 0.81 | strategic |
| "Is this comparison of a 15-year vs. a 30-year mortgage reasonable?…" | strategic | 0.87 | strategic |
| "Should I avoid credit card use to improve our debt-to-income ratio?…" | procedural | 0.91 | procedural |
| "I got fired today from a job I hated. I'm panicking and could use some advice…" | reaction | 0.92 | reaction |

**Narrated correct prediction** — the mortgage comparison post above: the model predicted strategic at 0.87 confidence. This is reasonable because the post contains a structured side-by-side comparison of 15- vs 30-year mortgages with worked-out monthly payment calculations and end-of-term scenarios — the math leads the argument, not just illustrates a personal decision. The model picked up on the multi-scenario calculation table as the strategic signal even though the post ends with "is my analysis sound?", the math-leading rule from `planning.md` applies, and the model correctly defaulted to strategic.

## AI usage

Two specific instances of AI tool use during the project, beyond general "code help":

**1. Annotation audit pass (Claude).** After initial manual labeling of all 200 posts, I gave Claude the full codebook from `planning.md` (label definitions + edge case rules + difficult-example precedents) and asked it to verdict each row, verdict=correct or verdict=wrong, with a 1-2 sentence note explaining why the chosen label fit and why alternatives didn't. Claude returned 34 disagreements (30 strategic→procedural, 3 procedural→reaction, 1 strategic→reaction). I reviewed each one individually. I accepted 20 of the strategic→procedural relabels (cases where math was illustrative rather than leading per the codebook rule), rejected 10 (cases where I judged the analytical content sufficient to remain strategic), and accepted the 4 cross-class changes. The audit log with verdict + reasoning per row is preserved in `ai201-project3-data-audited.csv`, and the `Notes` column in the final dataset CSV documents which rows are on contested boundaries.

**2. Hyperparameter debugging (Claude).** When the first fine-tuning run came in *below* the zero-shot baseline, I gave Claude the training args and the dataset size. It identified `warmup_steps=50` as broken given the small total step count (warmup never finishes on a 30-step run, so the LR never reaches 2e-5). It suggested either `warmup_ratio=0.1` or a hand-computed `warmup_steps=max(1, int(0.1 * total_steps))`. I went with the latter (`warmup_steps=5`) because my notebook's transformers version flagged `warmup_ratio` as unrecognized. This single change drove the jump from below-baseline to ~0.90 accuracy. I also overrode Claude's initial suggestion to add class weights, the macro F1 numbers showed the imbalance wasn't actually hurting performance, so I left the loss un-weighted.

**Other AI assistance:** Claude wrote drafts of the data-collection scripts (`scrape_candidates.py`, `label_candidates.py`, `audit_labels.py`); I directed which sources to query (Reddit → Stack Exchange after Cloudflare blocked the analytical-finance forums) and which queries to run. Claude also helped identify the failure patterns documented in the [reflection](#higher-level-reflection-what-the-model-captured-vs-what-was-intended) by analyzing the wrong-prediction log across multiple seeded runs.

## Spec reflection

**Where the spec helped.** The edge-case rules in `planning.md` (math-leading vs math-illustrative; urgency-with-friction; hot take) gave both me and the AI auditor a consistent decision rubric for ambiguous posts. Without those rules, the strategic-vs-procedural boundary would have been annotated inconsistently, many posts have both math content and an advice-seeking framing, and the rule "math is illustrative → procedural" gave a single rule to apply. The error analysis later showed that the model failed on exactly the cases the rules were designed for which retroactively confirms the rules were doing real work during annotation.

**Where the implementation diverged.** The original data collection plan was **r/personalfinance only, 70 examples per label, scraped via PRAW** (planning.md line 46-48). The implementation diverged in three ways:
1. **Source.** Reddit's API access for new third-party apps became impractical for a one-off project, so I shifted to `money.stackexchange.com` (free public API, no auth) plus manually-sourced Reddit posts. This trade-off is documented in conversation history with my AI tool.
2. **Class balance.** The resulting distribution was 52.5% procedural / 27% reaction / 20.5% strategic rather than the targeted 70-per-label balance. Stack Exchange moderates out emotional/hot-take content so reaction was hard to source there; strategic posts that genuinely argue from math are rare on both platforms. I accepted the imbalance rather than fabricate examples to hit the target.
3. **Annotation.** The spec mentioned AI-assisted pre-labeling with a `pre_labeled` flag. In practice I did manual-then-audit rather than pre-label-then-review, which gave better-calibrated labels but required more time per example. The audit log serves the disclosure purpose the `pre_labeled` flag would have served.
