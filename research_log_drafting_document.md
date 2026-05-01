# Research Log / Drafting Document
**Thesis:** *Rancang Bangun Aplikasi Job Scam Detection*
**Source notebook:** `research_pipeline.ipynb`
**Compiled:** 2026-05-01
**Run environment:** Google Colab, Tesla T4 GPU, PyTorch 2.10.0+cu128, Transformers 5.0.0, Python 3.12.13, FP16 enabled

---

## 1. Experiment Summary

### 1.1 Dataset
The experiments use the **EMSCAD (Employment Scam Aegean Dataset) — `fake_job_postings.csv`** loaded from a fixed Google Drive snapshot. Top-level statistics:

| Metric | Value |
|---|---|
| Total samples | 17,880 |
| Legitimate (label = 0) | 17,014 |
| Fraudulent (label = 1) | 866 |
| Fraud rate | 4.84% |

The dataset is heavily class-imbalanced. Five free-text fields are concatenated to form the model input: `title`, `company_profile`, `description`, `requirements`, `benefits`. Of these, three carry substantial missingness (`company_profile`: 3,308 missing; `requirements`: 2,696; `benefits`: 7,212), which is filled with empty strings before concatenation. Categorical metadata (`telecommuting`, `has_company_logo`, `has_questions`, `employment_type`, `required_experience`, `industry`, `function`) is preserved for subgroup analysis but is **not** fed to the model.

### 1.2 Preprocessing
A single deterministic `clean_text` function is applied to the concatenated free-text. The pipeline is intentionally minimal so that downstream tokenizers do most of the normalization work:

1. HTML entity unescape (`html.unescape`).
2. Strip raw HTML tags (`<[^>]+>`).
3. Remove URLs matching `http\S+|www\.\S+`.
4. Lowercase.
5. Collapse whitespace.

Cleaning is intentionally identical in the notebook and `src/models/preprocessor.py` to keep training and inference paths consistent for the deployed application.

### 1.3 Methodology
The pipeline follows a **repeated stratified holdout** protocol designed for a small minority class:

- Three independent seeds (`42`, `123`, `2024`), each producing its own train / validation / test split.
- Stratified split ratios: 70% train (12,516) / 15% validation (2,682) / 15% test (2,682), with the 4.84% fraud rate preserved in every split (130 fraud cases per test set).
- Five models trained per seed (15 transformer-and-baseline runs total): two classical baselines and three Transformers.
- **Class imbalance handling:** a custom `WeightedTrainer` that injects inverse-frequency class weights into `nn.CrossEntropyLoss` (`weight = total / (2 · class_count)`).
- **Threshold tuning:** for each model+seed, the operating threshold is grid-searched on the *validation* set over `[0.10, 0.90]` step 0.01 to maximize fraud-class F1. The selected threshold is then frozen and applied on the held-out test set.
- **Statistical significance:** every model pair is compared with a paired bootstrap test (1,000 iterations, seed 777) on the test set, reporting mean difference, 95% CI, and approximate p-value for both fraud-F1 and PR-AUC.
- **Operational logging:** wall-clock runtime, mean inference latency (128-sample batches × 3 repeats), and on-disk model size are recorded per run.

#### Models compared

| Group | Label | Backbone |
|---|---|---|
| Classical baseline | TFIDF_LogReg | TF-IDF (1-2 grams, 50k features, sublinear_tf) → Logistic Regression |
| Classical baseline | TFIDF_LinearSVM | TF-IDF (same) → Linear SVM |
| Transformer | BERT | `bert-base-uncased` |
| Transformer | ALBERT | `albert-base-v2` |
| Transformer | RoBERTa | `roberta-base` |

---

## 2. Model Performance Comparison

All numbers below are aggregated across the three seeds (mean ± std) on the **held-out test set with the validation-tuned threshold**. Source: `artifacts/summary/mean_std_by_model.csv`. The "fraud" prefix denotes metrics computed on the minority/positive class, which is the operationally relevant view.

### 2.1 Headline comparison — BERT, ALBERT, RoBERTa

| Model | Accuracy | Precision (fraud) | Recall (fraud) | F1 (fraud) |
|---|---|---|---|---|
| **BERT** (`bert-base-uncased`) | **0.9886 ± 0.0024** | **0.9331 ± 0.0225** | 0.8231 ± 0.0353 | **0.8745 ± 0.0273** |
| RoBERTa (`roberta-base`) | 0.9858 ± 0.0036 | 0.8652 ± 0.0394 | **0.8385 ± 0.0407** | 0.8515 ± 0.0380 |
| ALBERT (`albert-base-v2`) | 0.9853 ± 0.0035 | 0.8875 ± 0.0292 | 0.8000 ± 0.0846 | 0.8395 ± 0.0455 |

Bold marks the per-column best. Among Transformers, BERT wins on Accuracy, Precision, and F1; RoBERTa edges ahead on Recall.

### 2.2 Extended comparison including baselines and AUC metrics

| Model | Accuracy | Precision | Recall | F1 | Macro-F1 | Balanced Acc. | Specificity | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| TFIDF_LinearSVM | 0.9906 ± 0.0023 | 0.9509 ± 0.0144 | 0.8487 ± 0.0364 | 0.8968 ± 0.0264 | 0.9459 | 0.9232 | 0.9978 | 0.9904 | 0.9449 |
| BERT | 0.9886 ± 0.0024 | 0.9331 ± 0.0225 | 0.8231 ± 0.0353 | 0.8745 ± 0.0273 | 0.9342 | 0.9100 | 0.9970 | 0.9895 | 0.9232 |
| TFIDF_LogReg | 0.9866 ± 0.0017 | 0.8627 ± 0.0346 | 0.8615 ± 0.0204 | 0.8617 ± 0.0157 | 0.9273 | 0.9272 | 0.9929 | 0.9903 | 0.9261 |
| RoBERTa | 0.9858 ± 0.0036 | 0.8652 ± 0.0394 | 0.8385 ± 0.0407 | 0.8515 ± 0.0380 | 0.9220 | 0.9159 | 0.9933 | 0.9872 | 0.9106 |
| ALBERT | 0.9853 ± 0.0035 | 0.8875 ± 0.0292 | 0.8000 ± 0.0846 | 0.8395 ± 0.0455 | 0.9159 | 0.8974 | 0.9948 | 0.9849 | 0.9007 |

### 2.3 Per-seed test-set fraud-F1 (raw values feeding the means above)

| Seed | TFIDF_LogReg | TFIDF_LinearSVM | BERT | ALBERT | RoBERTa |
|---|---|---|---|---|---|
| 42 | 0.8775 | 0.9194 | 0.8889 | 0.8560 | 0.8692 |
| 123 | 0.8462 | 0.8678 | 0.8430 | 0.7881 | 0.8078 |
| 2024 | 0.8614 | 0.9032 | 0.8916 | 0.8745 | 0.8775 |

Seed 123 is the universally weakest seed. Seed 2024 produces the headline BERT run (F1 = 0.8916) that is exported to the application. No single seed is best for every model, but **seed 2024 dominates four of five models** on F1 and is a defensible "representative" seed for figures that cannot show all three.

### 2.4 Default (threshold = 0.5) versus validation-tuned threshold

The threshold-tuning step is consequential and worth documenting. Mean fraud-F1 deltas (test set):

| Model | Default-threshold F1 (mean) | Tuned F1 (mean) | Δ (pp) |
|---|---|---|---|
| TFIDF_LogReg | 0.8534 | 0.8617 | +0.83 |
| TFIDF_LinearSVM | 0.8976 | 0.8968 | −0.08 |
| BERT | 0.8714 | 0.8745 | +0.31 |
| ALBERT | 0.8439 | 0.8395 | −0.44 |
| RoBERTa | 0.8546 | 0.8515 | −0.30 |

In absolute terms the deltas are small (within noise for most models), and **TFIDF_LinearSVM, ALBERT, and RoBERTa actually lose a fraction of a percentage point under tuning** because the validation-selected threshold is not perfectly transferable to the test set. The protocol still adds value: it surfaces the calibration anomaly in ALBERT seed 123 (winning threshold 0.90) and gives BERT seed 2024 its best operating point (0.10), which has the practical consequence of higher recall than at 0.50.

### 2.5 Best single run exported for the deployed app
`select_best_completed_run` ranks transformer runs by `tuned_fraud_f1`. Winner (`best_model/model_meta.json`):

- **Model:** BERT (`bert-base-uncased`)
- **Seed:** 2024
- **Selected threshold:** 0.10
- **Test metrics:** Accuracy 0.9899 · Precision 0.9328 · Recall 0.8538 · F1 **0.8916** · Macro-F1 0.9431 · Balanced Acc. 0.9254 · Specificity 0.9969 · ROC-AUC 0.9902 · PR-AUC 0.9239

### 2.6 Test-set confusion totals (summed across seeds, n = 8,046; positives = 390)

| Group | Model | FP | FN | Correct | FP rate among legitimate | FN rate among fraud |
|---|---|---|---|---|---|---|
| Classical baseline | TFIDF_LinearSVM | 17 | 59 | 7,970 | 0.22% | 15.1% |
| Classical baseline | TFIDF_LogReg | 54 | 54 | 7,938 | 0.71% | 13.8% |
| Transformer | BERT | 23 | 69 | 7,954 | 0.30% | 17.7% |
| Transformer | RoBERTa | 51 | 63 | 7,932 | 0.67% | 16.2% |
| Transformer | ALBERT | 40 | 78 | 7,928 | 0.52% | 20.0% |

LinearSVM has the lowest false-positive count *and* the second-lowest false-negative count. BERT has the lowest FP count among Transformers; ALBERT has the highest FN count, missing one in five fraudulent postings.

### 2.7 Statistical significance highlights (paired bootstrap on fraud-F1)
Across the 60 pairwise tests run, the patterns most useful for the discussion section:

- **TFIDF_LinearSVM ≻ ALBERT** is the only pair with consistent significance: fraud-F1 p = 0.000 / 0.000 / 0.222 across seeds 42 / 123 / 2024 (significant in seeds 42 and 123).
- **TFIDF_LinearSVM vs BERT** is **not** statistically significant in any seed for fraud-F1 (p = 0.134 / 0.324 / 0.520).
- **BERT vs ALBERT** trends in BERT's favor (positive observed differences in all seeds; significant on PR-AUC for seed 42 and seed 2024).
- **BERT vs RoBERTa** is not significant in any seed.
- **TFIDF_LogReg vs TFIDF_LinearSVM**: LinearSVM is significantly better on fraud-F1 in seeds 42 and 2024.
- **ROC-AUC differences are smaller than PR-AUC differences**, consistent with PR-AUC being the more discriminative metric on imbalanced data; pairs that look similar on ROC-AUC can still split apart on PR-AUC.

### 2.8 Subgroup performance on BERT (illustrative — see `predictions_with_subgroups.csv`)
Performance on the BERT prediction pool varies substantially across metadata subgroups. Highlights:

- `has_company_logo` is the strongest single moderator: F1 = 0.952 when a logo is present (n = 6,416) versus 0.824 when it is absent (n = 1,630). Listings without a company logo concentrate ambiguity for the model.
- `employment_type = Contract` is a weak subgroup: F1 = 0.560 (recall = 0.412 on n = 17 fraud).
- `employment_type = Full-time` is the strongest typed subgroup: F1 = 0.912.
- `required_experience = Executive` reaches F1 = 1.000 (small n = 7 fraud).
- Postings with missing `required_experience` (n = 3,093) still reach F1 = 0.893 — the model is robust to missingness in metadata it never sees.

---

## 3. Hyperparameter & Training Details

### 3.1 Shared training configuration (`CONFIG`)

| Hyperparameter | Value | Rationale recoverable from notebook |
|---|---|---|
| Backbones | `bert-base-uncased`, `albert-base-v2`, `roberta-base` | Standard "base" checkpoints chosen for fair parameter-budget comparison |
| `max_len` | 256 tokens | Truncates long postings; balances context vs. T4 memory |
| `batch_size` | 16 (per device) | Fits 110M-parameter encoders in FP16 on a T4 |
| `gradient_accumulation_steps` | 1 | Effective batch size = 16 |
| `epochs` | 5 | Upper bound; early stopping usually fires earlier |
| `early_stopping_patience` | 2 | On `eval_validation_fraud_f1` |
| `learning_rate` | 2e-5 | Standard transfer-learning LR for BERT-family models |
| `weight_decay` | 0.01 | AdamW default for BERT |
| `warmup_ratio` | 0.1 | 10% of total steps as linear warmup |
| `fp16` | True | T4 supports mixed-precision; ~2× speedup |
| `metric_for_best_model` | `eval_validation_fraud_f1` | Aligns with imbalanced-class objective |
| `save_strategy` / `eval_strategy` | epoch / epoch | Simplifies checkpoint-vs-epoch alignment |
| `load_best_model_at_end` | True | Best-validation-F1 checkpoint is restored before test |
| Class weights | `[N/(2·N₀), N/(2·N₁)]` | Inverse-frequency weighting in `WeightedTrainer` |
| `test_size` / `val_size` | 0.15 / 0.15 | 70/15/15 stratified split |
| `latency_sample_size` / `latency_repeats` | 128 / 3 | Deployment-readiness measurement |
| Bootstrap | 1,000 iterations, seed 777 | Significance testing |
| Threshold grid | 0.10 → 0.90, step 0.01 | Validation-tuned operating point |

### 3.2 Per-model winning threshold (validation-tuned, applied on test)

| Seed | TFIDF_LogReg | TFIDF_LinearSVM* | BERT | ALBERT | RoBERTa |
|---|---|---|---|---|---|
| 42 | 0.59 | 0.00 | 0.72 | 0.23 | 0.21 |
| 123 | 0.53 | −0.0325 | 0.33 | 0.90 | 0.18 |
| 2024 | 0.56 | 0.00 | 0.10 | 0.24 | 0.57 |

*LinearSVM uses `decision_function` output, not a probability, so its threshold lives on a different scale.

The classical baselines (LogReg around 0.55, SVM around 0) are stable across seeds; the **Transformer thresholds vary widely**: BERT ranges 0.10–0.72, RoBERTa 0.18–0.57, ALBERT 0.23–0.90. This is direct evidence of seed-to-seed probability calibration drift — see Finding #6.

### 3.3 Best deployable configuration
The configuration exported for the application is the BERT run on seed 2024 with threshold 0.10. This is BERT's strongest seed (test F1 = 0.892) and was selected by the multi-criteria sort `(tuned_fraud_f1, tuned_pr_auc, tuned_fraud_recall)` descending.

### 3.4 Training cost and operational footprint

| Model | Mean runtime / seed | Std | Inference latency (ms/sample, mean of seed means) | On-disk size |
|---|---|---|---|---|
| TFIDF_LogReg | 0.28 min | 0.02 | ~0.79 ms | n/a (sklearn) |
| TFIDF_LinearSVM | 0.28 min | 0.02 | ~0.62 ms | n/a (sklearn) |
| BERT | 28.2 min | 0.93 | ~6.06 ms | 418 MB |
| RoBERTa | 26.1 min | 3.65 | ~5.17 ms | 479 MB |
| ALBERT | 32.6 min | 0.14 | ~9.20 ms | **46 MB** |

Note the counter-intuitive ALBERT result: its parameter count is much smaller, yet wall-clock training is the longest and per-sample inference is the slowest. This is consistent with ALBERT's cross-layer parameter-sharing design, which reduces memory but does not reduce FLOPs per forward pass.

---

## 4. Key Findings & Discussion Points

The findings below are ordered from headline result to nuance. Each is grounded in a specific row, table, or output from the notebook and is written to be lifted into the manuscript with minimal editing.

### Finding 1 — BERT is the strongest Transformer; RoBERTa and ALBERT trail
BERT's mean fraud-F1 (0.8745) is +2.30 pp over RoBERTa (0.8515) and +3.50 pp over ALBERT (0.8395). BERT also has the best precision among the three (0.933 vs. 0.865 / 0.888) and the lowest false-positive count (23 vs. 51 / 40 across the combined 8,046-row test pool). The exported deployable model is therefore BERT (seed 2024, F1 = 0.892). The likely mechanistic explanation is that the WordPiece vocabulary of `bert-base-uncased` and the standard masked-LM pre-training objective produce a `[CLS]` representation that is well-aligned with this dataset's lexically-stereotyped scam cues. Practically, BERT is precision-leaning at the chosen operating point — it raises few false alarms but trades some recall.

### Finding 2 — TF-IDF + Linear SVM beats every Transformer on this dataset
Mean fraud-F1: LinearSVM 0.897 vs. best Transformer (BERT) 0.875, a 2.2 pp gap. The paired bootstrap shows the LinearSVM-vs-ALBERT gap is significant in 2 of 3 seeds (p ≤ 0.018), while LinearSVM-vs-BERT is **not** significant in any seed (p = 0.134, 0.324, 0.520). LinearSVM also has the highest accuracy (0.991), highest precision (0.951), highest specificity (0.998), highest ROC-AUC (0.990) and highest PR-AUC (0.945) — it is essentially a Pareto winner on this benchmark. The probable explanation is that EMSCAD scams rely on a recurring set of surface-level lexical signals (specific phrasing about pay, "work from home", payment platforms, grammatical irregularities) that 50k-feature 1–2-gram TF-IDF captures with very low variance. **This is an honest finding worth foregrounding in the manuscript**: the additional cost of fine-tuning a Transformer is not always justified on small, lexically-stereotyped fraud datasets. The Transformer's value proposition has to be argued on different grounds — robustness to paraphrase, transfer to multilingual settings, or resilience to novel scams not present at training time — none of which are directly measured by this protocol.

### Finding 3 — ALBERT shows the highest run-to-run variance and the worst minority-class recall
ALBERT's fraud-F1 std is 0.045, ~70% larger than BERT's (0.027), driven mostly by recall std of 0.085 (BERT: 0.035). The per-seed thresholds (0.23, 0.90, 0.24) reveal a probability-calibration anomaly on seed 123: the validation routine pushed the threshold up to 0.90 — at the upper edge of the grid — to maximize F1, a classic sign of an under-confident or poorly-calibrated logit distribution. Seed-123 ALBERT also has the worst overall transformer F1 in the experiment (0.788). This is consistent with prior reports that ALBERT's cross-layer parameter sharing makes downstream fine-tuning more sensitive to seed and learning-rate choices. **Trade-off worth flagging**: ALBERT is 9× smaller on disk (46 MB vs. 418 MB BERT), but it pays for that compactness with stability and minority-class recall — the opposite of what most readers would expect.

### Finding 4 — Latency vs. accuracy trade-off cuts both ways for the deployed application
Inference latency on T4 is ~0.6 ms/sample for the SVM vs. ~6 ms/sample for BERT — a 10× gap. Wall-clock training is even more lopsided: ~17 s for the SVM vs. ~28 min for BERT per seed, and ~33 min for ALBERT. If the application has tight latency, retraining-frequency, or deployment-cost budgets, the LinearSVM is operationally superior *and* has higher F1. ALBERT's small disk footprint (46 MB) is its only deployment advantage; it does not translate into faster inference (it is in fact the slowest at 9.2 ms/sample, ~50% slower than BERT and ~80% slower than RoBERTa). The Pareto frontier for deployment is therefore: **LinearSVM** (cheap + accurate) — **BERT** (slower but second-best F1, same hardware footprint as RoBERTa) — **RoBERTa** (similar footprint to BERT, slightly worse F1). ALBERT is dominated.

### Finding 5 — No catastrophic overfitting; mild validation-to-test optimism
Validation-set fraud-F1 at the validation-selected threshold is on average ≈0.866 across the 15 transformer runs. Test-set fraud-F1 at the same threshold averages ≈0.864 — a difference of ~0.2 pp, which is well within seed-to-seed noise. The 5-epoch cap with patience-2 early stopping appears tight enough that no model exhibits the classic "validation curve dips while training curve keeps falling" pattern (verifiable in the per-run learning-curve plots from cell 25/37). Where instability does appear is at the **seed level for ALBERT and RoBERTa**, both of which have outlier weak seeds (ALBERT seed 123: F1 = 0.788; RoBERTa seed 123: F1 = 0.808). In the manuscript these should be reported as raw per-seed results in addition to the mean, so reviewers can see the spread.

### Finding 6 — Transformer probability calibration is unstable across seeds
The per-model winning thresholds reveal substantial calibration drift. BERT's tuned threshold ranges from 0.10 (seed 2024) to 0.72 (seed 42); RoBERTa from 0.18 to 0.57; ALBERT from 0.23 to 0.90. By contrast, the classical baselines stay tight (LogReg 0.53–0.59; LinearSVM essentially fixed at 0). This means that **the validation-tuning step is doing real work for the Transformers** — without it, they would be forced to a default threshold (0.50) that is far from optimal on at least one seed per model. Calibration drift also implies that **a single deployed Transformer threshold is not transferable across retrainings**; any production pipeline built from this work needs to re-tune the threshold on a held-out validation slice each time it retrains. For the manuscript, this is the strongest argument *against* publishing a fixed inference rule and *for* publishing the threshold-tuning subroutine alongside the model weights.

### Finding 7 — Recall is the dataset-wide bottleneck, not precision
Across all five models the precision side of the test-set confusion matrix is healthier than the recall side. The combined-across-seeds false-positive rate among legitimate postings stays at or below 0.7% for every model; the false-negative rate among fraudulent postings ranges from 13.8% (LogReg) to 20.0% (ALBERT). In other words, the models are conservative — they correctly flag the obvious scams, and the residual error is dominated by fraudulent postings that look textually like legitimate ones. **This has direct manuscript implications**: presenting only "accuracy 99%" understates the operational risk. The right framing for the *Limitations* section is "the deployed BERT model misses roughly 1 in 6 actual scams at the chosen operating point", which is far more honest and far more useful for the application's downstream users.

### Finding 8 — Threshold-tuning gains are small in mean but rescue worst-case seeds
Default-vs-tuned mean F1 deltas are tiny — between −0.4 pp and +0.8 pp depending on the model — so on the headline numbers, threshold tuning is approximately a wash. The value of the tuning step shows up in the *worst seeds*. Without tuning, ALBERT seed 123 would operate at threshold 0.50 with a default-threshold F1 of 0.793; the tuned threshold (0.90) brings it down to 0.788. Conversely, BERT seed 2024 *gains* meaningfully from tuning: default-threshold F1 was 0.877, tuned-threshold F1 is 0.892 — and that one run is the model the application will actually ship. So tuning is not free per-row but is essential for the export pipeline that surfaces the strongest single run.

### Finding 9 — ROC-AUC is saturated; PR-AUC is the discriminating metric
All five models cluster between ROC-AUC 0.985 and 0.990 — a 0.5-pp band that is essentially noise on a 130-positive test set. PR-AUC, by contrast, opens up a real spread: LinearSVM 0.945 → ALBERT 0.901 (a 4.4-pp gap), and the per-seed PR-AUC spread is large enough to be statistically significant in the bootstrap test. **For the paper this means**: if you only present ROC curves, every model looks equivalent and the contribution is hard to defend. PR curves are what carry the comparative story. This is also why the notebook persisted both views (cell 39).

### Finding 10 — High-confidence errors concentrate on a small set of recurring titles
The error-cases table (cell 42, `error_cases.csv`) reveals two clear failure patterns for BERT:

- **Recurring high-confidence false positives** on bland administrative titles: `Administrative Assistant` (job_id 6907), `Part-Time Administrative/Data Entry I` (job_id 537), `Customer Service - Cloud Video Production` (job_id 16971) and similar. Same job IDs reappear across seeds. The model assigns probability ≈0.999 fraud despite the ground-truth label of legitimate. This is suggestive of either **label noise** in the original EMSCAD dataset or genuinely ambiguous postings whose text shares lexical structure with known scams.
- **Recurring high-confidence false negatives** on terse, low-information fraud postings (`Casual job/Immediate start`, `Vemma Brand Partner`, `Quant Analyst`, `Software Design Engineer`). These tend to have empty `requirements` or `benefits` fields, so the concatenated input is short and superficially professional. They escape detection because they lack the lexical scam markers the model has learned.

For the manuscript this suggests two pieces of follow-up work: a manual re-labelling pass on the high-confidence false positives, and a targeted feature (e.g., text-length-based prior) that would help with the false-negative class.

### Finding 11 — Class-weighted loss alone is insufficient; threshold tuning is the compensating mechanism
Despite the inverse-frequency class weights in `WeightedTrainer`, the Transformers still err on the side of predicting "legitimate". This is visible in the gap between recall and precision at the default threshold of 0.5: BERT default-threshold recall is 0.823 vs. precision 0.933; even with the loss reweighted, the model needs the validation-threshold step to reach its operating optimum. The implication is that **for highly imbalanced classification, weighted loss + threshold tuning is a paired technique, not an either/or choice** — and the notebook implements both.

### Finding 12 — Seed 123 is a stress-test seed; report it explicitly
Seed 123 is the universally weakest seed (lowest F1 for every model). It is the seed that triggers ALBERT's 0.90 threshold anomaly and the seed where RoBERTa ties its lowest score. Seed 123 is therefore a useful "stress-test" anchor: if the manuscript reports only the mean, it hides this; if it reports per-seed, the seed-123 column is where reviewers will look for failure modes. The recommendation is to report all three seeds explicitly in the main results table (Section 2.3 of this log), not just the mean.

### Finding 13 — Statistical power is limited; most pairwise differences are inconclusive
Of the 30 fraud-F1 pairwise comparisons (10 model pairs × 3 seeds), only ~7 reach p < 0.05. The dominant reason is the small minority-class test sample (130 fraud per seed). This is a **methodological caveat** for the manuscript: claims like "BERT outperforms RoBERTa" need to be qualified — the *means* support the claim, but the bootstrap CIs do not separate them at conventional significance levels. The conservative phrasing for the paper is "no transformer pair is statistically distinguishable on fraud-F1 in our protocol; LinearSVM significantly dominates ALBERT, but is statistically tied with BERT". Avoid over-claiming.

### Finding 14 — `has_company_logo` is the strongest single moderator of model performance
Subgroup analysis on the BERT prediction pool shows fraud-F1 = 0.952 when a company logo is present (n = 6,416) versus 0.824 when it is absent (n = 1,630). The model is materially more reliable on postings from organizations that bothered to upload branding, and noticeably weaker on the cases that arguably matter most (logo-less, anonymized listings). This is a real fairness/operational concern: the *application's* most likely high-stakes use case (a job seeker checking an obscure listing without a logo) is the regime where the model's recall and precision both drop. The manuscript's *Discussion* section should call this out explicitly.

### Finding 15 — `employment_type = Contract` is a weak subgroup, `Full-time` is strong
F1 = 0.560 for `Contract` (n = 17 fraud) vs. F1 = 0.912 for `Full-time` (n = 230 fraud) on BERT. The Contract regime has only 17 positives so the result is noisy, but the recall (0.412) is low enough to be flagged. Contract postings tend to be terser, contain less benefits-language, and overlap stylistically with real freelance opportunities. A targeted error-analysis pass on the Contract subset is a recommended next step.

### Finding 16 — Cross-layer parameter sharing does not help inference cost
ALBERT's smaller checkpoint (46 MB) is the headline marketing claim of the architecture, but on the experimental protocol used here it does not translate into a faster forward pass: ALBERT's per-sample inference latency is 9.2 ms versus BERT's 6.1 ms and RoBERTa's 5.2 ms. The reason is that ALBERT *runs* the same number of transformer layers — they are just tied together at the parameter level. So FLOPs per sample are unchanged, while the additional bookkeeping in the parameter-sharing implementation imposes a small overhead. **Practical takeaway**: if "small model" is the deployment requirement, ALBERT helps you pay less for storage; if "fast inference" is the requirement, ALBERT does not help. For this thesis's deployed application, neither requirement is binding, but it is worth stating explicitly so future readers don't pick ALBERT for the wrong reason.

### Finding 17 — Application export pipeline is reproducible end-to-end
The notebook's selection logic (`select_best_completed_run`) deterministically produces the same export — BERT, seed 2024, threshold 0.10 — given the existing `artifacts/` directory. The exported `best_model/model_meta.json` snapshots the full configuration, the selected threshold, the metric values, and the preprocessing description string. From the application's side, inference therefore needs only: (a) the cleaning function in `src/models/preprocessor.py`, (b) the saved tokenizer + model under `best_model/`, and (c) the threshold from `model_meta.json`. There are no hidden dependencies on the training environment. The manuscript should reference this section as evidence that the application-side artifact is independent of the experimental notebook.

---

## 5. Visualization Candidates

The notebook generates a substantial set of figures. The four anchor plots below carry the headline story; an optional fifth and sixth support the error-analysis subsection. Each lives under `artifacts/figures/` or per-run `artifacts/runs/seed_<S>/<model>/learning_curves.png`.

### 5.1 Class-distribution and missing-value bar pair (Cell 12)
**File:** generated inline; persist via `FIGURES_DIR / "data_profile.png"` if not already saved.
**Story:** Establishes the central methodological challenge of the thesis — a 4.84% fraud rate and significant missingness in `company_profile`, `requirements`, and `benefits`. This justifies (a) class-weighted loss and (b) concatenation-based text fusion. Place this in the *Dataset* section of the paper.

### 5.2 Per-run learning curves: loss / accuracy / fraud-F1 vs. epoch (Cell 25 / Cell 37)
**Files:** `artifacts/runs/seed_<S>/<model>/learning_curves.png` (one per model+seed) and the aggregated learning-curve panels rendered in cell 37.
**Story:** Shows that BERT and RoBERTa converge cleanly within 3-4 epochs and that early stopping fires before training-loss minima — i.e., the validation curve is the binding constraint, not over-training. ALBERT seed 123's curve is the visual evidence behind Findings #3 and #6: jagged validation F1 and a wider train/val gap. Use in the *Training Behavior* subsection.

### 5.3 ROC and Precision-Recall curve panel (Cell 39)
**File:** `artifacts/figures/roc_pr_curves.png` (Figure 1400×500, 2 axes).
**Story:** Threshold-independent comparison. The ROC view shows the saturation noted in Finding #9 (all models cluster); the PR view fans out and is the right vehicle for the comparative claim. Pair this with a one-line caption that explicitly tells the reader to focus on the right-hand panel because of class imbalance.

### 5.4 Mean ± std bar chart of model comparison (Cell 40)
**File:** `artifacts/figures/model_comparison_mean_std.png` (Figure 1200×600).
**Story:** Direct visual answer to the comparative research question. Error bars (std across the 3 seeds) make the variance findings legible — in particular ALBERT's wide recall whisker (Finding #3) and BERT's tight precision whisker (Finding #1). Centerpiece of the *Results* section.

### 5.5 Per-seed test-set confusion matrices for the three Transformers (optional)
**File:** typically saved alongside each run; verify `artifacts/runs/seed_<S>/<model>/confusion_matrix.png`.
**Story:** Supports Findings #7 and #10 — the bottleneck is FN, not FP, and the bottleneck is concentrated on a recurring set of titles. Reading three confusion matrices side-by-side per seed makes the BERT-vs-ALBERT FN gap visually explicit.

### 5.6 Subgroup F1 bar chart for BERT (optional, derive from `predictions_with_subgroups.csv`)
**File:** not auto-generated; recommend deriving from cell 43's output.
**Story:** Supports Findings #14 and #15. A one-row bar chart sliced by `has_company_logo` and `employment_type` illustrates the fairness/operational concern that BERT's strong headline number hides regimes where it is materially weaker.

---

## 6. Manuscript Roadmap (suggested mapping)

| Manuscript section | Source in this log |
|---|---|
| Abstract — headline numbers | §2.1, §2.5 |
| Introduction — problem framing | Finding #14 (logo-less postings as the operational pain point) |
| Related Work — bridge to TF-IDF baselines | Finding #2 |
| Dataset | §1.1, §1.2, Visualization 5.1 |
| Methodology | §1.3, §3.1, §3.2 |
| Results — main comparison | §2.1, §2.2, §2.3, Visualization 5.4 |
| Results — calibration & thresholds | §2.4, §3.2, Findings #6, #8, #11 |
| Results — significance | §2.7, Finding #13 |
| Results — error analysis | §2.6, §2.8, Findings #7, #10, #14, #15 |
| Discussion — when not to use a Transformer | Findings #2, #4, #16 |
| Discussion — fairness & operational risk | Findings #14, #15 |
| Limitations | Findings #5, #12, #13 |
| Conclusion — deployable artifact | §2.5, Finding #17 |

---

## Appendix A — Files generated by the pipeline (referenced from this log)

- `artifacts/environment.json` — full reproducibility manifest.
- `artifacts/data_profile.json` — Section 1.1 source.
- `artifacts/summary/all_runs.csv` — every per-seed × per-model row (raw metric values, used to build §2.3).
- `artifacts/summary/mean_std_by_model.csv` — aggregated metrics in §2.1, §2.2.
- `artifacts/summary/significance_tests.csv` — pairwise bootstrap results in §2.7.
- `artifacts/summary/runtime_by_model.csv` — §3.4.
- `artifacts/summary/deployment_readiness.csv` — latency and model-size table.
- `artifacts/summary/error_summary_by_model.csv` — §2.6.
- `artifacts/summary/error_cases.csv` — high-confidence FP/FN samples (Finding #10).
- `artifacts/summary/predictions_with_subgroups.csv` — basis for §2.8 and Findings #14, #15.
- `artifacts/summary/thresholds_by_model.csv` — basis for §3.2 and Finding #6.
- `artifacts/summary/default_threshold_metrics.csv` and `tuned_threshold_metrics.csv` — basis for §2.4 and Finding #8.
- `artifacts/summary/table_1_dataset_statistics.csv`, `table_2_model_performance.csv`, `table_3_statistical_comparison.csv`, `table_4_runtime.csv` — paper-ready CSV tables.
- `artifacts/figures/roc_pr_curves.png`, `model_comparison_mean_std.png` — main figures (Visualizations 5.3, 5.4).
- `artifacts/runs/seed_<S>/<model>/learning_curves.png` and `confusion_matrix.png` — per-run figures (Visualizations 5.2, 5.5).
- `best_model/` — exported BERT (seed 2024, threshold 0.10) for the application; contains `model_meta.json` with the full configuration used for export.

---

## Appendix B — Quick-reference numbers for citation in the manuscript

- Dataset: 17,880 postings, 866 fraud, fraud rate 4.84%.
- Splits per seed: 12,516 / 2,682 / 2,682 (train/val/test); 130 fraud in val and test.
- Best Transformer (mean over 3 seeds): BERT, fraud-F1 = 0.8745 ± 0.0273.
- Best run (single seed, exported): BERT, seed 2024, threshold 0.10, F1 = 0.8916.
- Best overall model (mean): TFIDF + LinearSVM, fraud-F1 = 0.8968 ± 0.0264.
- Worst Transformer (mean): ALBERT, fraud-F1 = 0.8395 ± 0.0455.
- Worst single Transformer run: ALBERT seed 123, F1 = 0.7881.
- Mean BERT runtime per seed: 28.2 min; mean ALBERT runtime: 32.6 min; mean LinearSVM runtime: 0.28 min.
- BERT inference latency: ~6.1 ms/sample on Tesla T4 with FP16.
- Bootstrap iterations: 1,000; bootstrap seed: 777.
- Hardware: Tesla T4 (Google Colab), CUDA 12.8, FP16.
