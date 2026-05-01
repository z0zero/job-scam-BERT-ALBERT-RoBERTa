# Research Pipeline Application Paper Improvements Design

## Overview

This design extends `research_pipeline.ipynb` for an application-oriented paper on job scam detection. The current notebook already supports a paper-style Transformer comparison across BERT, ALBERT, and RoBERTa with stratified splits, weighted loss, validation-based threshold tuning, multi-seed aggregation, statistical testing, plots, error analysis, subgroup analysis, and best-model export.

The improvement keeps that foundation and adds two lightweight classical text-classification baselines, early stopping for Transformer training, and application-readiness outputs. The target execution environment is Google Colab Free with a T4 GPU and a runtime budget of roughly 4 hours 40 minutes per session, so the workflow must remain resumable and practical.

## Goals

- Compare Transformer models against two classical TF-IDF baselines under the same split and metric protocol.
- Improve Transformer training efficiency with validation-based early stopping.
- Strengthen application-paper reporting with runtime, inference latency, model size, selected threshold, and deployment-readiness outputs.
- Preserve the current Streamlit app compatibility by exporting only the best Transformer model to `best_model/`.
- Keep the workflow resumable for Colab Free T4 sessions.

## Non-Goals

- Do not add more classical baselines beyond Logistic Regression and Linear SVM.
- Do not add new Transformer families such as DistilBERT, DeBERTa, or ELECTRA.
- Do not add expensive ablation studies such as max sequence length comparison or no-class-weight experiments.
- Do not add formal XAI methods such as SHAP or LIME in this iteration.
- Do not rewrite the Streamlit application unless metadata compatibility requires a small adjustment.

## Model Scope

The final comparison includes five models:

| Group | Model Label | Implementation |
| --- | --- | --- |
| Classical baseline | TF-IDF + Logistic Regression | `TfidfVectorizer` plus `LogisticRegression` |
| Classical baseline | TF-IDF + Linear SVM | `TfidfVectorizer` plus `LinearSVC` |
| Transformer | BERT | `bert-base-uncased` |
| Transformer | ALBERT | `albert-base-v2` |
| Transformer | RoBERTa | `roberta-base` |

All models must use the same train, validation, and test split for a given seed. This keeps model comparisons fair and supports paired statistical comparison.

## Classical Baseline Design

Add a baseline registry separate from the Transformer registry:

```python
BASELINE_REGISTRY = {
    "TFIDF_LogReg": "tfidf_logistic_regression",
    "TFIDF_LinearSVM": "tfidf_linear_svm",
}
```

Recommended TF-IDF configuration:

```python
TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000,
    min_df=2,
    sublinear_tf=True,
)
```

Recommended Logistic Regression configuration:

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    solver="liblinear",
)
```

Recommended Linear SVM configuration:

```python
LinearSVC(
    class_weight="balanced",
)
```

Logistic Regression should use `predict_proba` as its fraud score. Linear SVM should use `decision_function` as its fraud score. Both scores must be passed through the same threshold-selection and metric helpers used by the Transformer runs.

## Transformer Training Design

Transformer training remains based on `WeightedTrainer`, class-weighted cross entropy, and HuggingFace `TrainingArguments`.

Update the training configuration:

```python
CONFIG = {
    "epochs": 5,
    "early_stopping_patience": 2,
    # keep the existing batch size, learning rate, weight decay, warmup,
    # split, threshold grid, and bootstrap settings unless implementation
    # testing shows a concrete reason to change them
}
```

Add `EarlyStoppingCallback` to the Trainer:

```python
callbacks=[
    EarlyStoppingCallback(
        early_stopping_patience=CONFIG["early_stopping_patience"],
        early_stopping_threshold=0.0,
    )
]
```

Keep these existing selection settings:

```python
load_best_model_at_end=True
metric_for_best_model="eval_validation_fraud_f1"
greater_is_better=True
```

This means `epochs = 5` is a maximum, not a promise that all runs will use five full epochs.

## Evaluation Protocol

Use the existing repeated stratified holdout protocol:

- Split per seed: 70 percent train, 15 percent validation, 15 percent test.
- Seeds: `[42, 123, 2024]` for final paper runs.
- Debug mode may use only `SINGLE_SEED = 42`.
- Validation set is used for threshold selection, Transformer checkpoint selection, and early stopping.
- Test set is used only for final evaluation.

Primary metric:

- `fraud_f1`

Supporting metrics:

- `macro_f1`
- `fraud_precision`
- `fraud_recall`
- `balanced_accuracy`
- `specificity`
- `pr_auc`
- `roc_auc`
- confusion matrix counts

Overall research ranking should use:

1. highest mean `tuned_fraud_f1`,
2. if tied, highest mean `tuned_pr_auc`,
3. if still tied, lower inference latency or runtime.

Application export selection should apply the same rule only within the Transformer group. Classical baselines are included in the paper comparison, but they are not exported to `best_model/` in this iteration.

## Runtime Strategy

The notebook must remain usable on Google Colab Free with a T4 GPU and limited runtime.

Execution order:

1. Run classical baselines first because they are CPU-friendly and fast.
2. Run Transformer experiments using the existing resumable artifact pattern.
3. Aggregate all completed runs after each session.

Recommended modes:

```python
RUN_MODE = "single_seed"      # debugging
RUN_MODE = "full_multi_seed"  # final paper experiment
```

`FORCE_RETRAIN = False` should remain the default so completed runs are skipped. If a Colab session disconnects, the next session can continue from the remaining model/seed combinations.

## Artifact Design

Each run, including classical baselines, should save:

```text
config.json
runtime.json
metrics_default_threshold.json
metrics_tuned_threshold.json
predictions.csv
```

Transformer runs should additionally save:

```text
training_history.csv
learning_curves.png
model/
```

Classical baseline runs may save model artifacts under their run folder if useful for inspection, but they are not exported to `best_model/` in this iteration.

## Application Reporting Outputs

Add paper-ready outputs under `artifacts/summary/`:

```text
all_runs.csv
mean_std_by_model.csv
default_threshold_metrics.csv
tuned_threshold_metrics.csv
runtime_by_model.csv
deployment_readiness.csv
thresholds_by_model.csv
error_summary_by_model.csv
significance_tests.csv
```

`deployment_readiness.csv` should include:

- model label
- model group
- selected threshold
- tuned fraud F1
- PR-AUC
- ROC-AUC
- training runtime
- inference latency mean
- inference latency standard deviation
- model size if available
- export eligibility

The notebook should also continue producing ROC/PR plots, learning curves for Transformer runs, model comparison plots, error cases, and subgroup metrics.

## Best Model Export

For this iteration, classical baselines are scientific comparators only. The deployed app model must still come from the Transformer group because the existing Streamlit app expects a HuggingFace sequence-classification model in `best_model/`.

`export_best_model()` should select only among Transformer runs. It should write expanded metadata:

```json
{
  "model_name": "ALBERT",
  "hf_model_id": "albert-base-v2",
  "selected_threshold": 0.5,
  "selection_metric": "tuned_fraud_f1",
  "selected_seed": 42,
  "metrics": {
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "macro_f1": 0.0,
    "balanced_accuracy": 0.0,
    "specificity": 0.0,
    "pr_auc": 0.0,
    "roc_auc": 0.0
  },
  "config": {},
  "preprocessing": {
    "text_columns": ["title", "company_profile", "description", "requirements", "benefits"],
    "cleaning": "html_unescape_strip_html_remove_urls_lowercase_normalize_whitespace"
  },
  "artifact_path": "",
  "exported_at": ""
}
```

The exact metric values are populated from the selected completed run.

## Paper Positioning

The intended paper claim should be application-oriented:

> This study compares Transformer-based job scam detection models against classical TF-IDF baselines under a consistent stratified evaluation protocol, then selects a deployable Transformer model for a real-time scam screening application using validation-tuned fraud F1 and deployment-oriented metrics.

This avoids overclaiming novelty in model architecture and focuses the contribution on a practical, evaluated application pipeline.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Colab runtime ends before all Transformer runs finish | Keep `FORCE_RETRAIN=False` and save artifacts per run |
| Classical SVM lacks calibrated probabilities | Use `decision_function` scores for threshold tuning, PR-AUC, and ROC-AUC |
| Baseline wins on test F1 but cannot be exported to current app | Report it honestly as a comparator; export still limited to best Transformer for app compatibility |
| Test leakage through threshold tuning | Select threshold only on validation, then evaluate once on test |
| Accuracy looks high because dataset is imbalanced | Make `fraud_f1`, `PR-AUC`, recall, and balanced accuracy central in paper tables |

## Acceptance Criteria

- Notebook includes both classical baselines and the three Transformer models.
- Classical baselines use the same split, threshold tuning, metric helpers, and artifact format as Transformer models.
- Transformer training uses maximum five epochs with early stopping patience two.
- Aggregated paper tables include all five models.
- Deployment-readiness table includes latency and model-size information where available.
- `best_model/` export remains restricted to the best Transformer run.
- The notebook remains resumable in Colab through per-run artifacts and `FORCE_RETRAIN=False`.
