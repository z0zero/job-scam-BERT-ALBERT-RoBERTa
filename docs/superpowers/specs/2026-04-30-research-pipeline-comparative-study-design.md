# Research Pipeline Comparative Study Design

## Overview

This design upgrades `research_pipeline.ipynb` from a single-run model comparison into a paper-oriented comparative study of BERT, ALBERT, and RoBERTa for fake job posting detection. The core research question is whether one pre-trained transformer model performs better and more consistently than the others under repeated stratified evaluation on EMSCAD.

The implementation remains focused on the existing notebook and keeps the Streamlit app compatible with the exported `best_model/` directory.

## Goals

- Compare `bert-base-uncased`, `albert-base-v2`, and `roberta-base` with stronger experimental rigor.
- Support reruns on free Google Colab with a T4 GPU by making every model/seed run resumable.
- Report mean and standard deviation across repeated stratified holdout runs.
- Include paper-ready metrics, plots, statistical comparisons, runtime logs, and error analysis.
- Preserve existing preprocessing behavior and exported model metadata expected by the app.

## Non-Goals

- Add new model families beyond BERT, ALBERT, and RoBERTa.
- Add TF-IDF, SVM, or logistic regression baselines in this iteration.
- Add SHAP, LIME, or other XAI methods in this iteration.
- Change the Streamlit MVC app unless notebook export compatibility requires it.
- Introduce new dependencies unless they are strictly required for the approved evaluation workflow.

## Experimental Protocol

Use repeated stratified holdout rather than k-fold cross-validation. This gives stronger evidence than a single split while staying practical on free Colab runtimes.

Default protocol:

- Dataset: EMSCAD `fake_job_postings.csv`.
- Models:
  - BERT: `bert-base-uncased`
  - ALBERT: `albert-base-v2`
  - RoBERTa: `roberta-base`
- Seeds: `[42, 123, 2024]`
- Split per seed: stratified `70% train / 15% validation / 15% test`
- All models use the same split for a given seed.
- Training seed is set per run for NumPy, PyTorch, CUDA, and Trainer.
- Primary reported result: mean +/- std across the three seeds.

The main paper claim should be framed as model consistency across repeated stratified splits, not as a single test-set win.

## Colab Runtime Strategy

The notebook should assume free Google Colab GPU sessions can disconnect or change availability. It must be safe to run one model/seed combination at a time and aggregate results later.

Default training settings:

```python
per_device_train_batch_size = 16
gradient_accumulation_steps = 1
effective_batch_size = 16
fp16 = torch.cuda.is_available()
save_total_limit = 1
```

Fallback setting for CUDA out-of-memory only:

```python
per_device_train_batch_size = 8
gradient_accumulation_steps = 2
effective_batch_size = 16
```

The fallback preserves effective batch size and must be recorded in each run's config. It should not be used silently.

Notebook controls:

```python
RUN_MODE = "single_seed"  # or "full_multi_seed"
EXPERIMENT_SEEDS = [42, 123, 2024]
MODELS_TO_RUN = ["BERT", "ALBERT", "RoBERTa"]
FORCE_RETRAIN = False
EVALUATE_TRAIN_EACH_EPOCH = True
```

`FORCE_RETRAIN=False` means an existing completed artifact skips that run, making the notebook resumable.

## Notebook Structure

The revised notebook should be organized into these sections:

1. Setup and configuration
2. Environment and reproducibility log
3. Data loading and provenance
4. EDA and data profile export
5. Text preprocessing
6. Repeated stratified split generation
7. Dataset class
8. Metrics helpers
9. Weighted Trainer and callbacks
10. Training/evaluation runner
11. Per-run artifact persistence
12. Aggregate evaluation
13. Statistical testing
14. Learning curve visualization
15. ROC and PR curve visualization
16. Error analysis and subgroup analysis
17. Paper tables and figures
18. Best model export
19. Colab Drive backup/export

The notebook may remain one file, but repeated logic should move into functions so runs are reproducible and easier to audit.

## Data Flow

1. Load EMSCAD from `fake_job_postings.csv`.
2. Preserve raw columns needed for subgroup and error analysis.
3. Combine text fields:
   - `title`
   - `company_profile`
   - `description`
   - `requirements`
   - `benefits`
4. Clean text with the same behavior as `src/models/preprocessor.py`.
5. For each seed, generate stratified train/validation/test splits.
6. For each model in each seed, train with weighted cross-entropy.
7. Save predictions, probabilities, metrics, training history, runtime, and config immediately after each run.
8. Aggregate all completed runs into summary tables and paper-ready plots.
9. Export the best model to `best_model/` for app compatibility.

## Reproducibility Artifacts

Create an `artifacts/` directory with machine-readable outputs.

Environment:

```text
artifacts/environment.json
artifacts/data_profile.json
```

Per-run artifacts:

```text
artifacts/runs/seed_42/albert/config.json
artifacts/runs/seed_42/albert/metrics_default_threshold.json
artifacts/runs/seed_42/albert/metrics_tuned_threshold.json
artifacts/runs/seed_42/albert/predictions.csv
artifacts/runs/seed_42/albert/confusion_matrix.json
artifacts/runs/seed_42/albert/training_history.csv
artifacts/runs/seed_42/albert/learning_curves.png
artifacts/runs/seed_42/albert/runtime.json
```

Summary artifacts:

```text
artifacts/summary/all_runs.csv
artifacts/summary/mean_std_by_model.csv
artifacts/summary/default_threshold_metrics.csv
artifacts/summary/tuned_threshold_metrics.csv
artifacts/summary/significance_tests.csv
artifacts/summary/runtime_by_model.csv
artifacts/summary/subgroup_metrics.csv
artifacts/summary/error_cases.csv
```

Figures:

```text
artifacts/figures/model_comparison_mean_std.png
artifacts/figures/roc_curves.png
artifacts/figures/pr_curves.png
artifacts/figures/learning_curves_loss_mean_std.png
artifacts/figures/learning_curves_accuracy_mean_std.png
artifacts/figures/learning_curves_f1_mean_std.png
```

## Metrics

Primary metric:

- Fraud-class F1

Secondary metrics:

- Fraud-class precision
- Fraud-class recall
- Macro-F1
- Accuracy
- Balanced accuracy
- Specificity
- ROC-AUC
- PR-AUC
- Confusion matrix

PR-AUC is required because EMSCAD is heavily imbalanced. ROC-AUC should still be reported, but it should not be the only ranking metric.

## Threshold Tuning

Evaluate each trained model under two decision rules:

1. Default threshold: fraud probability >= `0.50`
2. Tuned threshold: selected on the validation set to maximize fraud-class F1

Threshold candidates:

```python
np.arange(0.10, 0.91, 0.01)
```

The tuned threshold must be chosen only from validation predictions, then applied once to the test predictions. Test labels must not influence threshold selection.

## Training History and Learning Curves

Learning curves are required paper artifacts.

For each model/seed, save:

- Training loss per epoch
- Validation loss per epoch
- Training accuracy per epoch
- Validation accuracy per epoch
- Training F1 per epoch
- Validation F1 per epoch

`EVALUATE_TRAIN_EACH_EPOCH = True` is enabled by default. This increases runtime, but it makes the comparative study more complete for publication.

Aggregate plots:

- Train vs validation loss, mean +/- std across seeds
- Train vs validation accuracy, mean +/- std across seeds
- Train vs validation F1, mean +/- std across seeds

## Runtime Logging

Each run must record:

- Seed
- Model label
- HuggingFace model id
- Start time
- End time
- Total training seconds and minutes
- GPU name
- CUDA availability
- Batch size
- Gradient accumulation steps
- Effective batch size
- FP16 status
- Number of epochs
- Max sequence length

Runtime summaries should be included in the paper discussion because ALBERT is expected to be lighter than BERT and RoBERTa.

## Statistical Testing

Use paired bootstrap testing for model comparisons on the same test split.

Required comparisons:

- ALBERT vs BERT
- ALBERT vs RoBERTa
- BERT vs RoBERTa

For each comparison, report:

- Metric compared, at minimum fraud-class F1 and PR-AUC
- Mean difference
- 95% confidence interval
- Approximate p-value

Bootstrap samples should resample test examples with replacement and recompute metric differences. Because models share the same test examples for each seed, paired bootstrap is appropriate.

## Error Analysis

Save complete test predictions for every run with:

- `job_id`
- true label
- predicted label
- fraud probability
- text length
- title
- location
- cleaned text excerpt
- error type: `TP`, `TN`, `FP`, `FN`

Create summary tables for:

- Highest-confidence false positives
- Highest-confidence false negatives
- Cases ALBERT gets correct while BERT or RoBERTa gets wrong
- Cases all models get wrong

This gives qualitative evidence for paper discussion without expanding scope into full explainable AI.

## Subgroup Analysis

Compute subgroup metrics for EMSCAD metadata fields:

- `telecommuting`
- `has_company_logo`
- `has_questions`
- `employment_type`
- `required_experience`
- `industry`
- `function`

For each subgroup, report:

- Sample count
- Fraud count
- Fraud precision
- Fraud recall
- Fraud F1

Small subgroups must be labeled carefully to avoid overclaiming from tiny sample counts.

## Paper Tables and Figures

The notebook should end with a "Paper Tables and Figures" section that produces ready-to-use outputs:

- Table 1: Dataset statistics and class distribution
- Table 2: Model performance mean +/- std
- Table 3: Statistical comparison results
- Table 4: Runtime comparison
- Figure 1: Model comparison with error bars
- Figure 2: ROC curves
- Figure 3: PR curves
- Figure 4: Loss learning curves
- Figure 5: Accuracy learning curves
- Figure 6: F1 learning curves

## Best Model Export

Keep exporting the selected model to:

```text
best_model/
```

The metadata file must keep existing app-compatible fields:

```json
{
  "model_name": "...",
  "hf_model_id": "...",
  "metrics": {},
  "config": {},
  "exported_at": "..."
}
```

Additional fields may be added:

```json
{
  "selection_metric": "fraud_f1_mean",
  "multi_seed_summary": {},
  "selected_seed": 42,
  "selected_threshold": 0.5,
  "artifact_path": "artifacts/runs/seed_42/albert"
}
```

The Streamlit app should continue to load the model without modification.

## Recommended Later Improvements

These are intentionally outside this implementation scope:

- TF-IDF + Logistic Regression or SVM baseline
- DistilBERT, ELECTRA, DeBERTa, or domain-specific variants
- External validation dataset
- Calibration metrics such as Brier score and expected calibration error
- SHAP or LIME explanations
- Packaging repeated notebook logic into Python modules

## Acceptance Criteria

- The notebook can run a single model/seed and save complete artifacts.
- The notebook can resume without retraining completed runs when `FORCE_RETRAIN=False`.
- The notebook can aggregate partial or complete run artifacts.
- Multi-seed summaries report mean +/- std by model.
- ROC, PR, loss, accuracy, and F1 figures are generated.
- Training time is recorded for every model/seed.
- Statistical tests are saved to CSV.
- Error and subgroup analysis outputs are saved.
- `best_model/` export remains compatible with the existing Streamlit app.
