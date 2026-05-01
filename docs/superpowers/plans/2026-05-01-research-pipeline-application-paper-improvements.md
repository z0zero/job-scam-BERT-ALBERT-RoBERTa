# Research Pipeline Application Paper Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `research_pipeline.ipynb` with TF-IDF Logistic Regression and Linear SVM baselines, Transformer early stopping, and application-paper reporting outputs while preserving the current Streamlit-compatible Transformer export.

**Architecture:** Keep the notebook as the single research pipeline, but add focused helper functions for baseline training, score-based threshold tuning, model grouping, latency measurement, deployment reporting, and Transformer-only export selection. Existing Transformer training, artifact persistence, aggregation, plots, and error analysis should be adapted rather than replaced.

**Tech Stack:** Jupyter Notebook, Python, pandas, NumPy, scikit-learn, PyTorch, HuggingFace Transformers Trainer, matplotlib, seaborn, Google Colab Free T4, Streamlit-compatible HuggingFace export.

---

## Spec Reference

Approved spec:

```text
docs/superpowers/specs/2026-05-01-research-pipeline-application-paper-improvements-design.md
```

## File Map

| File | Action | Responsibility |
| --- | --- | --- |
| `research_pipeline.ipynb` | Modify | Main notebook: imports, config, baseline models, early stopping, unified execution, aggregation, plots, application-readiness outputs, Transformer-only export metadata. |
| `scripts/validate_research_notebook.py` | Modify | Structural validator for the new notebook markers. |
| `docs/superpowers/plans/2026-05-01-research-pipeline-application-paper-improvements.md` | Create | This implementation plan. |

Generated files after notebook execution remain under:

```text
artifacts/
best_model/
```

No Streamlit app code is modified in this plan. The notebook will export richer metadata, but `best_model/` remains a HuggingFace Transformer model directory.

---

### Task 1: Verify The Current Notebook Baseline

**Files:**
- Read: `research_pipeline.ipynb`
- Read: `scripts/validate_research_notebook.py`

- [ ] **Step 1: Run the existing structural validator**

Run:

```powershell
python scripts/validate_research_notebook.py
```

Expected output:

```text
Notebook validation passed: research_pipeline.ipynb (48 cells)
```

- [ ] **Step 2: Confirm the current Git state is clean**

Run:

```powershell
git status --short
```

Expected output:

```text

```

If there are unrelated user changes, do not revert them. Work around them or stop and ask only if they block this plan.

---

### Task 2: Add Imports, Registries, And Runtime Controls

**Files:**
- Modify: `research_pipeline.ipynb`
- Modify: `scripts/validate_research_notebook.py`

- [ ] **Step 1: Update the notebook import cell**

In the first code cell, extend the imports so the notebook can train classical baselines and use Transformer early stopping.

Add these imports near the existing scikit-learn and transformers imports:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import EarlyStoppingCallback
```

Keep the existing imports for `AutoModelForSequenceClassification`, `AutoTokenizer`, `Trainer`, and `TrainingArguments`.

- [ ] **Step 2: Update the setup/configuration cell**

In the configuration cell, keep `MODEL_REGISTRY` as the Transformer registry and add a separate baseline registry plus execution list.

Use this structure:

```python
MODEL_REGISTRY = {
    "BERT": "bert-base-uncased",
    "ALBERT": "albert-base-v2",
    "RoBERTa": "roberta-base",
}

BASELINE_REGISTRY = {
    "TFIDF_LogReg": "TF-IDF + Logistic Regression",
    "TFIDF_LinearSVM": "TF-IDF + Linear SVM",
}

EXPERIMENT_SEEDS = [42, 123, 2024]
RUN_MODE = "single_seed"  # options: "single_seed", "full_multi_seed"
SINGLE_SEED = 42
BASELINES_TO_RUN = ["TFIDF_LogReg", "TFIDF_LinearSVM"]
TRANSFORMERS_TO_RUN = ["BERT", "ALBERT", "RoBERTa"]
MODELS_TO_RUN = BASELINES_TO_RUN + TRANSFORMERS_TO_RUN
FORCE_RETRAIN = False
EVALUATE_TRAIN_EACH_EPOCH = True
```

Update `CONFIG` so Transformer training uses five maximum epochs and early stopping patience two:

```python
CONFIG = {
    "max_len": 256,
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "epochs": 5,
    "early_stopping_patience": 2,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "test_size": 0.15,
    "val_size": 0.15,
    "threshold_grid": np.round(np.arange(0.10, 0.91, 0.01), 2).tolist(),
    "score_threshold_quantiles": np.round(np.arange(0.01, 1.00, 0.01), 2).tolist(),
    "latency_sample_size": 128,
    "latency_repeats": 3,
    "bootstrap_iterations": 1000,
    "bootstrap_seed": 777,
}
```

- [ ] **Step 3: Add model grouping helpers**

In the helper cell containing `model_slug()` and `run_dir_for()`, add these functions after `run_dir_for()`:

```python
def get_all_model_labels() -> list[str]:
    return list(BASELINE_REGISTRY) + list(MODEL_REGISTRY)


def model_group_for(model_label: str) -> str:
    if model_label in BASELINE_REGISTRY:
        return "classical_baseline"
    if model_label in MODEL_REGISTRY:
        return "transformer"
    raise ValueError(f"Unknown model label: {model_label}")


def is_transformer_model(model_label: str) -> bool:
    return model_label in MODEL_REGISTRY
```

- [ ] **Step 4: Update the structural validator markers**

In `scripts/validate_research_notebook.py`, add these strings to `REQUIRED_MARKERS`:

```python
"BASELINE_REGISTRY",
"BASELINES_TO_RUN",
"TRANSFORMERS_TO_RUN",
"get_all_model_labels",
"model_group_for",
"is_transformer_model",
"EarlyStoppingCallback",
"early_stopping_patience",
"score_threshold_quantiles",
"latency_sample_size",
"latency_repeats",
```

- [ ] **Step 5: Run the validator**

Run:

```powershell
python scripts/validate_research_notebook.py
```

Expected output:

```text
Notebook validation passed: research_pipeline.ipynb (48 cells)
```

- [ ] **Step 6: Commit Task 2**

Run:

```powershell
git add research_pipeline.ipynb scripts/validate_research_notebook.py
git commit -S -m "feat: add model registries and runtime controls"
```

---

### Task 3: Make Threshold Tuning Score-Based

**Files:**
- Modify: `research_pipeline.ipynb`
- Modify: `scripts/validate_research_notebook.py`

- [ ] **Step 1: Update metric helper names and threshold candidate generation**

In the metrics helper cell, keep `compute_binary_metrics()` compatible with existing callers, but treat its second argument as a score rather than always as a probability. Add `build_threshold_candidates()` and update `find_best_threshold()`.

Use this implementation:

```python
def build_threshold_candidates(fraud_scores: np.ndarray) -> list[float]:
    fraud_scores = np.asarray(fraud_scores).astype(float)
    fraud_scores = fraud_scores[np.isfinite(fraud_scores)]
    if fraud_scores.size == 0:
        return [0.50]
    if fraud_scores.min() >= 0.0 and fraud_scores.max() <= 1.0:
        return [float(threshold) for threshold in CONFIG["threshold_grid"]]
    quantiles = np.quantile(fraud_scores, CONFIG["score_threshold_quantiles"])
    candidates = sorted({float(value) for value in quantiles if np.isfinite(value)})
    if 0.0 >= fraud_scores.min() and 0.0 <= fraud_scores.max():
        candidates.append(0.0)
    return sorted(set(candidates))


def find_best_threshold(labels: np.ndarray, fraud_scores: np.ndarray) -> tuple[float, dict]:
    candidates = build_threshold_candidates(fraud_scores)
    best_threshold = candidates[0]
    best_metrics = compute_binary_metrics(labels, fraud_scores, best_threshold)
    for threshold in candidates[1:]:
        metrics = compute_binary_metrics(labels, fraud_scores, threshold)
        if metrics["fraud_f1"] > best_metrics["fraud_f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics
```

Keep `compute_trainer_metrics()` unchanged except for local variable naming if desired.

- [ ] **Step 2: Update prediction frame output**

In `build_predictions_frame()`, change the signature so it records generic scores and score type:

```python
def build_predictions_frame(
    test_df: pd.DataFrame,
    fraud_scores: np.ndarray,
    threshold: float,
    score_type: str,
) -> pd.DataFrame:
    predictions = (fraud_scores >= threshold).astype(int)
    output_columns = [
        "job_id",
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits",
        "fraudulent",
    ]
    available_columns = [column for column in output_columns if column in test_df.columns]
    predictions_df = test_df[available_columns].copy()
    predictions_df["true_label"] = test_df["fraudulent"].astype(int).values
    predictions_df["fraud_score"] = np.asarray(fraud_scores).astype(float)
    predictions_df["fraud_probability"] = (
        np.asarray(fraud_scores).astype(float) if score_type == "probability" else np.nan
    )
    predictions_df["score_type"] = score_type
    predictions_df["threshold"] = float(threshold)
    predictions_df["predicted_label"] = predictions
    predictions_df["error_type"] = np.select(
        [
            (predictions_df["true_label"] == 0) & (predictions_df["predicted_label"] == 1),
            (predictions_df["true_label"] == 1) & (predictions_df["predicted_label"] == 0),
        ],
        ["FP", "FN"],
        default="correct",
    )
    return predictions_df
```

- [ ] **Step 3: Update the Transformer call site**

In `train_and_evaluate()`, update the prediction frame call:

```python
predictions_df = build_predictions_frame(test_df, test_probs, tuned_threshold, "probability")
```

- [ ] **Step 4: Update validator markers**

Add these strings to `REQUIRED_MARKERS`:

```python
"build_threshold_candidates",
"fraud_score",
"score_type",
```

- [ ] **Step 5: Run the validator**

Run:

```powershell
python scripts/validate_research_notebook.py
```

Expected output:

```text
Notebook validation passed: research_pipeline.ipynb (48 cells)
```

- [ ] **Step 6: Commit Task 3**

Run:

```powershell
git add research_pipeline.ipynb scripts/validate_research_notebook.py
git commit -S -m "feat: support score-based threshold tuning"
```

---

### Task 4: Add Classical Baseline Training And Evaluation

**Files:**
- Modify: `research_pipeline.ipynb`
- Modify: `scripts/validate_research_notebook.py`

- [ ] **Step 1: Add a baseline training cell after the Transformer runner cell**

Add a new code cell after `train_and_evaluate()` with these functions:

```python
def build_baseline_pipeline(model_label: str, seed: int) -> Pipeline:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
        sublinear_tf=True,
    )
    if model_label == "TFIDF_LogReg":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
            random_state=seed,
        )
    elif model_label == "TFIDF_LinearSVM":
        classifier = LinearSVC(class_weight="balanced", random_state=seed)
    else:
        raise ValueError(f"Unsupported baseline model: {model_label}")
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def score_baseline_model(pipeline: Pipeline, texts: pd.Series | list[str]) -> tuple[np.ndarray, str]:
    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(texts)[:, 1]
        return np.asarray(scores).astype(float), "probability"
    scores = pipeline.decision_function(texts)
    return np.asarray(scores).astype(float), "decision_function"


def train_and_evaluate_baseline(
    model_label: str,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    set_seed(seed)
    current_run_dir = run_dir_for(seed, model_label)
    completed_marker = current_run_dir / "metrics_tuned_threshold.json"

    if completed_marker.exists() and not FORCE_RETRAIN:
        print(f"Skipping completed run: seed={seed}, model={model_label}")
        return {"seed": seed, "model_label": model_label, "run_dir": str(current_run_dir), "status": "skipped"}

    start_time = time.time()
    started_at = datetime.now().isoformat()
    current_run_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_baseline_pipeline(model_label, seed)
    pipeline.fit(train_df["text"], train_df["fraudulent"].astype(int))

    val_scores, score_type = score_baseline_model(pipeline, val_df["text"])
    test_scores, _ = score_baseline_model(pipeline, test_df["text"])
    val_labels = val_df["fraudulent"].astype(int).values
    test_labels = test_df["fraudulent"].astype(int).values

    tuned_threshold, validation_tuned_metrics = find_best_threshold(val_labels, val_scores)
    default_threshold = 0.50 if score_type == "probability" else 0.0
    default_metrics = compute_binary_metrics(test_labels, test_scores, threshold=default_threshold)
    tuned_metrics = compute_binary_metrics(test_labels, test_scores, threshold=tuned_threshold)
    tuned_metrics["validation_selected_threshold"] = float(tuned_threshold)
    tuned_metrics["validation_fraud_f1_at_threshold"] = float(validation_tuned_metrics["fraud_f1"])

    predictions_df = build_predictions_frame(test_df, test_scores, tuned_threshold, score_type)
    training_history_df = pd.DataFrame(columns=["epoch"])

    total_seconds = time.time() - start_time
    runtime_payload = {
        "seed": seed,
        "model_label": model_label,
        "model_group": model_group_for(model_label),
        "started_at": started_at,
        "ended_at": datetime.now().isoformat(),
        "total_seconds": float(total_seconds),
        "total_minutes": float(total_seconds / 60),
        "score_type": score_type,
        "epochs": None,
        "max_len": None,
    }
    config_payload = {
        "seed": seed,
        "model_label": model_label,
        "model_group": model_group_for(model_label),
        "baseline_description": BASELINE_REGISTRY[model_label],
        "config": CONFIG,
    }
    save_run_artifacts(
        current_run_dir,
        config_payload,
        runtime_payload,
        default_metrics,
        tuned_metrics,
        predictions_df,
        training_history_df,
    )

    return {
        "seed": seed,
        "model_label": model_label,
        "run_dir": str(current_run_dir),
        "status": "completed",
        "fraud_f1": tuned_metrics["fraud_f1"],
        "pr_auc": tuned_metrics["pr_auc"],
        "runtime_minutes": runtime_payload["total_minutes"],
    }
```

- [ ] **Step 2: Add the unified model dispatcher**

In the same new cell, add:

```python
def train_and_evaluate_any_model(
    model_label: str,
    seed: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    if model_label in BASELINE_REGISTRY:
        return train_and_evaluate_baseline(model_label, seed, train_df, val_df, test_df)
    if model_label in MODEL_REGISTRY:
        return train_and_evaluate(model_label, seed, train_df, val_df, test_df)
    raise ValueError(f"Unknown model label: {model_label}")
```

- [ ] **Step 3: Update the execution cell**

Replace the call to `train_and_evaluate()` in the execution loop with:

```python
result = train_and_evaluate_any_model(model_label, seed, train_df, val_df, test_df)
```

Keep the loop over `MODELS_TO_RUN` so baselines run before Transformers.

- [ ] **Step 4: Update validator markers**

Add these strings to `REQUIRED_MARKERS`:

```python
"build_baseline_pipeline",
"score_baseline_model",
"train_and_evaluate_baseline",
"train_and_evaluate_any_model",
"TFIDF_LogReg",
"TFIDF_LinearSVM",
```

- [ ] **Step 5: Run the validator**

Run:

```powershell
python scripts/validate_research_notebook.py
```

Expected output:

```text
Notebook validation passed: research_pipeline.ipynb (49 cells)
```

The exact cell count may be higher than 49 if implementation splits baseline helpers across multiple cells. The pass condition is the required markers, not the exact cell count.

- [ ] **Step 6: Commit Task 4**

Run:

```powershell
git add research_pipeline.ipynb scripts/validate_research_notebook.py
git commit -S -m "feat: add tfidf baseline training"
```

---

### Task 5: Add Transformer Early Stopping

**Files:**
- Modify: `research_pipeline.ipynb`

- [ ] **Step 1: Add callbacks to `WeightedTrainer` creation**

In `train_and_evaluate()`, add the callback argument to the `WeightedTrainer` constructor:

```python
trainer = WeightedTrainer(
    class_weights=compute_class_weights(train_df),
    model=model,
    args=build_training_arguments(current_run_dir / "trainer_output", seed, "eval_validation_fraud_f1"),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_trainer_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=CONFIG["early_stopping_patience"],
            early_stopping_threshold=0.0,
        )
    ],
)
```

- [ ] **Step 2: Record early stopping configuration in runtime artifacts**

In the `runtime_payload` created by `train_and_evaluate()`, add:

```python
"model_group": model_group_for(model_label),
"early_stopping_patience": CONFIG["early_stopping_patience"],
"score_type": "probability",
```

- [ ] **Step 3: Run the validator**

Run:

```powershell
python scripts/validate_research_notebook.py
```

Expected output:

```text
Notebook validation passed: research_pipeline.ipynb
```

- [ ] **Step 4: Commit Task 5**

Run:

```powershell
git add research_pipeline.ipynb
git commit -S -m "feat: add transformer early stopping"
```

---

### Task 6: Aggregate All Model Groups

**Files:**
- Modify: `research_pipeline.ipynb`

- [ ] **Step 1: Update `load_completed_run()`**

Add `model_group` and `score_type` to the row:

```python
row = {
    "seed": seed,
    "model_label": model_label,
    "model_group": model_group_for(model_label),
    "run_dir": str(current_run_dir),
    "default_threshold": default_metrics["threshold"],
    "tuned_threshold": tuned_metrics["threshold"],
    "runtime_minutes": runtime_metrics["total_minutes"],
    "score_type": runtime_metrics.get("score_type"),
}
```

- [ ] **Step 2: Update aggregation loops**

In `aggregate_completed_runs()`, replace:

```python
for model_label in MODEL_REGISTRY:
```

with:

```python
for model_label in get_all_model_labels():
```

Update the summary grouping to include model group:

```python
summary_df = (
    all_runs_df.groupby(["model_group", "model_label"])[metric_columns]
    .agg(["mean", "std"])
    .sort_values(("tuned_fraud_f1", "mean"), ascending=False)
)
```

- [ ] **Step 3: Update runtime CSV**

Replace the runtime export with:

```python
all_runs_df[["seed", "model_group", "model_label", "runtime_minutes"]].to_csv(
    SUMMARY_DIR / "runtime_by_model.csv",
    index=False,
)
```

- [ ] **Step 4: Run the validator**

Run:

```powershell
python scripts/validate_research_notebook.py
```

Expected output:

```text
Notebook validation passed: research_pipeline.ipynb
```

- [ ] **Step 5: Commit Task 6**

Run:

```powershell
git add research_pipeline.ipynb
git commit -S -m "feat: aggregate baseline and transformer runs"
```

---

### Task 7: Update Prediction Loading, Plots, Statistical Tests, And Error Summaries

**Files:**
- Modify: `research_pipeline.ipynb`
- Modify: `scripts/validate_research_notebook.py`

- [ ] **Step 1: Update prediction loading loops**

In `load_prediction_frames()`, replace the model loop with:

```python
for model_label in get_all_model_labels():
```

Add model group to each loaded frame:

```python
frame["model_group"] = model_group_for(model_label)
```

- [ ] **Step 2: Update ROC/PR plotting to use `fraud_score`**

In `plot_roc_pr_curves()`, replace:

```python
probs = model_predictions["fraud_probability"].values
```

with:

```python
scores = model_predictions["fraud_score"].values
```

Then update metric calls:

```python
fpr, tpr, _ = roc_curve(labels, scores)
precision, recall, _ = precision_recall_curve(labels, scores)
axes[0].plot(fpr, tpr, label=f"{model_label} AUC={roc_auc_score(labels, scores):.3f}")
axes[1].plot(recall, precision, label=f"{model_label} AP={average_precision_score(labels, scores):.3f}")
```

- [ ] **Step 3: Update statistical tests to use all completed models**

In `run_statistical_tests()`, replace the hard-coded comparison list with completed pair generation:

```python
available_models = [model_label for model_label in get_all_model_labels() if model_label in prediction_frames]
comparisons = [
    (available_models[i], available_models[j])
    for i in range(len(available_models))
    for j in range(i + 1, len(available_models))
]
```

Replace prediction score columns:

```python
probs_a=frame_a["fraud_score"].values,
probs_b=frame_b["fraud_score"].values,
```

Keep the existing metric names `fraud_f1` and `pr_auc`.

- [ ] **Step 4: Add error summary by model**

After `build_error_analysis()`, add:

```python
def build_error_summary_by_model() -> pd.DataFrame:
    predictions_df = load_prediction_frames()
    if predictions_df.empty:
        empty = pd.DataFrame()
        empty.to_csv(SUMMARY_DIR / "error_summary_by_model.csv", index=False)
        return empty
    summary_df = (
        predictions_df.groupby(["model_group", "model_label", "error_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ["FP", "FN", "correct"]:
        if column not in summary_df.columns:
            summary_df[column] = 0
    summary_df["total"] = summary_df[["FP", "FN", "correct"]].sum(axis=1)
    summary_df.to_csv(SUMMARY_DIR / "error_summary_by_model.csv", index=False)
    return summary_df


error_summary_by_model = build_error_summary_by_model()
display(error_summary_by_model)
```

- [ ] **Step 5: Update model comparison plotting index handling**

Because `mean_std_by_model` now has a multi-index, update `plot_model_comparison()` before plotting:

```python
plot_means = means.copy()
plot_means.index = [
    f"{model_group}\n{model_label}" if isinstance(model_group, str) else str(model_label)
    for model_group, model_label in plot_means.index
]
plot_stds = stds.copy()
plot_stds.index = plot_means.index
ax = plot_means.plot(kind="bar", yerr=plot_stds, figsize=(12, 6), capsize=4)
```

- [ ] **Step 6: Update validator markers**

Add these strings to `REQUIRED_MARKERS`:

```python
"build_error_summary_by_model",
"error_summary_by_model.csv",
"fraud_score",
```

- [ ] **Step 7: Run the validator**

Run:

```powershell
python scripts/validate_research_notebook.py
```

Expected output:

```text
Notebook validation passed: research_pipeline.ipynb
```

- [ ] **Step 8: Commit Task 7**

Run:

```powershell
git add research_pipeline.ipynb scripts/validate_research_notebook.py
git commit -S -m "feat: compare all model predictions"
```

---

### Task 8: Add Application-Readiness Outputs

**Files:**
- Modify: `research_pipeline.ipynb`
- Modify: `scripts/validate_research_notebook.py`

- [ ] **Step 1: Add size and latency helpers**

Add a new code cell before "Execute Resumable Experiments". These helpers must be defined before the training loop calls `train_and_evaluate()` or `train_and_evaluate_baseline()`.

```python
def compute_directory_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    total_bytes = sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())
    return float(total_bytes / (1024 * 1024))


def summarize_latency_seconds(latencies: list[float], sample_count: int) -> dict:
    if not latencies or sample_count == 0:
        return {"inference_latency_ms_per_sample_mean": None, "inference_latency_ms_per_sample_std": None}
    per_sample_ms = np.asarray(latencies, dtype=float) * 1000.0 / sample_count
    return {
        "inference_latency_ms_per_sample_mean": float(per_sample_ms.mean()),
        "inference_latency_ms_per_sample_std": float(per_sample_ms.std(ddof=0)),
    }
```

- [ ] **Step 2: Add baseline latency measurement inside `train_and_evaluate_baseline()`**

After computing `test_scores`, add:

```python
latency_texts = test_df["text"].head(CONFIG["latency_sample_size"])
latency_seconds = []
for _ in range(CONFIG["latency_repeats"]):
    latency_start = time.time()
    score_baseline_model(pipeline, latency_texts)
    latency_seconds.append(time.time() - latency_start)
latency_payload = summarize_latency_seconds(latency_seconds, len(latency_texts))
```

Then merge the payload into `runtime_payload`:

```python
runtime_payload.update(latency_payload)
```

- [ ] **Step 3: Add Transformer latency measurement inside `train_and_evaluate()`**

After `test_output = trainer.predict(test_dataset)`, add:

```python
latency_count = min(CONFIG["latency_sample_size"], len(test_dataset))
latency_dataset = torch.utils.data.Subset(test_dataset, list(range(latency_count)))
latency_seconds = []
for _ in range(CONFIG["latency_repeats"]):
    latency_start = time.time()
    trainer.predict(latency_dataset)
    latency_seconds.append(time.time() - latency_start)
latency_payload = summarize_latency_seconds(latency_seconds, latency_count)
```

Then merge the payload into `runtime_payload`:

```python
runtime_payload.update(latency_payload)
```

- [ ] **Step 4: Build deployment readiness table**

Add this function near the paper table section:

```python
def build_deployment_readiness_table(all_runs_df: pd.DataFrame) -> pd.DataFrame:
    if all_runs_df.empty:
        empty = pd.DataFrame()
        empty.to_csv(SUMMARY_DIR / "deployment_readiness.csv", index=False)
        return empty
    rows = []
    for _, row in all_runs_df.iterrows():
        runtime = read_json(Path(row["run_dir"]) / "runtime.json")
        model_dir = Path(row["run_dir"]) / "model"
        rows.append(
            {
                "seed": row["seed"],
                "model_group": row["model_group"],
                "model_label": row["model_label"],
                "selected_threshold": row["tuned_threshold"],
                "tuned_fraud_f1": row["tuned_fraud_f1"],
                "tuned_pr_auc": row["tuned_pr_auc"],
                "tuned_roc_auc": row["tuned_roc_auc"],
                "runtime_minutes": row["runtime_minutes"],
                "inference_latency_ms_per_sample_mean": runtime.get("inference_latency_ms_per_sample_mean"),
                "inference_latency_ms_per_sample_std": runtime.get("inference_latency_ms_per_sample_std"),
                "model_size_mb": compute_directory_size_mb(model_dir),
                "export_eligible": bool(is_transformer_model(row["model_label"])),
            }
        )
    readiness_df = pd.DataFrame(rows)
    readiness_df.to_csv(SUMMARY_DIR / "deployment_readiness.csv", index=False)
    return readiness_df


deployment_readiness_df = build_deployment_readiness_table(all_runs_df)
display(deployment_readiness_df)
```

- [ ] **Step 5: Export thresholds by model**

In the paper table section, add:

```python
thresholds_by_model = (
    all_runs_df[["seed", "model_group", "model_label", "tuned_threshold", "score_type"]].copy()
    if "all_runs_df" in globals() and not all_runs_df.empty
    else pd.DataFrame()
)
thresholds_by_model.to_csv(SUMMARY_DIR / "thresholds_by_model.csv", index=False)
display(thresholds_by_model.head())
```

- [ ] **Step 6: Update validator markers**

Add these strings to `REQUIRED_MARKERS`:

```python
"compute_directory_size_mb",
"summarize_latency_seconds",
"build_deployment_readiness_table",
"deployment_readiness.csv",
"thresholds_by_model.csv",
```

- [ ] **Step 7: Run the validator**

Run:

```powershell
python scripts/validate_research_notebook.py
```

Expected output:

```text
Notebook validation passed: research_pipeline.ipynb
```

- [ ] **Step 8: Commit Task 8**

Run:

```powershell
git add research_pipeline.ipynb scripts/validate_research_notebook.py
git commit -S -m "feat: add deployment readiness reporting"
```

---

### Task 9: Restrict Best Model Export To Transformers And Expand Metadata

**Files:**
- Modify: `research_pipeline.ipynb`

- [ ] **Step 1: Update `select_best_completed_run()`**

Replace the function with:

```python
def select_best_completed_run(all_runs_df: pd.DataFrame) -> pd.Series:
    if all_runs_df.empty:
        raise ValueError("No completed runs available for best model export.")
    transformer_runs = all_runs_df[all_runs_df["model_label"].isin(MODEL_REGISTRY)].copy()
    if transformer_runs.empty:
        raise ValueError("No completed Transformer runs available for best model export.")
    return transformer_runs.sort_values(
        ["tuned_fraud_f1", "tuned_pr_auc", "tuned_fraud_recall"],
        ascending=False,
    ).iloc[0]
```

- [ ] **Step 2: Expand `model_meta.json` payload**

In `export_best_model()`, replace the `meta` assignment with:

```python
meta = {
    "model_name": model_label,
    "hf_model_id": MODEL_REGISTRY[model_label],
    "selected_threshold": float(best_run["tuned_threshold"]),
    "selection_metric": "tuned_fraud_f1",
    "selected_seed": int(best_run["seed"]),
    "metrics": {
        "accuracy": float(best_run["tuned_accuracy"]),
        "precision": float(best_run["tuned_fraud_precision"]),
        "recall": float(best_run["tuned_fraud_recall"]),
        "f1": float(best_run["tuned_fraud_f1"]),
        "macro_f1": float(best_run["tuned_macro_f1"]),
        "balanced_accuracy": float(best_run["tuned_balanced_accuracy"]),
        "specificity": float(best_run["tuned_specificity"]),
        "pr_auc": float(best_run["tuned_pr_auc"]),
        "roc_auc": float(best_run["tuned_roc_auc"]),
    },
    "config": CONFIG,
    "preprocessing": {
        "text_columns": TEXT_COLUMNS,
        "cleaning": "html_unescape_strip_html_remove_urls_lowercase_normalize_whitespace",
    },
    "artifact_path": str(best_run["run_dir"]),
    "multi_seed_summary_path": str(SUMMARY_DIR / "mean_std_by_model.csv"),
    "deployment_readiness_path": str(SUMMARY_DIR / "deployment_readiness.csv"),
    "exported_at": datetime.now().isoformat(),
}
```

- [ ] **Step 3: Run the validator**

Run:

```powershell
python scripts/validate_research_notebook.py
```

Expected output:

```text
Notebook validation passed: research_pipeline.ipynb
```

- [ ] **Step 4: Commit Task 9**

Run:

```powershell
git add research_pipeline.ipynb
git commit -S -m "feat: expand transformer export metadata"
```

---

### Task 10: Final Structural Verification

**Files:**
- Read: `research_pipeline.ipynb`
- Read: `scripts/validate_research_notebook.py`

- [ ] **Step 1: Run the validator**

Run:

```powershell
python scripts/validate_research_notebook.py
```

Expected output:

```text
Notebook validation passed: research_pipeline.ipynb
```

- [ ] **Step 2: Compile the validator script**

Run:

```powershell
python -m py_compile scripts/validate_research_notebook.py
```

Expected output:

```text

```

- [ ] **Step 3: Check Git status**

Run:

```powershell
git status --short
```

Expected output:

```text

```

---

## Colab Execution Checklist After Implementation

Run this in Google Colab after uploading `fake_job_postings.csv`:

1. Set debug mode:

```python
RUN_MODE = "single_seed"
SINGLE_SEED = 42
FORCE_RETRAIN = False
```

2. Run all cells once. Confirm baselines complete and at least one Transformer run starts.

3. For final paper results, set:

```python
RUN_MODE = "full_multi_seed"
FORCE_RETRAIN = False
```

4. Run the notebook in resumable sessions until all intended model/seed artifacts exist.

5. Confirm these files exist:

```text
artifacts/summary/all_runs.csv
artifacts/summary/mean_std_by_model.csv
artifacts/summary/deployment_readiness.csv
artifacts/summary/thresholds_by_model.csv
artifacts/summary/error_summary_by_model.csv
best_model/model_meta.json
```

6. Confirm `best_model/model_meta.json` names one of:

```text
BERT
ALBERT
RoBERTa
```

It must not name `TFIDF_LogReg` or `TFIDF_LinearSVM`.
