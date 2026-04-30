"""Validate required sections and code markers in research_pipeline.ipynb.

This script intentionally does not execute notebook cells. It checks that the
paper-oriented comparative study workflow is present after refactoring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


NOTEBOOK_PATH = Path("research_pipeline.ipynb")

REQUIRED_MARKERS = [
    "EXPERIMENT_SEEDS",
    "MODEL_REGISTRY",
    "RUN_MODE",
    "FORCE_RETRAIN",
    "EVALUATE_TRAIN_EACH_EPOCH",
    "get_environment_info",
    "write_json",
    "load_emscad_dataset",
    "prepare_text_columns",
    "clean_text",
    "create_stratified_split",
    "JobPostingDataset",
    "compute_binary_metrics",
    "find_best_threshold",
    "WeightedTrainer",
    "train_and_evaluate",
    "save_run_artifacts",
    "aggregate_completed_runs",
    "paired_bootstrap_test",
    "plot_roc_pr_curves",
    "plot_learning_curves",
    "build_error_analysis",
    "build_subgroup_metrics",
    "export_best_model",
]

REQUIRED_HEADINGS = [
    "## 1. Setup and Configuration",
    "## 2. Environment and Reproducibility Log",
    "## 3. Data Loading and Provenance",
    "## 4. EDA and Data Profile Export",
    "## 5. Text Preprocessing",
    "## 6. Repeated Stratified Split Generation",
    "## 7. Dataset Class",
    "## 8. Metrics Helpers",
    "## 9. Weighted Trainer and Training Runner",
    "## 10. Execute Resumable Experiments",
    "## 11. Aggregate Evaluation",
    "## 12. Statistical Testing",
    "## 13. Learning Curve Visualization",
    "## 14. ROC and PR Curve Visualization",
    "## 15. Error Analysis and Subgroup Analysis",
    "## 16. Paper Tables and Figures",
    "## 17. Export Best Model",
    "## 18. Colab Drive Backup",
]


def load_notebook(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    notebook = load_notebook(NOTEBOOK_PATH)
    cells = notebook.get("cells", [])
    source = "\n".join("".join(cell.get("source", [])) for cell in cells)

    missing_markers = [marker for marker in REQUIRED_MARKERS if marker not in source]
    missing_headings = [heading for heading in REQUIRED_HEADINGS if heading not in source]

    if missing_markers or missing_headings:
        if missing_markers:
            print("Missing required code markers:")
            for marker in missing_markers:
                print(f"- {marker}")
        if missing_headings:
            print("Missing required headings:")
            for heading in missing_headings:
                print(f"- {heading}")
        return 1

    print(f"Notebook validation passed: {NOTEBOOK_PATH} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
