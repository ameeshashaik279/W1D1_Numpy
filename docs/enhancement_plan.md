# W3D5 – MLflow Projects Enhancement Plan

## 1. Goal

Package the machine learning experiments using MLflow Projects so that experiments can be executed reproducibly with a single command.

## 2. Technique

Use MLflow Projects with a project configuration file (`MLproject`) to define:
- Python environment and dependencies.
- Experiment entry point.
- Training script and parameters.
- Model evaluation and MLflow tracking.

## 3. Expected Metric Improvement

- Maintain or improve the current best model accuracy of 93.33%.
- Improve experiment reproducibility and consistency.
- Enable easy comparison of model performance across runs.

## 4. Files to Change

- `MLproject` – Define project configuration and entry point.
- `conda.yaml` – Define Python dependencies.
- `models/model_selector.py` – Integrate model training and evaluation.
- `models/results.csv` – Store evaluation results.
- `README.md` – Document execution instructions.

## 5. Libraries and Data Sources

**Libraries:**
- MLflow
- Scikit-learn
- Pandas
- NumPy
- Joblib

**Data Source:**
- `data/sample_dataset.csv` (Iris dataset)

## 6. Subtasks

| Subtask | Description | Estimated Time |
|---|---|---|
| 1 | Create MLproject configuration and environment file | 1 hour |
| 2 | Integrate training script with MLflow tracking | 1.5 hours |
| 3 | Test reproducible execution and update README | 1 hour |

## 7. Expected Outcome

The project will support reproducible machine learning experiments through a single MLflow command, with consistent tracking of parameters, metrics, and model artifacts.

## 8. Approval

This enhancement plan is submitted for mentor review and approval before implementation.
