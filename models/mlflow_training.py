
import argparse
import sys
from pathlib import Path
from contextlib import nullcontext

import joblib
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Project root directory
ROOT = Path(__file__).resolve().parents[1]

# Add project root to Python path
sys.path.insert(0, str(ROOT))

# Import test data
from preprocessing.pipeline import X_test, y_test


# Project root directory
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/tuned_model.pkl"
    )
    args = parser.parse_args()

    # Load trained model
    model_path = ROOT / args.model_path
    model = joblib.load(model_path)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # Calculate ROC-AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)

        if len(model.classes_) == 2:
            roc_auc = roc_auc_score(y_test, y_score[:, 1])
        else:
            roc_auc = roc_auc_score(
                y_test,
                y_score,
                multi_class="ovr",
                average="weighted"
            )
    else:
        roc_auc = 0.0

    # MLflow Projects creates an active run automatically.
    # Start a run only when running this script independently.
    run_context = (
        nullcontext()
        if mlflow.active_run() is not None
        else mlflow.start_run()
    )

    with run_context:
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("model_path", str(args.model_path))

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        print("Model logged successfully to MLflow!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()
