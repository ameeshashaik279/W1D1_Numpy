
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.pipeline import X_test, y_test

# Load tuned model
model = joblib.load(ROOT / "models" / "tuned_model.pkl")

# Predict test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", auc)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(cmap="Blues")
plt.title("Tuned Logistic Regression - Confusion Matrix")
plt.tight_layout()

# Save heatmap
plt.savefig(ROOT / "models" / "confusion_matrix.png")
plt.close()

# Save metrics
metrics = pd.DataFrame([{
    "metric": "Accuracy",
    "score": accuracy
}, {
    "metric": "Precision",
    "score": precision
}, {
    "metric": "Recall",
    "score": recall
}, {
    "metric": "F1 Score",
    "score": f1
}, {
    "metric": "ROC-AUC",
    "score": auc
}])

metrics.to_csv(
    ROOT / "models" / "evaluation_metrics.csv",
    index=False
)

print("Confusion matrix saved successfully!")
print("Evaluation metrics saved successfully!")
