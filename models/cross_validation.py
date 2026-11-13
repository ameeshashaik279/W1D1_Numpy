
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import date
from sklearn.model_selection import cross_val_score, learning_curve

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import training data
from preprocessing.pipeline import X_train, y_train

# Load best saved model
model_path = ROOT / "models" / "best_model.pkl"
model = joblib.load(model_path)

# 5-fold cross-validation
cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

mean_score = cv_scores.mean()
std_score = cv_scores.std()

print("Cross-validation scores:", cv_scores)
print(f"Mean Accuracy: {mean_score:.4f}")
print(f"Standard Deviation: {std_score:.4f}")
print(f"Accuracy: {mean_score:.4f} ± {std_score:.4f}")

# Generate learning curve
train_sizes, train_scores, validation_scores = learning_curve(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.2, 1.0, 5),
    random_state=42
)

train_mean = train_scores.mean(axis=1)
validation_mean = validation_scores.mean(axis=1)

# Plot learning curve
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, marker="o", label="Training Score")
plt.plot(train_sizes, validation_mean, marker="o", label="Validation Score")

plt.title("Learning Curve - Logistic Regression")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
plot_path = ROOT / "models" / "learning_curve.png"
plt.savefig(plot_path)
plt.close()

# Append CV results
results_path = ROOT / "models" / "results.csv"
results = pd.read_csv(results_path)

new_rows = pd.DataFrame([
    {
        "model_name": "LogisticRegression",
        "metric_name": "CV_accuracy_mean",
        "score": mean_score,
        "date": date.today()
    },
    {
        "model_name": "LogisticRegression",
        "metric_name": "CV_accuracy_std",
        "score": std_score,
        "date": date.today()
    }
])

results = pd.concat([results, new_rows], ignore_index=True)
results.to_csv(results_path, index=False)

print("Learning curve saved successfully!")
print("CV scores added to results.csv")
print("W4D1 completed!")
