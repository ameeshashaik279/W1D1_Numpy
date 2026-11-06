import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# Load dataset
df = pd.read_csv("data/sample_dataset.csv")

# Select two classes for binary classification
df = df[df["target"].isin([0, 1])]

X = df.drop("target", axis=1)
y = df["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)

print("LOGISTIC REGRESSION")
print(classification_report(y_test, log_pred))

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("RANDOM FOREST")
print(classification_report(y_test, rf_pred))

# Compare models
print("\nMODEL COMPARISON")

for name, actual, predicted in [
    ("Logistic Regression", y_test, log_pred),
    ("Random Forest", y_test, rf_pred)
]:
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(actual, predicted))
    print("Precision:", precision_score(actual, predicted))
    print("Recall:", recall_score(actual, predicted))

# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ConfusionMatrixDisplay.from_predictions(
    y_test, log_pred, ax=axes[0]
)
axes[0].set_title("Logistic Regression")

ConfusionMatrixDisplay.from_predictions(
    y_test, rf_pred, ax=axes[1]
)
axes[1].set_title("Random Forest")

plt.tight_layout()
plt.savefig("models/confusion_matrix_comparison.png")
plt.close()

# ROC-AUC curves
fig, ax = plt.subplots(figsize=(7, 5))

RocCurveDisplay.from_estimator(
    log_model, X_test_scaled, y_test,
    ax=ax, name="Logistic Regression"
)

RocCurveDisplay.from_estimator(
    rf_model, X_test, y_test,
    ax=ax, name="Random Forest"
)

plt.title("ROC-AUC Comparison")
plt.savefig("models/roc_auc_comparison.png")
plt.show()
