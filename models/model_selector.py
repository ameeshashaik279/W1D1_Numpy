
import pandas as pd
import joblib
import sys

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Project root path
ROOT = Path(__file__).resolve().parent.parent

# Add project root to Python path
sys.path.insert(0, str(ROOT))

# Import training data
from preprocessing.pipeline import X_train, y_train

# Load model results
results_path = ROOT / "models" / "results.csv"
results = pd.read_csv(results_path)

# Select best model based on accuracy
best_row = results.loc[results["score"].idxmax()]
best_model_name = best_row["model_name"]

print("Best Model:", best_model_name)
print("Best Accuracy:", best_row["score"])

# Initialize the selected model
if best_model_name == "LogisticRegression":
    best_model = LogisticRegression(max_iter=200)

elif best_model_name == "RandomForestClassifier":
    best_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

elif best_model_name == "DecisionTreeClassifier":
    best_model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

else:
    raise ValueError("Unknown model name")

# Retrain selected model on full training data
best_model.fit(X_train, y_train)

# Save the trained model
model_path = ROOT / "models" / "best_model.pkl"
joblib.dump(best_model, model_path)

print("Best model saved successfully!")
print("Saved location:", model_path)
