
import sys
import joblib
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import training data
from preprocessing.pipeline import X_train, y_train

# Define parameter grid
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["lbfgs", "liblinear"],
    "max_iter": [100, 200, 500]
}

# Initialize Logistic Regression
model = LogisticRegression()

# Randomized Search CV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

# Fit the search
random_search.fit(X_train, y_train)

# Display best results
print("Best Parameters:", random_search.best_params_)
print("Best CV Accuracy:", random_search.best_score_)

# Retrain best model on full available training set
best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)

# Save tuned model
model_path = ROOT / "models" / "tuned_model.pkl"
joblib.dump(best_model, model_path)

print("Tuned model saved successfully!")
print("Saved location:", model_path)
