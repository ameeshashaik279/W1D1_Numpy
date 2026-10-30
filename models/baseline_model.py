import pandas as pd
import sys
from pathlib import Path
from datetime import date
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Add project root to Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.pipeline import X_train, X_test, y_train, y_test
# Train baseline model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Baseline Model: Logistic Regression")
print("Accuracy:", accuracy)

# Compare actual and predicted values
predictions = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nFirst 5 Predictions:")
print(predictions.head())

# Save predictions
predictions.to_csv("models/linear_predictions.csv", index=False)

# Save baseline results
results = pd.DataFrame([{
    "model_name": "LogisticRegression",
    "metric_name": "accuracy",
    "score": accuracy,
    "date": date.today()
}])

results.to_csv("models/results.csv", index=False)

print("\nFiles saved successfully!")
