import pandas as pd
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Add project root to Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.pipeline import X_train, X_test, y_train, y_test
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Random Forest Accuracy:", accuracy)

features = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop 4 Feature Importances:")
print(importance)

scored = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
scored.to_csv("data/scored_records.csv", index=False)

results = pd.read_csv("models/results.csv")

new_result = pd.DataFrame([{
    "model_name": "RandomForestClassifier",
    "metric_name": "accuracy",
    "score": accuracy,
    "date": pd.Timestamp.today().date()
}])

results = pd.concat([results, new_result], ignore_index=True)
results.to_csv("models/results.csv", index=False)

print("W3D2 files saved successfully!")