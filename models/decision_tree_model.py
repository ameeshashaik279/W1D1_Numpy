
import pandas as pd
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

# Add project root to Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.pipeline import X_train, X_test, y_train, y_test

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Decision Tree Accuracy:", accuracy)

# Extract human-readable rules
df = pd.read_csv(ROOT / "data/sample_dataset.csv")
feature_names = df.drop(columns=["target"]).columns.tolist()

rules = export_text(model, feature_names=feature_names)

# Save decision tree rules
(ROOT / "models/decision_tree_rules.txt").write_text(
    rules, encoding="utf-8"
)

# Plot feature importances
importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(8, 5))
plt.bar(importance["feature"], importance["importance"])
plt.xticks(rotation=30, ha="right")
plt.title("Decision Tree Feature Importances")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(ROOT / "models/feature_importances.png")
plt.close()

# Append results
results_path = ROOT / "models/results.csv"
results = pd.read_csv(results_path)

new_result = pd.DataFrame([{
    "model_name": "DecisionTreeClassifier",
    "metric_name": "accuracy",
    "score": accuracy,
    "date": date.today()
}])

results = pd.concat([results, new_result], ignore_index=True)
results.to_csv(results_path, index=False)

print("Rules saved successfully!")
print("Feature importance plot saved successfully!")
print("W3D3 completed!")
