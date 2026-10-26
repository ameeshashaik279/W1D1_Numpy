
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path

# Load the Iris dataset saved on W1D4
df = pd.read_csv("data/sample_dataset.csv")

# Create output folder
Path("results").mkdir(exist_ok=True)

# Log a simple experiment
with mlflow.start_run(run_name="Iris_Week1_Analysis"):
    mlflow.log_param("dataset", "Iris")
    mlflow.log_param("rows", len(df))
    mlflow.log_param("features", 4)

    # Record a basic metric
    mlflow.log_metric("mean_sepal_length", df["sepal length (cm)"].mean())

    # Visualise sepal length distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["sepal length (cm)"], bins=15, edgecolor="black")
    plt.title("Iris Sepal Length Distribution")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    # Save chart
    output_path = "results/week1_visualization.png"
    plt.savefig(output_path)
    mlflow.log_artifact(output_path)
    plt.close()

print("Visualization saved to results/week1_visualization.png")
print("MLflow experiment logged successfully!")
