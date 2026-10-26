
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Convert dataset into a DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add target labels
df["target"] = iris.target

# Save dataset as CSV
df.to_csv("data/sample_dataset.csv", index=False)

# Display dataset information
print("Dataset loaded successfully!")
print("Rows and columns:", df.shape)
print("\nData types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())
