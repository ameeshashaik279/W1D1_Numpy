
import pandas as pd
import numpy as np

# Create a sample CSV dataset
df = pd.DataFrame({
    "Hours_Studied": [2, 3, 4, 5, 6],
    "Marks": [45, 55, 65, 75, 85]
})

df.to_csv("sample_dataset.csv", index=False)

# Calculate statistics
print("Mean:")
print(df.mean())

print("\nStandard Deviation:")
print(df.std())

print("\nCorrelation:")
print(df.corr())
