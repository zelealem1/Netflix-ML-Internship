import pandas as pd

# Load the Netflix dataset
df = pd.read_csv("Dataset.csv")

# Basic information
print("====================================")
print("NETFLIX DATASET CHECK")
print("====================================")

print("Dataset loaded successfully!")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())