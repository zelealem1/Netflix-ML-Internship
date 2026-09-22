import pandas as pd

# =====================================================
# TASK 3 - RATING EXPLORATION
# =====================================================

df = pd.read_csv("Dataset.csv")

print("==============================================")
print("NETFLIX AUDIENCE RATING EXPLORATION")
print("==============================================")

print("\nTotal records:", len(df))

print("\nNumber of rating categories:")
print(df["rating"].nunique())

print("\nRating categories:")
print(df["rating"].value_counts())

print("\nLeast common rating categories:")
print(
    df["rating"]
    .value_counts()
    .sort_values()
    .head(10)
)