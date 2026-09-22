import pandas as pd

# Load dataset
df = pd.read_csv("Dataset.csv")

print("====================================")
print("NETFLIX DATA EXPLORATION")
print("====================================")

# 1. Count Movies and TV Shows
print("\nCONTENT TYPES:")
print(df["type"].value_counts())

# 2. Most common ratings
print("\nTOP RATINGS:")
print(df["rating"].value_counts().head(10))

# 3. Most common genres
print("\nMOST COMMON GENRE COMBINATIONS:")
print(df["listed_in"].value_counts().head(10))

# 4. Most recent release years
print("\nMOST RECENT RELEASE YEARS:")
print(df["release_year"].value_counts().sort_index(ascending=False).head(10))

# 5. Example titles
print("\nEXAMPLE TITLES:")
print(df["title"].head(10).to_string(index=False))