import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score


# =====================================================
# TASK 4 - NETFLIX CONTENT SEGMENTATION
# =====================================================

print("==============================================")
print("NETFLIX CONTENT SEGMENTATION")
print("==============================================")


# =====================================================
# 1. Load dataset
# =====================================================

df = pd.read_csv("Dataset.csv")

print("\nDataset loaded successfully!")
print("Total titles:", len(df))


# =====================================================
# 2. Prepare data
# =====================================================

df["type"] = df["type"].fillna("Unknown")
df["rating"] = df["rating"].fillna("Unknown")
df["listed_in"] = df["listed_in"].fillna("")
df["release_year"] = df["release_year"].fillna(
    df["release_year"].median()
)


# =====================================================
# 3. Create combined content features
# =====================================================

df["cluster_text"] = (
    df["type"] + " " +
    df["rating"] + " " +
    df["listed_in"]
)


# =====================================================
# 4. Convert text features to numerical values
# =====================================================

vectorizer = TfidfVectorizer(
    stop_words="english"
)

text_matrix = vectorizer.fit_transform(
    df["cluster_text"]
)

print("\nText feature matrix shape:")
print(text_matrix.shape)


# =====================================================
# 5. Scale release year
# =====================================================

scaler = StandardScaler()

year_scaled = scaler.fit_transform(
    df[["release_year"]]
)


# =====================================================
# 6. Add release year to text features
# =====================================================

from scipy.sparse import csr_matrix, hstack

year_sparse = csr_matrix(year_scaled)

feature_matrix = hstack(
    [text_matrix, year_sparse]
)

print("\nFinal clustering feature matrix:")
print(feature_matrix.shape)


# =====================================================
# 7. Apply K-Means clustering
# =====================================================

number_of_clusters = 5

kmeans = KMeans(
    n_clusters=number_of_clusters,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(
    feature_matrix
)


print("\nK-Means clustering completed!")


# =====================================================
# 8. Cluster sizes
# =====================================================

print("\n==============================================")
print("CLUSTER SIZES")
print("==============================================")

cluster_sizes = df["cluster"].value_counts().sort_index()

print(cluster_sizes)


# =====================================================
# 9. Calculate silhouette score
# =====================================================

sample_size = min(2000, len(df))

sample_indices = df.sample(
    n=sample_size,
    random_state=42
).index

sample_matrix = feature_matrix[
    sample_indices
]

sample_labels = df.loc[
    sample_indices,
    "cluster"
]

silhouette = silhouette_score(
    sample_matrix,
    sample_labels
)

print("\nSilhouette Score:")
print(f"{silhouette:.4f}")


# =====================================================
# 10. Analyze clusters
# =====================================================

print("\n==============================================")
print("CLUSTER CHARACTERISTICS")
print("==============================================")


for cluster_number in range(number_of_clusters):

    cluster_data = df[
        df["cluster"] == cluster_number
    ]

    print("\n----------------------------------------------")
    print(f"CLUSTER {cluster_number}")
    print("----------------------------------------------")

    print(
        "Number of titles:",
        len(cluster_data)
    )

    print("\nContent types:")
    print(
        cluster_data["type"]
        .value_counts()
        .head(3)
    )

    print("\nRatings:")
    print(
        cluster_data["rating"]
        .value_counts()
        .head(5)
    )

    print("\nMost common genres:")
    print(
        cluster_data["listed_in"]
        .value_counts()
        .head(5)
    )

    print("\nAverage release year:")
    print(
        round(
            cluster_data["release_year"].mean(),
            1
        )
    )


# =====================================================
# 11. Show example titles from each cluster
# =====================================================

print("\n==============================================")
print("SAMPLE TITLES BY CLUSTER")
print("==============================================")


for cluster_number in range(number_of_clusters):

    cluster_titles = df[
        df["cluster"] == cluster_number
    ]["title"].head(5)

    print(
        f"\nCluster {cluster_number}:"
    )

    for title in cluster_titles:
        print("-", title)


# =====================================================
# 12. Reduce dimensions for visualization
# =====================================================

svd = TruncatedSVD(
    n_components=2,
    random_state=42
)

coordinates = svd.fit_transform(
    feature_matrix
)


# =====================================================
# 13. Create cluster visualization
# =====================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    coordinates[:, 0],
    coordinates[:, 1],
    c=df["cluster"],
    s=12,
    alpha=0.6
)

plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("Netflix Content Segmentation - K-Means")

plt.tight_layout()

plt.savefig(
    "task4_clusters.png",
    dpi=150
)

plt.close()


print("\nCluster visualization saved as:")
print("task4_clusters.png")


# =====================================================
# 14. Save clustered dataset
# =====================================================

df.to_csv(
    "netflix_clustered_dataset.csv",
    index=False
)

print("\nClustered dataset saved as:")
print("netflix_clustered_dataset.csv")


# =====================================================
# 15. Complete
# =====================================================

print("\n==============================================")
print("TASK 4 COMPLETED")
print("==============================================")