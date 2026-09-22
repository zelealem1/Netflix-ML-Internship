import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# NETFLIX CONTENT RECOMMENDATION SYSTEM
# =====================================================

# 1. Load dataset
df = pd.read_csv("Dataset.csv")

print("Dataset loaded successfully!")
print("Total titles:", len(df))

# =====================================================
# 2. Prepare content features
# =====================================================

features = ["type", "director", "country", "rating", "listed_in"]

for feature in features:
    df[feature] = df[feature].fillna("")

# Combine the important content information
df["content_features"] = (
    df["type"] + " " +
    df["director"] + " " +
    df["country"] + " " +
    df["rating"] + " " +
    df["listed_in"]
)

print("\nContent features prepared successfully.")

# =====================================================
# 3. Convert text into numbers using TF-IDF
# =====================================================

vectorizer = TfidfVectorizer(stop_words="english")

tfidf_matrix = vectorizer.fit_transform(df["content_features"])

print("TF-IDF matrix created!")
print("Matrix shape:", tfidf_matrix.shape)

# =====================================================
# 4. Calculate cosine similarity
# =====================================================

similarity_matrix = cosine_similarity(tfidf_matrix)

print("Similarity matrix created!")

# =====================================================
# 5. Create title index
# =====================================================

indices = pd.Series(
    df.index,
    index=df["title"]
).drop_duplicates()

# =====================================================
# 6. Recommendation function
# =====================================================

def recommend(title, number_of_recommendations=10):

    if title not in indices:
        print("\nTitle not found in the dataset.")
        print("Please check the spelling and try again.")
        return

    # Index of selected title
    idx = indices[title]

    # Content profile of selected title
    source_profile = df.iloc[idx]["content_features"]

    # Similarity scores
    similarity_scores = list(
        enumerate(similarity_matrix[idx])
    )

    # Sort by highest similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []
    seen_profiles = {source_profile}

    for movie_index, score in similarity_scores:

        # Skip the selected title
        if movie_index == idx:
            continue

        recommended_title = df.iloc[movie_index]["title"]
        profile = df.iloc[movie_index]["content_features"]

        # Skip titles with exactly the same content profile
        if profile in seen_profiles:
            continue

        seen_profiles.add(profile)

        recommendations.append({
            "title": recommended_title,
            "score": score,
            "type": df.iloc[movie_index]["type"],
            "rating": df.iloc[movie_index]["rating"],
            "genre": df.iloc[movie_index]["listed_in"]
        })

        if len(recommendations) >= number_of_recommendations:
            break

    # =================================================
    # Display recommendations
    # =================================================

    print("\nRecommendations similar to:", title)
    print("=" * 70)

    for i, item in enumerate(recommendations, start=1):

        print(f"\n{i}. {item['title']}")
        print(f"   Similarity : {item['score']:.3f}")
        print(f"   Type       : {item['type']}")
        print(f"   Rating     : {item['rating']}")
        print(f"   Genre      : {item['genre']}")


# =====================================================
# 7. Interactive system
# =====================================================

user_title = input("\nEnter a Netflix title: ")

recommend(user_title, 10)