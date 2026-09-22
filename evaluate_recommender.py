import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# NETFLIX RECOMMENDER - EVALUATION
# =====================================================

# 1. Load dataset
df = pd.read_csv("Dataset.csv")

# =====================================================
# 2. Prepare features
# =====================================================

features = ["type", "director", "country", "rating", "listed_in"]

for feature in features:
    df[feature] = df[feature].fillna("")

df["content_features"] = (
    df["type"] + " " +
    df["director"] + " " +
    df["country"] + " " +
    df["rating"] + " " +
    df["listed_in"]
)

# =====================================================
# 3. TF-IDF
# =====================================================

vectorizer = TfidfVectorizer(stop_words="english")

tfidf_matrix = vectorizer.fit_transform(
    df["content_features"]
)

# =====================================================
# 4. Similarity
# =====================================================

similarity_matrix = cosine_similarity(tfidf_matrix)

# =====================================================
# 5. Title index
# =====================================================

indices = pd.Series(
    df.index,
    index=df["title"]
).drop_duplicates()

# =====================================================
# 6. Recommendation function
# =====================================================

def get_recommendations(title, number_of_recommendations=10):

    if title not in indices:
        return []

    idx = indices[title]

    source_profile = df.iloc[idx]["content_features"]

    similarity_scores = list(
        enumerate(similarity_matrix[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []
    seen_profiles = {source_profile}

    for movie_index, score in similarity_scores:

        if movie_index == idx:
            continue

        recommended_title = df.iloc[movie_index]["title"]
        profile = df.iloc[movie_index]["content_features"]

        if profile in seen_profiles:
            continue

        seen_profiles.add(profile)

        recommendations.append({
            "title": recommended_title,
            "score": score,
            "genre": df.iloc[movie_index]["listed_in"]
        })

        if len(recommendations) >= number_of_recommendations:
            break

    return recommendations

# =====================================================
# 7. Genre overlap calculation
# =====================================================

def get_genres(genre_text):

    return set(
        genre.strip()
        for genre in genre_text.split(",")
        if genre.strip()
    )


def genre_overlap_rate(title, recommendations):

    source_row = df[df["title"] == title]

    if source_row.empty:
        return 0

    source_genres = get_genres(
        source_row.iloc[0]["listed_in"]
    )

    matching_titles = 0

    for recommendation in recommendations:

        recommended_genres = get_genres(
            recommendation["genre"]
        )

        if source_genres.intersection(recommended_genres):
            matching_titles += 1

    if len(recommendations) == 0:
        return 0

    return matching_titles / len(recommendations)

# =====================================================
# 8. Test titles
# =====================================================

test_titles = [
    "Lupin",
    "Ganglands",
    "Midnight Mass",
    "Sankofa",
    "The Starling"
]

print("==============================================")
print("NETFLIX RECOMMENDATION SYSTEM EVALUATION")
print("==============================================")

scores = []

for title in test_titles:

    recommendations = get_recommendations(title, 10)

    score = genre_overlap_rate(
        title,
        recommendations
    )

    scores.append(score)

    print(f"\nTitle: {title}")
    print(f"Genre overlap rate: {score:.2%}")

# =====================================================
# 9. Average evaluation score
# =====================================================

average_score = sum(scores) / len(scores)

print("\n==============================================")
print(f"Average genre overlap rate: {average_score:.2%}")
print("==============================================")

print("\nNote:")
print("Genre overlap rate is a simple proxy for")
print("recommendation relevance. It is not a perfect")
print("ground-truth accuracy measure.")