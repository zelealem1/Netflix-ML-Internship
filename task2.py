import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =====================================================
# NETFLIX CONTENT TYPE PREDICTION
# =====================================================

print("==============================================")
print("NETFLIX CONTENT TYPE PREDICTION")
print("==============================================")


# =====================================================
# 1. Load dataset
# =====================================================

df = pd.read_csv("Dataset.csv")

print("\nDataset loaded successfully!")
print("Number of records:", len(df))


# =====================================================
# 2. Select features
# =====================================================

features = [
    "director",
    "country",
    "rating",
    "listed_in",
    "release_year"
]

X = df[features].copy()
y = df["type"].copy()


# =====================================================
# 3. Handle missing values
# =====================================================

categorical_features = [
    "director",
    "country",
    "rating",
    "listed_in"
]

numeric_features = [
    "release_year"
]

for column in categorical_features:
    X[column] = X[column].fillna("Unknown")


# =====================================================
# 4. Split dataset into training and testing sets
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# =====================================================
# 5. Encode categorical features
# =====================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# =====================================================
# 6. Create machine learning models
# =====================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# =====================================================
# 7. Train and evaluate models
# =====================================================

results = {}

for model_name, model in models.items():

    print("\n----------------------------------------------")
    print(model_name)
    print("----------------------------------------------")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[model_name] = {
        "pipeline": pipeline,
        "predictions": predictions,
        "accuracy": accuracy
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )


# =====================================================
# 8. Display model comparison
# =====================================================

print("\n==============================================")
print("MODEL ACCURACY COMPARISON")
print("==============================================")

for model_name, result in results.items():

    print(
        f"{model_name}: "
        f"{result['accuracy']:.2%}"
    )


# =====================================================
# 9. Select highest-accuracy model
# =====================================================

selected_model_name = max(
    results,
    key=lambda name: results[name]["accuracy"]
)

selected_result = results[selected_model_name]

print("\n==============================================")
print("SELECTED MODEL FOR DEMONSTRATION")
print("==============================================")

print("Model:", selected_model_name)
print(
    "Test accuracy:",
    f"{selected_result['accuracy']:.2%}"
)


# =====================================================
# 10. Confusion matrix
# =====================================================

confusion = confusion_matrix(
    y_test,
    selected_result["predictions"],
    labels=["Movie", "TV Show"]
)

print("\nConfusion Matrix:")
print(confusion)


display = ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=["Movie", "TV Show"]
)

display.plot()
plt.title(
    f"Confusion Matrix - {selected_model_name}"
)
plt.tight_layout()
plt.savefig("task2_confusion_matrix.png")
plt.show()


# =====================================================
# 11. Test a few real records
# =====================================================

sample_data = X_test.head(5)

sample_predictions = selected_result["pipeline"].predict(
    sample_data
)

print("\n==============================================")
print("SAMPLE PREDICTIONS")
print("==============================================")

for index, prediction in zip(
    sample_data.index,
    sample_predictions
):

    actual = y.loc[index]
    title = df.loc[index, "title"]

    print(f"\nTitle: {title}")
    print(f"Actual type: {actual}")
    print(f"Predicted type: {prediction}")