import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)


# =====================================================
# TASK 3 - NETFLIX AUDIENCE RATING CLASSIFICATION
# =====================================================

print("==============================================")
print("NETFLIX AUDIENCE RATING CLASSIFICATION")
print("==============================================")


# =====================================================
# 1. Load dataset
# =====================================================

df = pd.read_csv("Dataset.csv")

print("\nDataset loaded successfully!")
print("Total records:", len(df))


# =====================================================
# 2. Examine rating distribution
# =====================================================

print("\nOriginal rating distribution:")
print(df["rating"].value_counts())


# =====================================================
# 3. Group extremely rare ratings
# =====================================================

rating_counts = df["rating"].value_counts()

rare_ratings = rating_counts[
    rating_counts < 20
].index

df["rating_target"] = df["rating"].apply(
    lambda x: "Other" if x in rare_ratings else x
)

print("\nRare categories grouped into 'Other':")
print(list(rare_ratings))

print("\nNew target distribution:")
print(df["rating_target"].value_counts())


# =====================================================
# 4. Select features
# =====================================================

features = [
    "type",
    "director",
    "country",
    "listed_in",
    "duration",
    "release_year"
]

X = df[features].copy()
y = df["rating_target"].copy()


# =====================================================
# 5. Handle missing values
# =====================================================

categorical_features = [
    "type",
    "director",
    "country",
    "listed_in",
    "duration"
]

numeric_features = [
    "release_year"
]

for column in categorical_features:
    X[column] = X[column].fillna("Unknown")


# =====================================================
# 6. Split into training and testing data
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
# 7. Encode categorical variables
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
# 8. Decision Tree model
# =====================================================

decision_tree_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

decision_tree_pipeline.fit(
    X_train,
    y_train
)

dt_predictions = decision_tree_pipeline.predict(
    X_test
)

dt_accuracy = accuracy_score(
    y_test,
    dt_predictions
)

dt_f1 = f1_score(
    y_test,
    dt_predictions,
    average="macro"
)


print("\n==============================================")
print("DECISION TREE RESULTS")
print("==============================================")

print(f"Accuracy: {dt_accuracy:.2%}")
print(f"Macro F1: {dt_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        dt_predictions,
        zero_division=0
    )
)


# =====================================================
# 9. Random Forest baseline
# =====================================================

random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        )
    ]
)

random_forest_pipeline.fit(
    X_train,
    y_train
)

rf_predictions = random_forest_pipeline.predict(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

rf_f1 = f1_score(
    y_test,
    rf_predictions,
    average="macro"
)


print("\n==============================================")
print("RANDOM FOREST RESULTS")
print("==============================================")

print(f"Accuracy: {rf_accuracy:.2%}")
print(f"Macro F1: {rf_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_predictions,
        zero_division=0
    )
)


# =====================================================
# 10. Hyperparameter tuning
# =====================================================

print("\n==============================================")
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("==============================================")

parameter_grid = {
    "model__n_estimators": [100, 150],
    "model__max_depth": [None, 20],
    "model__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=random_forest_pipeline,
    param_grid=parameter_grid,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(
    X_train,
    y_train
)

best_model = grid_search.best_estimator_

print("\nBest parameters:")
print(grid_search.best_params_)

print(
    "\nBest cross-validation Macro F1:",
    f"{grid_search.best_score_:.4f}"
)


# =====================================================
# 11. Evaluate tuned model
# =====================================================

tuned_predictions = best_model.predict(
    X_test
)

tuned_accuracy = accuracy_score(
    y_test,
    tuned_predictions
)

tuned_f1 = f1_score(
    y_test,
    tuned_predictions,
    average="macro"
)


print("\n==============================================")
print("TUNED RANDOM FOREST RESULTS")
print("==============================================")

print(f"Test Accuracy: {tuned_accuracy:.2%}")
print(f"Test Macro F1: {tuned_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        tuned_predictions,
        zero_division=0
    )
)


# =====================================================
# 12. Model comparison
# =====================================================

print("\n==============================================")
print("MODEL COMPARISON")
print("==============================================")

print(
    f"Decision Tree     - Accuracy: {dt_accuracy:.2%} "
    f"| Macro F1: {dt_f1:.4f}"
)

print(
    f"Random Forest     - Accuracy: {rf_accuracy:.2%} "
    f"| Macro F1: {rf_f1:.4f}"
)

print(
    f"Tuned Random Forest - Accuracy: {tuned_accuracy:.2%} "
    f"| Macro F1: {tuned_f1:.4f}"
)


# =====================================================
# 13. Confusion matrix for tuned model
# =====================================================

labels = sorted(
    y_test.unique()
)

confusion = confusion_matrix(
    y_test,
    tuned_predictions,
    labels=labels
)

print("\nConfusion Matrix:")
print(confusion)

display = ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=labels
)

display.plot(
    xticks_rotation=45
)

plt.title(
    "Task 3 - Tuned Random Forest Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "task3_confusion_matrix.png",
    dpi=150
)

plt.close()

print(
    "\nConfusion matrix saved as "
    "task3_confusion_matrix.png"
)


# =====================================================
# 14. Example predictions
# =====================================================

print("\n==============================================")
print("SAMPLE PREDICTIONS")
print("==============================================")

sample = X_test.head(10)

sample_predictions = best_model.predict(
    sample
)

for index, prediction in zip(
    sample.index,
    sample_predictions
):

    print(
        f"\nTitle: {df.loc[index, 'title']}"
    )

    print(
        f"Actual rating: "
        f"{df.loc[index, 'rating_target']}"
    )

    print(
        f"Predicted rating: {prediction}"
    )


print("\n==============================================")
print("TASK 3 COMPLETED")
print("==============================================")