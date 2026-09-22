# Netflix ML Internship Project

## Project Overview

This project applies machine learning and data analysis techniques to a Netflix titles dataset containing **8,790 records**. The project covers data exploration, content type classification, audience rating classification, and content segmentation using clustering.

The work was completed using Python, Pandas, NumPy, Scikit-learn, Matplotlib, and SciPy.

---

## Dataset

The dataset contains **8,790 Netflix titles** and **10 columns**:

* `show_id`
* `type`
* `title`
* `director`
* `country`
* `date_added`
* `release_year`
* `rating`
* `duration`
* `listed_in`

### Dataset Summary

* Total records: **8,790**
* Total columns: **10**
* Movies: **6,126**
* TV Shows: **2,664**

---

# Task 1: Netflix Data Exploration

## Objective

The first task focused on understanding the Netflix dataset through exploratory data analysis.

The dataset was checked for its structure, columns, data types, content types, ratings, genres, and release-year distribution.

## Data Validation

The dataset contains:

* **8,790 rows**
* **10 columns**
* 9 string columns
* 1 integer column (`release_year`)

The dataset check confirmed that all 8,790 records contained values in the displayed columns.

## Content Type Distribution

| Content Type | Number of Titles |
| ------------ | ---------------: |
| Movie        |            6,126 |
| TV Show      |            2,664 |

## Most Common Ratings

| Rating | Number of Titles |
| ------ | ---------------: |
| TV-MA  |            3,205 |
| TV-14  |            2,157 |
| TV-PG  |              861 |
| R      |              799 |
| PG-13  |              490 |
| TV-Y7  |              333 |
| TV-Y   |              306 |
| PG     |              287 |
| TV-G   |              220 |
| NR     |               79 |

## Most Common Genre Combinations

The most frequent genre combinations included:

1. Dramas, International Movies — 362
2. Documentaries — 359
3. Stand-Up Comedy — 334
4. Comedies, Dramas, International Movies — 274
5. Dramas, Independent Movies, International Movies — 252
6. Kids' TV — 219
7. Children & Family Movies — 215
8. Children & Family Movies, Comedies — 201
9. Documentaries, International Movies — 186
10. Dramas, International Movies, Romantic Movies — 180

## Recent Release-Year Distribution

| Release Year | Number of Titles |
| ------------ | ---------------: |
| 2021         |              592 |
| 2020         |              953 |
| 2019         |            1,030 |
| 2018         |            1,146 |
| 2017         |            1,030 |
| 2016         |              901 |
| 2015         |              555 |
| 2014         |              352 |
| 2013         |              286 |
| 2012         |              236 |

### Task 1 Output

The exploratory analysis provided a basic understanding of Netflix's content mix, audience ratings, genre combinations, and release-year distribution.

Files:

* `data_check.py`
* `explore_data.py`

---

# Task 2: Netflix Content Type Prediction

## Objective

The second task focused on predicting whether a Netflix title is a **Movie** or **TV Show** using machine learning.

The dataset was divided into:

* Training records: **7,032**
* Testing records: **1,758**

Three classification models were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest

## Model Results

| Model               | Accuracy |
| ------------------- | -------: |
| Logistic Regression |   97.72% |
| Decision Tree       |   98.75% |
| Random Forest       |   98.46% |

The Decision Tree achieved a test accuracy of **98.75%** and was selected as the demonstration model.

### Decision Tree Classification Performance

| Class   | Precision | Recall | F1-score |
| ------- | --------: | -----: | -------: |
| Movie   |      0.99 |   1.00 |     0.99 |
| TV Show |      0.99 |   0.97 |     0.98 |

### Decision Tree Confusion Matrix

```text
[[1219    6]
 [  16  517]]
```

This shows that the model correctly classified most Movies and TV Shows in the test set.

### Sample Predictions

| Title                              | Actual  | Predicted |
| ---------------------------------- | ------- | --------- |
| Coffee for All                     | Movie   | Movie     |
| An American Tail: Fievel Goes West | Movie   | Movie     |
| Joy                                | Movie   | Movie     |
| Marco Polo: One Hundred Eyes       | Movie   | Movie     |
| Happyish                           | TV Show | TV Show   |

### Task 2 Output

* `task2.py`
* `task2_confusion_matrix.png`

### Technical Note

During Logistic Regression training, Scikit-learn produced a convergence warning indicating that the optimizer reached the configured iteration limit. The model still produced the reported test results. The Decision Tree and Random Forest models completed without this reported convergence warning.

---

# Task 3: Netflix Audience Rating Classification

## Objective

The third task focused on predicting the **audience rating** of Netflix titles.

The original dataset contained several rating categories, including some categories with very few records. Three rare categories were grouped into a single `Other` class:

* `TV-Y7-FV`
* `NC-17`
* `UR`

This resulted in **12 target classes**.

## Original Rating Distribution

| Rating   | Records |
| -------- | ------: |
| TV-MA    |   3,205 |
| TV-14    |   2,157 |
| TV-PG    |     861 |
| R        |     799 |
| PG-13    |     490 |
| TV-Y7    |     333 |
| TV-Y     |     306 |
| PG       |     287 |
| TV-G     |     220 |
| NR       |      79 |
| G        |      41 |
| TV-Y7-FV |       6 |
| NC-17    |       3 |
| UR       |       3 |

After grouping the three rare categories, the `Other` class contained **12 records**.

## Data Split

* Training records: **7,032**
* Testing records: **1,758**

## Model Results

| Model               | Accuracy | Macro F1 |
| ------------------- | -------: | -------: |
| Decision Tree       |   45.39% |   0.3299 |
| Random Forest       |   49.20% |   0.3711 |
| Tuned Random Forest |   48.46% |   0.3732 |

The tuned Random Forest achieved the highest reported **Macro F1 score of 0.3732**, while the standard Random Forest achieved the highest reported **accuracy of 49.20%**.

## Hyperparameter Tuning

The Random Forest model was tuned using GridSearchCV with 3-fold cross-validation.

Best parameters:

```text
model__max_depth = None
model__min_samples_split = 5
model__n_estimators = 100
```

Best cross-validation Macro F1:

```text
0.3535
```

## Tuned Random Forest Results

| Metric        | Result |
| ------------- | -----: |
| Test Accuracy | 48.46% |
| Test Macro F1 | 0.3732 |

### Classification Performance

| Rating | Precision | Recall | F1-score |
| ------ | --------: | -----: | -------: |
| G      |      0.36 |   0.50 |     0.42 |
| NR     |      0.12 |   0.19 |     0.14 |
| Other  |      0.00 |   0.00 |     0.00 |
| PG     |      0.51 |   0.59 |     0.54 |
| PG-13  |      0.37 |   0.50 |     0.42 |
| R      |      0.41 |   0.57 |     0.48 |
| TV-14  |      0.51 |   0.51 |     0.51 |
| TV-G   |      0.17 |   0.23 |     0.19 |
| TV-MA  |      0.68 |   0.52 |     0.59 |
| TV-PG  |      0.25 |   0.26 |     0.26 |
| TV-Y   |      0.44 |   0.56 |     0.49 |
| TV-Y7  |      0.42 |   0.43 |     0.43 |

The results show that performance varied substantially across rating categories. The very small `Other` class had only two test examples and was not correctly classified.

### Sample Predictions

| Title                               | Actual Rating | Predicted Rating |
| ----------------------------------- | ------------- | ---------------- |
| Robocar Poli                        | TV-Y          | TV-Y             |
| The Super Parental Guardians        | TV-14         | TV-14            |
| Queen Sono                          | TV-MA         | TV-MA            |
| Sparta                              | TV-MA         | TV-MA            |
| Ajaibnya Cinta                      | TV-PG         | TV-PG            |
| Prince Jai Aur Dumdaar Viru         | TV-Y          | TV-Y7            |
| Chasing Cameron                     | TV-14         | TV-PG            |
| 2012                                | PG-13         | PG-13            |
| The Witch                           | R             | R                |
| Little Singham: Legend of Dugabakka | TV-Y7         | TV-Y             |

### Task 3 Confusion Matrix

The confusion matrix was generated and saved as:

`task3_confusion_matrix.png`

The classification task is affected by the uneven distribution of rating categories, particularly the very small classes such as `G`, `NR`, and `Other`.

### Task 3 Output

* `task3.py`
* `task3_explore.py`
* `task3_confusion_matrix.png`

---

# Task 4: Netflix Content Segmentation

## Objective

The fourth task used unsupervised machine learning to segment Netflix titles into groups with similar characteristics.

The clustering process used text-based features together with release-year information and K-Means clustering.

## Feature Preparation

Text feature matrix:

```text
(8790, 55)
```

Final clustering feature matrix:

```text
(8790, 56)
```

K-Means was used to create **5 clusters**.

## Clustering Evaluation

Silhouette Score:

```text
0.1393
```

## Cluster Sizes

| Cluster | Number of Titles |
| ------- | ---------------: |
| 0       |            4,303 |
| 1       |              479 |
| 2       |            1,447 |
| 3       |            2,382 |
| 4       |              179 |

## Cluster Characteristics

### Cluster 0

* Titles: **4,303**
* Content type: 4,303 Movies
* Average release year: **2017.5**
* Common ratings include TV-MA, TV-14, R, and TV-PG.
* Common genres include Documentaries, Stand-Up Comedy, and international drama combinations.

Sample titles:

* Dick Johnson Is Dead
* Confessions of an Invisible Girl
* The Starling
* Motu Patlu in the Game of Zones
* Je Suis Karl

### Cluster 1

* Titles: **479**
* Movies: 424
* TV Shows: 55
* Average release year: **1995.6**
* Common ratings include R, TV-14, PG-13, TV-MA, and PG.

Sample titles:

* Sankofa
* Jeans
* Minsara Kanavu
* Avvai Shanmughi
* Jaws: The Revenge

### Cluster 2

* Titles: **1,447**
* Movies: 1,230
* TV Shows: 217
* Average release year: **2009.3**
* Common ratings include TV-14, TV-MA, R, PG-13, and TV-PG.
* Common genres include international dramas, comedies, and children's/family content.

Sample titles:

* Motu Patlu in Wonderland
* Motu Patlu: Mission Moon
* Grown Ups
* Dark Skies
* Paranoia

### Cluster 3

* Titles: **2,382**
* Content type: 2,382 TV Shows
* Average release year: **2018.0**
* Common ratings include TV-MA, TV-14, TV-PG, TV-Y, and TV-Y7.
* Common genres include Kids' TV, crime TV shows, international TV shows, TV dramas, and reality TV.

Sample titles:

* Ganglands
* Midnight Mass
* The Great British Baking Show
* Jailbirds New Orleans
* Crime Stories: India Detectives

### Cluster 4

* Titles: **179**
* Movies: 169
* TV Shows: 10
* Average release year: **1971.8**
* Common genres include classic movies, dramas, documentaries, and action/adventure.

Sample titles:

* Jaws
* Jaws 2
* Jaws 3
* Blade Runner: The Final Cut
* Once Upon a Time in America

## Task 4 Outputs

The clustering visualization was saved as:

`task4_clusters.png`

The complete clustered dataset was saved as:

`netflix_clustered_dataset.csv`

Additional file:

* `task4.py`

---

# Additional Recommendation Component

In addition to the four main tasks, the project contains a content recommendation component.

The recommendation system uses the available Netflix metadata to identify titles with similar characteristics.

Files:

* `recommendation.py`
* `evaluate_recommender.py`

This component is included as an additional project feature and is separate from the four completed internship tasks.

---

# Project Structure

```text
Netflix-ML-Internship/
│
├── Dataset.csv
├── README.md
├── .gitignore
│
├── data_check.py
├── explore_data.py
│
├── task2.py
├── task2_confusion_matrix.png
│
├── task3.py
├── task3_explore.py
├── task3_confusion_matrix.png
│
├── task4.py
├── task4_clusters.png
├── netflix_clustered_dataset.csv
│
├── recommendation.py
└── evaluate_recommender.py
```

The local `.venv` environment is excluded from version control through `.gitignore`.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* SciPy

### Machine Learning Techniques

* Exploratory Data Analysis
* Logistic Regression
* Decision Tree Classification
* Random Forest Classification
* GridSearchCV Hyperparameter Tuning
* TF-IDF/Text Feature Extraction
* K-Means Clustering
* Silhouette Score Evaluation

---

# Running the Project

The project uses a Python virtual environment located in `.venv`.

From the project directory, run the scripts using:

```powershell
.\.venv\Scripts\python.exe data_check.py
.\.venv\Scripts\python.exe explore_data.py
.\.venv\Scripts\python.exe task2.py
.\.venv\Scripts\python.exe task3.py
.\.venv\Scripts\python.exe task4.py
```

Using the virtual-environment Python executable ensures that the required packages are available.

---

# Project Results Summary

| Task   | Project Area                   | Main Result                                             |
| ------ | ------------------------------ | ------------------------------------------------------- |
| Task 1 | Data Exploration               | 8,790 Netflix titles analyzed                           |
| Task 2 | Content Type Classification    | 98.75% Decision Tree accuracy                           |
| Task 3 | Audience Rating Classification | 49.20% Random Forest accuracy; 0.3732 tuned RF Macro F1 |
| Task 4 | Content Segmentation           | 5 K-Means clusters; silhouette score 0.1393             |

---

# Key Observations

1. The dataset contains substantially more Movies than TV Shows.
2. TV-MA and TV-14 are the most frequent audience ratings.
3. Movie/TV Show classification achieved high test accuracy with all three evaluated models.
4. Audience rating classification was more challenging because there are many rating categories and substantial class imbalance.
5. Hyperparameter tuning slightly improved Macro F1 for the Random Forest model.
6. K-Means identified five content groups with different content-type, rating, genre, and release-year characteristics.
7. The clustering silhouette score of 0.1393 indicates that the identified groups have some overlap, so the clusters should be interpreted as exploratory segments rather than completely separated categories.

---

# Limitations

* Some audience-rating categories contain relatively few examples.
* The `Other` category contains only 12 records in the complete dataset.
* Audience rating classification therefore has weaker performance on several minority classes.
* The clustering silhouette score is relatively low, indicating overlapping clusters.
* The dataset represents a particular snapshot of Netflix titles and should not automatically be treated as a current catalog.

---

# Conclusion

This project demonstrates an end-to-end machine learning workflow using a Netflix titles dataset.

The work began with data validation and exploratory analysis, followed by supervised classification of content type and audience rating. The project then applied unsupervised learning to segment titles into content groups.

The strongest classification result was obtained in the Movie vs TV Show prediction task, where the Decision Tree achieved **98.75% test accuracy**. Audience rating prediction was more challenging, with the standard Random Forest achieving **49.20% accuracy** and the tuned Random Forest achieving a **0.3732 Macro F1 score**. The clustering task produced five exploratory segments with a **0.1393 silhouette score**.

Overall, the project demonstrates practical use of data preprocessing, exploratory analysis, classification, hyperparameter tuning, model evaluation, text feature engineering, and clustering.

---

# Author

**Zelealem Zeleke**

Netflix Machine Learning Internship Project
