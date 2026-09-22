# Netflix ML Internship Project

## Project Overview

This project analyzes the Netflix titles dataset using Python, data analysis, visualization, and machine learning techniques.

The project demonstrates practical skills in data preprocessing, exploratory data analysis, supervised learning, model evaluation, recommendation systems, and unsupervised learning.

Four internship tasks were completed as part of the project.

---

## Dataset

The project uses `Dataset.csv`, which contains information about Netflix movies and TV shows.

The dataset contains **8,790 titles**.

Important features include:

* Title
* Type
* Release year
* Rating
* Listed genres/categories
* Country
* Duration
* Description

---

# Completed Tasks

## Task 1 — Data Exploration

The Netflix dataset was loaded and examined using Python and pandas.

The analysis included:

* Dataset dimensions
* Data types
* Missing-value analysis
* Rating distribution
* Basic descriptive statistics
* Exploration of Netflix content characteristics

Main files:

```text
data_check.py
explore_data.py
```

---

## Task 2 — Audience Rating Classification

A machine learning classification workflow was developed to predict Netflix audience ratings using available dataset features.

The workflow included:

* Data preprocessing
* Feature engineering
* Model preparation
* Model training
* Model evaluation
* Confusion matrix visualization

Main files:

```text
task2.py
task2_confusion_matrix.png
```

The confusion matrix provides a visual representation of the model's classification performance across the rating categories.

---

## Task 3 — Classification Analysis and Evaluation

The classification workflow was further analyzed using model evaluation and exploratory analysis.

The task includes:

* Classification analysis
* Evaluation metrics
* Data exploration
* Visualization of classification results

Main files:

```text
task3.py
task3_explore.py
task3_confusion_matrix.png
```

The confusion matrix was generated to examine the distribution of correct and incorrect predictions across the classification categories.

---

## Task 4 — Netflix Content Segmentation

K-Means clustering was used to group Netflix titles into **five content clusters**.

The clustering workflow combined information from several Netflix dataset features, including:

* `type`
* `rating`
* `listed_in`
* `release_year`

Text-based features were transformed using **TF-IDF**, while release year was standardized before being incorporated into the clustering workflow.

### Clustering Model

```text
Algorithm: K-Means
Number of clusters: 5
Random state: 42
```

### Clustering Results

| Cluster | Number of Titles | Average Release Year |
| ------: | ---------------: | -------------------: |
|       0 |            4,303 |               2017.5 |
|       1 |              479 |               1995.6 |
|       2 |            1,447 |               2009.3 |
|       3 |            2,382 |               2018.0 |
|       4 |              179 |               1971.8 |

The calculated silhouette score was:

```text
0.1393
```

The score suggests that the five clusters have considerable overlap. Nevertheless, the clustering process provides a useful segmentation of the Netflix titles based on the selected features.

### Task 4 Outputs

```text
task4_clusters.png
netflix_clustered_dataset.csv
```

`task4_clusters.png` provides a visualization of the resulting clusters, while `netflix_clustered_dataset.csv` contains the cluster assignment for each Netflix title.

---

# Recommendation System

The project also contains a content recommendation component.

Main files:

```text
recommendation.py
evaluate_recommender.py
```

These scripts support recommendation functionality and evaluation of the recommendation workflow.

---

# Project Structure

```text
Netflix-ML-Internship/
│
├── Dataset.csv
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
├── evaluate_recommender.py
│
├── README.md
└── .gitignore
```

> The Python virtual environment is intentionally not included in the repository. It should be created locally when setting up the project.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* SciPy

Machine-learning and data-analysis techniques used include:

* Exploratory Data Analysis
* Data preprocessing
* Feature engineering
* TF-IDF
* Supervised classification
* Model evaluation
* K-Means clustering
* Truncated SVD
* Data visualization

---

# Machine Learning Approaches

## Supervised Learning

Supervised machine learning was used for the audience-rating classification tasks.

The workflow included feature preparation, model training, prediction, and evaluation using classification metrics and confusion matrices.

## Unsupervised Learning

Unsupervised machine learning was used for Netflix content segmentation.

K-Means clustering was applied to group titles according to selected content and metadata features.

---

# Project Outputs

The project produces several outputs that provide evidence of the analysis and machine-learning workflows:

```text
task2_confusion_matrix.png
task3_confusion_matrix.png
task4_clusters.png
netflix_clustered_dataset.csv
```

These outputs can be used to examine classification performance and the resulting Netflix content clusters.

---

# How to Run the Project

## 1. Clone the repository

```bash
git clone https://github.com/zelealem1/Netflix-ML-Internship.git
```

## 2. Open the project directory

```bash
cd Netflix-ML-Internship
```

## 3. Create a virtual environment

On Windows:

```powershell
python -m venv .venv
```

## 4. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, the Python executable can be run directly:

```powershell
.\.venv\Scripts\python.exe task2.py
```

## 5. Install the required libraries

```powershell
pip install pandas numpy matplotlib scikit-learn scipy
```

## 6. Run the analysis scripts

For example:

```powershell
python data_check.py
python explore_data.py
python task2.py
python task3.py
python task4.py
```

The recommendation scripts can also be executed when working with the recommendation component:

```powershell
python recommendation.py
python evaluate_recommender.py
```

---

# Project Purpose

The purpose of this project is to demonstrate practical application of data science and machine learning techniques to a real-world entertainment dataset.

The project demonstrates experience in:

* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Supervised machine learning
* Unsupervised machine learning
* Model evaluation
* Data visualization
* Recommendation systems
* Python programming
* Project documentation

---

# Author

**Zelealem Zeleke**

Netflix ML Internship Project

GitHub:

```text
https://github.com/zelealem1/Netflix-ML-Internship
```

---

## Project Status

**Completed: 4 internship tasks**

The project is maintained as an internship portfolio and submission project.
