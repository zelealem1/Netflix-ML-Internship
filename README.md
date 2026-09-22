\# Netflix ML Internship Project



\## Project Overview



This project analyzes the Netflix titles dataset using Python and machine learning techniques.



The project covers data exploration, audience rating classification, content recommendation, and content segmentation using unsupervised machine learning.



\## Dataset



The project uses `Dataset.csv`, containing Netflix movie and TV show information.



The dataset contains \*\*8,790 titles\*\*.



Important features include:



\* Title

\* Type

\* Release year

\* Rating

\* Listed genres/categories

\* Country

\* Duration

\* Description



\## Tasks Completed



\### Task 1 — Data Exploration



The dataset was loaded and examined using Python and pandas.



The exploration included:



\* Dataset dimensions

\* Data types

\* Missing values

\* Rating distribution

\* Basic dataset statistics



\### Task 2 — Audience Rating Classification



A machine learning classification model was developed to predict Netflix audience ratings using available dataset features.



The task included:



\* Data preprocessing

\* Feature engineering

\* Model training

\* Model evaluation

\* Confusion matrix visualization



\### Task 3 — Model Evaluation / Classification Analysis



The classification workflow was further analyzed using evaluation metrics and visualization.



The project includes:



\* `task3.py`

\* `task3\_explore.py`

\* `task3\_confusion\_matrix.png`



\### Task 4 — Netflix Content Segmentation



K-Means clustering was used to group Netflix titles into \*\*5 content clusters\*\*.



The clustering model used:



\* `type`

\* `rating`

\* `listed\_in`

\* `release\_year`



Text features were transformed using TF-IDF, while release year was standardized before being combined with the text features.



The final clustering model used:



```text

K-Means

Number of clusters: 5

Random state: 42

```



\### Clustering Results



| Cluster | Number of Titles | Average Release Year |

| ------- | ---------------: | -------------------: |

| 0       |            4,303 |               2017.5 |

| 1       |              479 |               1995.6 |

| 2       |            1,447 |               2009.3 |

| 3       |            2,382 |               2018.0 |

| 4       |              179 |               1971.8 |



The calculated silhouette score was:



```text

0.1393

```



The score indicates that the five clusters have substantial overlap, while still providing identifiable groups based on the selected features.



\## Project Files



```text

Netflix-ML-Internship/

│

├── Dataset.csv

├── data\_check.py

├── explore\_data.py

├── task2.py

├── task2\_confusion\_matrix.png

├── task3.py

├── task3\_explore.py

├── task3\_confusion\_matrix.png

├── task4.py

├── task4\_clusters.png

├── netflix\_clustered\_dataset.csv

├── recommendation.py

├── evaluate\_recommender.py

├── README.md

└── .venv/

```



\## Technologies Used



\* Python

\* Pandas

\* NumPy

\* Matplotlib

\* Scikit-learn

\* SciPy

\* TF-IDF

\* K-Means Clustering

\* Truncated SVD



\## Machine Learning Techniques



The project demonstrates both supervised and unsupervised machine learning.



\### Supervised Learning



Used for audience rating classification.



\### Unsupervised Learning



K-Means clustering was used for Netflix content segmentation.



\## Task 4 Output



The clustering process generated:



\* `task4\_clusters.png` — visualization of the five content clusters

\* `netflix\_clustered\_dataset.csv` — dataset containing the assigned cluster for each title



\## Project Purpose



The purpose of this project is to demonstrate practical skills in:



\* Data preprocessing

\* Exploratory data analysis

\* Feature engineering

\* Machine learning

\* Model evaluation

\* Data visualization

\* Unsupervised learning

\* Python programming

\* Project documentation



