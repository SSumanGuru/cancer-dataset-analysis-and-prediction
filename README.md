# Predictive Modeling of Brain and Lungs Cancer

> **Classifying patient vital status (Alive/Dead) for brain and lung cancer using ensemble machine learning on 1.9 lakh+ SEER records — achieving 90.56% test accuracy.**

---

## Overview

Brain and lung cancers are among the leading causes of mortality worldwide. Early and accurate prediction of patient outcomes is critical in enabling timely clinical intervention and improving survival rates. This project leverages the **SEER (Surveillance, Epidemiology, and End Results)** database from the National Cancer Institute, USA, which contains records of patients diagnosed between **2018 and 2021**.

Using a combination of statistical analysis, feature engineering, and ensemble machine learning techniques, this project builds a **binary classification model to predict vital status (Alive / Dead)** of brain and lung cancer patients. The **Gradient Boosting Classifier (GBC)**, optimised through Grid Search Cross-Validation, emerged as the best-performing model with a test accuracy of **90.56%** and an F1-score of **0.91** — providing a reliable, data-driven tool for healthcare professionals to support clinical decision-making.

This project is associated with **UN SDG Goal 3 — Good Health and Well-Being**, developed as part of the MSc Applied Data Science program at SRM Institute of Science and Technology.

---

## Problem Statement

Cancer cases — particularly brain and lung cancer — are on the rise globally. The key challenges faced by healthcare providers are:

- **Lung cancer** accounts for approximately 5.69% of lifetime diagnoses, with metastasis to other organs drastically reducing survival rates
- **Brain cancer**, though rarer, has highly variable survival outcomes depending on tumor type, location, and whether the tumor extends beyond its original site
- Healthcare systems need a **proactive, data-driven tool** to classify patient vital status early, enabling more effective treatment planning and resource allocation
- Existing systems rely heavily on manual clinical assessment without leveraging the predictive power of large-scale patient data

**Objective:** To develop a machine learning–based predictive classification model using SEER cancer data that accurately determines the vital status (Alive / Dead) of brain and lung cancer patients, and to identify the most significant clinical risk factors influencing patient outcomes.

---

## About Dataset

**Source:** National Cancer Institute, United States — [SEER (Surveillance, Epidemiology, and End Results) Program](https://seer.cancer.gov/)

**Dataset Size:** 1,93,481 records × 17 attributes | Patients diagnosed between **2018 – 2021**

| Attribute | Description |
|---|---|
| **Patient ID** | Unique identifier for each patient |
| **Year of Diagnosis** | Year the patient was diagnosed (2018–2021) |
| **Age** | Age of the patient at diagnosis |
| **Sex** | Male or Female |
| **Marital Status** | Married / Single / Divorced |
| **Site Recode** | Cancer site — Lung & Bronchus or Brain |
| **Primary Site** | Specific primary location of the tumor |
| **Laterality** | Left / Right / Bilateral organ affected |
| **Summary Stage** | Cancer progression stage — Distant / Localized / Regional / Unstaged |
| **Reason** | Explanation for specific treatment decisions |
| **Radiation** | Whether radiation therapy was administered |
| **Chemotherapy** | Whether chemotherapy was administered |
| **Tumor Size** | Measured size of the tumor at diagnosis |
| **Metastasis** | Whether cancer has spread to distant organs |
| **Survival Months** | Number of months survived post-diagnosis |
| **Diagnosis to Treatment** | Time interval (days) from diagnosis to treatment start |
| **Vital Status** | Target Variable — Alive (0) / Dead (1) |

> **Target Balance:** The dataset showed near-equal class distribution — **52.6% Dead / 47.4% Alive** — confirming no significant class imbalance requiring SMOTE or IMBLearn correction.

---

## Tools & Technologies

| Category | Tools |
|---|---|
| **Programming Language** | Python 3.12 |
| **Data Manipulation** | Pandas, NumPy |
| **Statistical Analysis** | Statsmodels, SciPy |
| **Machine Learning** | Scikit-learn |
| **Ensemble Methods** | Gradient Boosting Classifier, Random Forest, AdaBoost |
| **Hyperparameter Tuning** | GridSearchCV with Stratified K-Fold (10-fold CV) |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook |
| **Dataset Source** | SEER — National Cancer Institute, USA |

---

## Methods

### 1. Data Cleaning & Preprocessing
- Imported 1,93,481 records in spreadsheet format using Pandas
- Handled missing values using mean/median imputation for numerical fields and mode for categorical fields
- Standardized inconsistent categorical labels (e.g., radiation types, treatment descriptions)
- Checked and confirmed **no class imbalance** in target variable (Dead: 52.6%, Alive: 47.4%)
- Scaled numerical features to ensure uniform range across variables

### 2. Feature Engineering
New features were derived to enhance predictive power:

```python
# Survival days derived from survival months
df['Survival days'] = df['Survival months'] * 30

# Binary surgery flag from Reason column
df['Surgery Performed'] = df['Reason'].apply(lambda x: 1 if x == 'Surgery performed' else 0)

# Binary radiation flag
df['Radiation Performed'] = df['Radiation'].apply(lambda x: 0 if x in no_categories else 1)

# Label encoding
df['Surgery Performed'] = df['Surgery Performed'].astype('category').cat.codes

# One-hot encoding for high-cardinality categoricals
df_dummy = pd.get_dummies(df, columns=['Marital status', 'Primary Site',
                          'Laterality', 'Summary Stage', 'Tumor Size',
                          'Metastasis'], drop_first=True)
```

### 3. Exploratory Data Analysis (EDA)
EDA was conducted through multiple visual and statistical lenses:

- **Distribution of Year of Diagnosis** — Almost equal patient counts diagnosed each year from 2018 to 2021
- **Survival Months vs Count per Year** — Exponential downward trend in patient count across survival months (0–10) observed in every year
- **Vital Status by Year** — 2018 had the highest proportion of dead patients (67.2%); 2021 had the lowest (29.6%)
- **Primary Site Analysis** — Upper Lobe is the primary site for lung cancer; Frontal Lobe is the primary site for brain cancer
- **Laterality Distribution** — 53.2% of cancer cases occurred on the right side; 38.9% on the left; only 1% bilateral
- **Summary Stage** — Distant (high-risk) stage patients significantly outnumber regional stage patients
- **Survival Months vs Age** — Negative correlation: longer survival is associated with younger patient age
- **Vital Status by Age** — Alive patients show a flat trend against survival months; Dead patients show an initial negative correlation, then a positive correlation at extended survival periods
- **Correlation Heatmap** — Identified moderately correlated variables including survival days (−0.43 with vital status) and year of diagnosis (−0.27 with vital status)

### 4. Feature Selection — Random Forest Importance
Random Forest was used as a feature selector by ranking features based on **mean decrease in accuracy (permutation importance)**. Features with importance > 0.02 (base threshold) collectively accounted for **99.16% predictive power**.

**Top 5 Most Important Features:**

| Feature | Importance (%) |
|---|---|
| Survival in Days | 28.83% |
| Year of Diagnosis | 18.51% |
| Diagnosis to Treatment | 8.64% |
| Age | 8.60% |
| Surgery Performed | 4.93% |

### 5. Model Building

Three ensemble models were developed and compared using **10-fold Stratified Cross-Validation**. The dataset was split 80% training / 20% testing.

**Random Forest Classifier**
- Ensemble of decision trees trained via bootstrap aggregation (bagging)
- Each tree votes independently; majority vote determines final class
- Chosen for its ability to handle high-dimensional clinical data and reduce overfitting through ensemble averaging
- `n_estimators=100`, `class_weight='balanced'`, `random_state=42`
- Cross-validation accuracy: **89.78%**

**Gradient Boosting Classifier (GBC)**
- Sequential ensemble — each tree corrects residual errors of the previous tree through gradient descent
- Chosen for its superior ability to capture complex, non-linear interactions between clinical risk factors such as tumor stage, metastasis, and treatment history
- Initial: `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, `subsample=0.8`
- Cross-validation accuracy: **90.10%** — best performing model, selected for hyperparameter tuning

**AdaBoost Classifier**
- Adaptive Boosting — assigns higher weights to misclassified patients in each round, forcing subsequent models to focus on hard cases
- Chosen because patients with ambiguous clinical profiles (e.g., intermediate survival months, borderline staging) are typically the most misclassified; AdaBoost's adaptive weighting directly addresses this
- `base_estimator=DecisionTreeClassifier(max_depth=3)`, `n_estimators=100`, `learning_rate=0.1`
- Cross-validation accuracy: **90.00%**

### 6. Hyperparameter Tuning — Grid Search CV
Grid Search CV was applied to the GBC model (best cross-validation performer):

```python
param_grid = {
    'n_estimators': [100, 150],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}
grid_search = GridSearchCV(GBC_classifier, param_grid,
                           cv=cv, scoring=scoring_metric, n_jobs=-1)
```
Best estimator parameters applied to final test evaluation.

---

## Key Insights

-  **Survival in Days is the strongest predictor** of vital status with 28.83% feature importance — longer survival time is strongly linked to alive status
-  **Year of Diagnosis matters** — 2018 had the highest mortality rate (67.2% dead); 2021 had the lowest (29.6%), suggesting improvements in treatment or reporting
-  **Upper Lobe (Lung) and Frontal Lobe (Brain)** are the primary sites with the highest cancer incidence in their respective organ categories
-  **53.2% of cancers occur on the right side** of the body; bilateral occurrence is rare at just 1%
-  **Distant stage (high-risk) cancer patients significantly outnumber regional stage patients** — indicating most patients are diagnosed late
-  **Negative correlation between Age and Survival Months** — older patients tend to have shorter survival durations
-  **Radiation and Chemotherapy show a moderate positive correlation (0.34)** — patients who receive radiation are more likely to also receive chemotherapy
-  **Surgery Performed negatively correlates with Vital Status (−0.34)** — patients who underwent surgery had relatively better survival outcomes

---

## Model Output

### Classification Report — Gradient Boost Classifier (Final Model)

```
Test Accuracy GBC Classifier : 90.557

Confusion Matrix GBC Classifier :
  [[16990  1372]
   [ 2281 18042]]

Classification Report GBC Classifier :

                precision    recall    f1-score    support

           0       0.88       0.93       0.90       18362
           1       0.93       0.89       0.91       20323

    accuracy                             0.91       38685
   macro avg       0.91       0.91       0.91       38685
weighted avg       0.91       0.91       0.91       38685
```

### EDA Visualizations Generated
-  Distribution of Target Variable Labels (Vital Status Pie Chart — 52.6% Dead / 47.4% Alive)
-  Distribution of Year of Diagnosis (2018–2021 Histogram)
-  Survival Months vs Count of Patients — per year (2018, 2019, 2020, 2021)
-  Distribution of Vital Status by Year of Diagnosis (Stacked Bar Chart)
-  Count of Cases with respect to Survival Months (Exponential Decay Curve)
-  Count of Cases for Primary Site (Upper Lobe, Frontal Lobe, etc.)
-  Distribution of Laterality (Right: 53.2%, Left: 38.9%, Bilateral: 1%)
-  Count of Cases for Summary Stage (Distant, Localized, Regional, Unstaged)
-  Count of Cases for Site Recode (Lung & Bronchus vs Brain)
-  Survival Months vs Age of Patients
-  Survival Months vs Age by Vital Status (Alive vs Dead trend comparison)
-  Correlation Heatmap with Vital Status

---


### Folder Structure
```
brain-lung-cancer-prediction/
│
├── data/
│   ├── raw/                              # Original SEER dataset (193,481 records × 17 cols)
│   └── processed/                        # Cleaned and feature-engineered dataset
│
├── notebooks/
│   ├── cancer-data-codes.ipynb
│   ├── cancer-data-codes.html
│
├── outputs/
│   ├── vital_status_predictions.csv
│   ├── feature_importance_chart.png
│   └── classification_report.txt
│
├── report/
│   └── project_report.pdf
│
└── README.md
```

---

## Results & Conclusion

### Model Comparison

| Model | Cross-Val Accuracy | Test Accuracy | F1-Score | AUC |
|---|---|---|---|---|
| Random Forest | 89.78% | — | — | — |
| AdaBoost | 90.00% | — | — | — |
| **Gradient Boosting** | **90.10%** | **90.56%** | **0.91** | **Best** |

>  Grid Search CV hyperparameter tuning on GBC improved test accuracy to **90.56%** with a weighted F1-score of **0.91** across both classes.

### Clinical & Research Impact

-  **Gradient Boosting Classifier** identified as the most reliable model for predicting vital status in brain and lung cancer patients
-  **Survival in Days (28.83%)** and **Year of Diagnosis (18.51%)** are the two most critical predictors — supporting the case for early, timely intervention
-  **Diagnosis-to-treatment interval** ranked as the 3rd most important feature — reinforcing that faster treatment commencement is associated with better outcomes
- **EDA revealed meaningful clinical patterns** — right-side laterality dominance, exponential survival decay in early months, and year-on-year improvement in survival rates from 2018 to 2021
-  The model's **99.16% combined predictive power** across selected features demonstrates robust signal extraction from clinical data

### Conclusion

> This project demonstrates that ensemble learning — particularly Gradient Boosting with Grid Search hyperparameter tuning — can accurately classify brain and lung cancer patient vital status with **90.56% test accuracy and 0.91 F1-score**. Feature importance analysis revealed that survival duration, diagnosis year, and time-to-treatment are the most influential predictors. Combined with rich EDA insights, this system provides healthcare professionals with a reliable, data-driven foundation for improving treatment planning, early intervention, and ultimately — patient survival outcomes.

---

## Academic Details

| Field | Details |
|---|---|
| **Institution** | SRM Institute of Science and Technology, Kattankulathur |
| **Program** | MSc Applied Data Science |
| **Submission** | April 2025 |

---

## 👤 Author

**Shraddha Suman Guru**
MSc Applied Data Science | SRM Institute of Science and Technology
📧 ssumanguru2000@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/ssumanguru) | [GitHub](https://github.com/SSumanGuru)

---
