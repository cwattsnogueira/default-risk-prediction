# Default Prediction with Deep Learning — V2

**Notebook:** `default_prediction_exploratory_v2.ipynb`  
**Author:** Carllos Watts-Nogueira  
**Date:** July 2025  
**Course:** Deep Learning with TensorFlow and Keras  
**Institution:** University of San Diego / Fullstack Academy  

---

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)  
![License](https://img.shields.io/badge/License-MIT-green.svg)  
![Status](https://img.shields.io/badge/Status-Exploratory-yellow.svg)  
![Model](https://img.shields.io/badge/Model-Sequential%20NN-lightgrey.svg)  
![Imbalance](https://img.shields.io/badge/Class%20Imbalance-Undersampling-orange.svg)  
![Evaluation](https://img.shields.io/badge/Evaluation-AUC%3D0.74%2C%20Recall%3D0.67-success.svg)

---

##  Project Overview

This project explores loan default prediction using a deep learning model built with TensorFlow and Keras. The dataset contains over **300,000 records** and **122 features** related to customer demographics, financial behavior, and credit history. The pipeline emphasizes:

- Strategic imputation  
- Class balancing  
- Feature engineering  
- Model evaluation using AUC and recall  

---

##  Project Structure

```
default-risk-prediction/
├── final_version_default_prediction/
│   ├── models/
│   ├── notebooks/
│   ├── data/
│   ├── README.md
│   └── requirements.txt
├── default_prediction_exploratory_pre_final/
│   ├── default_prediction_exploratory_pre_final_models.ipynb
│   ├── default_prediction_exploratory_pre_final_models.py
│   └── README.md
├── default_prediction_exploratory_v2/
│   ├── default_prediction_exploratory_v2.ipynb
│   ├── default_prediction_exploratory_v2.py
│   └── README.md
├── default_prediction_exploratory_v3/
│   ├── default_prediction_exploratory_v3.ipynb
│   ├── default_prediction_exploratory_v3.py
│   └── README.md
└── README.md
```

---

##  Data Cleaning & Feature Engineering

### Imputation Strategies

| Category                    | Strategy                     | Rationale                        |
|----------------------------|------------------------------|----------------------------------|
| Low Missingness (≤1%)      | Median Imputation            | Preserves distribution           |
| Moderate (10–20%)          | Mean Imputation              | Compatible with scaling          |
| Bureau Features            | Zero + Missing Flags         | Missingness may reflect behavior |
| High Missingness (>50%)    | Median + Flags               | Absence may be predictive        |
| Categorical (Light)        | Mode Imputation              | Preserves category distribution  |
| Categorical (Heavy)        | "Unknown" + Flags            | Treats missingness as signal     |
| String Normalization       | `.str.lower().strip()`       | Prevents encoding duplication    |

Additional steps:
- Dropped `SK_ID_CURR` (non-predictive)  
- Created binary flags for missingness  
- Preserved all rows (no `dropna()` used)  

---

##  Class Balancing

The original dataset had only **8.07%** default cases. To balance:

```python
from sklearn.utils import resample

non_default = df[df['TARGET'] == 0]
default = df[df['TARGET'] == 1]

non_default_sample = resample(non_default, replace=False, n_samples=len(default), random_state=42)
balanced_df = pd.concat([non_default_sample, default])
```

- Final shape: `(49,650, 181)`  
- Encoding: `pd.get_dummies(drop_first=True)`  
- Final feature set: **293 columns**

---

##  Model Architecture

```python
model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

- Optimizer: Adam  
- Loss: Binary Crossentropy  
- Metrics: Accuracy, AUC  
- EarlyStopping on `val_auc`  

---

##  Evaluation Metrics

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.68  |
| Recall     | 0.67  |
| ROC AUC    | 0.74  |

### Confusion Matrix

```
[[3422 1531]
 [1636 3341]]
```

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.68      | 0.69   | 0.68     | 4953    |
| 1     | 0.69      | 0.67   | 0.68     | 4977    |

---

##  Reflections & Open Questions

- Is selective imputation + flag creation the most robust strategy?  
- Could SMOTE or hybrid balancing improve generalization?  
- How replicable is this pipeline across different datasets?  
- What improvements are needed for deployment-readiness?

---

##  Contact

**Author:** Carllos Watts-Nogueira  
**Email:** [carlloswattsnogueira@gmail.com](mailto:carlloswattsnogueira@gmail.com)  
**LinkedIn:** [linkedin.com/in/carlloswattsnogueira](https://www.linkedin.com/in/carlloswattsnogueira/)

---
